from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import TemplateValidationError
from .io import load_dlis, load_las
from .logfile_schema import validate_logfile_mapping
from .model import LogDocument, RasterChannel, ScalarChannel, WellDataset
from .templates import document_from_mapping


@dataclass(slots=True, frozen=True)
class LogFileSpec:
    name: str
    data_source_path: str | None
    data_source_format: str
    render_backend: str
    render_output_path: str
    render_dpi: int
    document: dict[str, Any]
    render_continuous_strip_page_height_mm: float | None = None
    render_matplotlib: dict[str, Any] = field(default_factory=dict)


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TemplateValidationError(
            f"Expected a mapping for {context}, got {type(value).__name__}."
        )
    return value


def _ensure_sequence(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TemplateValidationError(
            f"Expected a sequence for {context}, got {type(value).__name__}."
        )
    return value


def _normalized_source_format(value: Any, *, context: str) -> str:
    source_format = str(value).strip().lower()
    if source_format not in {"auto", "las", "dlis"}:
        raise TemplateValidationError(f"{context} must be one of: auto, las, dlis.")
    return source_format


def _deep_merge_config(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[str, Any] = deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge_config(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    if isinstance(base, list) and isinstance(override, list):
        return deepcopy(override)

    return deepcopy(override)


def _load_yaml_mapping(path: Path, *, context: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TemplateValidationError(f"{context} root must be a mapping.")
    return payload


def _resolve_template_inheritance(
    payload: dict[str, Any],
    *,
    base_dir: Path,
    visited_templates: set[Path] | None = None,
) -> dict[str, Any]:
    template_ref = payload.get("template")
    if template_ref is None:
        return deepcopy(payload)

    template_section = _ensure_mapping(template_ref, context="template")
    template_path_value = template_section.get("path")
    if not isinstance(template_path_value, str) or not template_path_value.strip():
        raise TemplateValidationError("template.path must be a non-empty string.")

    template_path = Path(template_path_value)
    if not template_path.is_absolute():
        template_path = (base_dir / template_path).resolve()
    else:
        template_path = template_path.resolve()

    visited = set() if visited_templates is None else set(visited_templates)
    if template_path in visited:
        raise TemplateValidationError(f"Template inheritance cycle detected at {template_path}.")
    visited.add(template_path)

    template_payload = _load_yaml_mapping(template_path, context=f"Template file {template_path}")
    resolved_template = _resolve_template_inheritance(
        template_payload,
        base_dir=template_path.parent,
        visited_templates=visited,
    )

    override = deepcopy(payload)
    override.pop("template", None)
    return _deep_merge_config(resolved_template, override)


def _validate_reference_track(reference: dict[str, Any], *, context: str) -> None:
    if "axis" in reference:
        axis = str(reference["axis"]).strip().lower()
        if axis not in {"depth", "time"}:
            raise TemplateValidationError(f"{context}.axis must be either depth or time.")
    if "unit" in reference:
        _ = str(reference["unit"])
    if "scale_ratio" in reference:
        if int(reference["scale_ratio"]) <= 0:
            raise TemplateValidationError(f"{context}.scale_ratio must be positive.")
    if "major_step" in reference and float(reference["major_step"]) <= 0:
        raise TemplateValidationError(f"{context}.major_step must be positive.")
    if "minor_step" in reference and float(reference["minor_step"]) <= 0:
        raise TemplateValidationError(f"{context}.minor_step must be positive.")
    if "values_orientation" in reference:
        orientation = str(reference["values_orientation"]).strip().lower()
        if orientation not in {"horizontal", "vertical"}:
            raise TemplateValidationError(
                f"{context}.values_orientation must be horizontal or vertical."
            )

    secondary_grid_data = reference.get("secondary_grid")
    if secondary_grid_data is not None:
        secondary_grid = _ensure_mapping(secondary_grid_data, context=f"{context}.secondary_grid")
        if "line_count" in secondary_grid and int(secondary_grid["line_count"]) <= 0:
            raise TemplateValidationError(f"{context}.secondary_grid.line_count must be positive.")

    header_data = reference.get("header")
    if header_data is not None:
        _ = _ensure_mapping(header_data, context=f"{context}.header")

    number_format_data = reference.get("number_format")
    if number_format_data is not None:
        number_format = _ensure_mapping(number_format_data, context=f"{context}.number_format")
        if "format" in number_format:
            fmt = str(number_format["format"]).strip().lower()
            if fmt not in {"automatic", "fixed", "scientific", "concise"}:
                raise TemplateValidationError(f"{context}.number_format.format is invalid.")
        if "precision" in number_format and int(number_format["precision"]) < 0:
            raise TemplateValidationError(
                f"{context}.number_format.precision must be non-negative."
            )


def _validate_layout_track(track: dict[str, Any], *, context: str) -> None:
    _ = str(track["id"])
    _ = float(track["width_mm"])
    kind = str(track.get("kind", "normal")).strip().lower()
    if kind not in {"reference", "normal", "array", "annotation", "depth", "curve", "image"}:
        raise TemplateValidationError(f"{context}.kind is invalid.")
    if "x_scale" in track:
        _ = _ensure_mapping(track["x_scale"], context=f"{context}.x_scale")
    if "grid" in track:
        _ = _ensure_mapping(track["grid"], context=f"{context}.grid")
    if "track_header" in track:
        _ = _ensure_mapping(track["track_header"], context=f"{context}.track_header")
    if kind in {"reference", "depth"} and "reference" in track:
        reference = _ensure_mapping(track["reference"], context=f"{context}.reference")
        _validate_reference_track(reference, context=f"{context}.reference")


def _validate_document_layout(layout: dict[str, Any], *, context: str) -> None:
    if "heading" in layout:
        _ = _ensure_mapping(layout["heading"], context=f"{context}.heading")
    if "comments" in layout:
        comments = _ensure_sequence(layout["comments"], context=f"{context}.comments")
        for index, item in enumerate(comments):
            _ = _ensure_mapping(item, context=f"{context}.comments[{index}]")
    if "tail" in layout:
        _ = _ensure_mapping(layout["tail"], context=f"{context}.tail")

    sections = _ensure_sequence(layout["log_sections"], context=f"{context}.log_sections")
    if not sections:
        raise TemplateValidationError(f"{context}.log_sections cannot be empty.")
    seen_section_ids: set[str] = set()
    for index, item in enumerate(sections):
        section = _ensure_mapping(item, context=f"{context}.log_sections[{index}]")
        section_id = str(section["id"])
        if section_id in seen_section_ids:
            raise TemplateValidationError(
                f"{context}.log_sections[{index}].id {section_id!r} must be unique."
            )
        seen_section_ids.add(section_id)
        if "title" in section:
            _ = str(section["title"])
        if "subtitle" in section:
            _ = str(section["subtitle"])
        if "data" in section:
            section_data = _ensure_mapping(
                section["data"],
                context=f"{context}.log_sections[{index}].data",
            )
            source_path = section_data.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise TemplateValidationError(
                    f"{context}.log_sections[{index}].data.source_path must be a non-empty string."
                )
            _normalized_source_format(
                section_data.get("source_format", "auto"),
                context=f"{context}.log_sections[{index}].data.source_format",
            )
        tracks = _ensure_sequence(
            section["tracks"], context=f"{context}.log_sections[{index}].tracks"
        )
        if not tracks:
            raise TemplateValidationError(
                f"{context}.log_sections[{index}].tracks cannot be empty."
            )
        for track_index, track_item in enumerate(tracks):
            track = _ensure_mapping(
                track_item,
                context=f"{context}.log_sections[{index}].tracks[{track_index}]",
            )
            _validate_layout_track(
                track,
                context=f"{context}.log_sections[{index}].tracks[{track_index}]",
            )


def _validate_document_bindings(
    bindings: dict[str, Any],
    *,
    context: str,
    available_sections: set[str],
) -> None:
    on_missing = str(bindings.get("on_missing", "skip")).strip().lower()
    if on_missing not in {"skip", "error"}:
        raise TemplateValidationError(f"{context}.on_missing must be either 'skip' or 'error'.")

    channels = _ensure_sequence(bindings["channels"], context=f"{context}.channels")
    if not channels:
        raise TemplateValidationError(f"{context}.channels cannot be empty.")
    for index, item in enumerate(channels):
        channel_cfg = _ensure_mapping(item, context=f"{context}.channels[{index}]")
        _ = str(channel_cfg["channel"])
        _ = str(channel_cfg["track_id"])
        if "section" in channel_cfg:
            section = str(channel_cfg["section"])
            if section not in available_sections:
                raise TemplateValidationError(
                    f"{context}.channels[{index}].section must match one of the layout section ids."
                )
        kind = str(channel_cfg.get("kind", "curve")).strip().lower()
        if kind not in {"curve", "raster"}:
            raise TemplateValidationError(f"{context}.channels[{index}].kind is invalid.")
        if kind == "raster":
            if "profile" in channel_cfg:
                profile = str(channel_cfg["profile"]).strip().lower()
                if profile not in {"generic", "vdl", "waveform"}:
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].profile must be generic, vdl, or waveform."
                    )
            if "normalization" in channel_cfg:
                normalization = str(channel_cfg["normalization"]).strip().lower()
                if normalization not in {"auto", "none", "trace_maxabs", "global_maxabs"}:
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].normalization is invalid."
                    )
            if "waveform_normalization" in channel_cfg:
                normalization = str(channel_cfg["waveform_normalization"]).strip().lower()
                if normalization not in {"auto", "none", "trace_maxabs", "global_maxabs"}:
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].waveform_normalization is invalid."
                    )
            if "show_raster" in channel_cfg and not isinstance(channel_cfg["show_raster"], bool):
                raise TemplateValidationError(
                    f"{context}.channels[{index}].show_raster must be boolean."
                )
            if "raster_alpha" in channel_cfg:
                alpha = float(channel_cfg["raster_alpha"])
                if alpha < 0 or alpha > 1:
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].raster_alpha must be between 0 and 1."
                    )
            if "clip_percentiles" in channel_cfg:
                percentiles = _ensure_sequence(
                    channel_cfg["clip_percentiles"],
                    context=f"{context}.channels[{index}].clip_percentiles",
                )
                if len(percentiles) != 2:
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].clip_percentiles must contain two values."
                    )
                low = float(percentiles[0])
                high = float(percentiles[1])
                if low < 0 or low > 100 or high < 0 or high > 100 or low >= high:
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].clip_percentiles must be increasing "
                        "values within 0..100."
                    )
        if "style" in channel_cfg:
            _ = _ensure_mapping(channel_cfg["style"], context=f"{context}.channels[{index}].style")
        if "scale" in channel_cfg:
            _ = _ensure_mapping(channel_cfg["scale"], context=f"{context}.channels[{index}].scale")
        if "value_labels" in channel_cfg:
            labels = _ensure_mapping(
                channel_cfg["value_labels"],
                context=f"{context}.channels[{index}].value_labels",
            )
            if "step" in labels and float(labels["step"]) <= 0:
                raise TemplateValidationError(
                    f"{context}.channels[{index}].value_labels.step must be positive."
                )
            if "precision" in labels and int(labels["precision"]) < 0:
                raise TemplateValidationError(
                    f"{context}.channels[{index}].value_labels.precision must be non-negative."
                )
        if "header_display" in channel_cfg:
            display = _ensure_mapping(
                channel_cfg["header_display"],
                context=f"{context}.channels[{index}].header_display",
            )
            for key in ("show_name", "show_unit", "show_limits", "show_color"):
                if key in display and not isinstance(display[key], bool):
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].header_display.{key} must be boolean."
                    )
        if "wrap" in channel_cfg:
            wrap_value = channel_cfg["wrap"]
            if isinstance(wrap_value, bool):
                pass
            else:
                wrap = _ensure_mapping(wrap_value, context=f"{context}.channels[{index}].wrap")
                if "enabled" in wrap and not isinstance(wrap["enabled"], bool):
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].wrap.enabled must be boolean."
                    )
                if "color" in wrap and not str(wrap["color"]).strip():
                    raise TemplateValidationError(
                        f"{context}.channels[{index}].wrap.color must be non-empty."
                    )
        if "colorbar" in channel_cfg:
            _parse_binding_raster_colorbar(
                channel_cfg["colorbar"],
                context=f"{context}.channels[{index}].colorbar",
            )
        if "sample_axis" in channel_cfg:
            _parse_binding_raster_sample_axis(
                channel_cfg["sample_axis"],
                context=f"{context}.channels[{index}].sample_axis",
            )
        if "waveform" in channel_cfg:
            _parse_binding_raster_waveform(
                channel_cfg["waveform"],
                context=f"{context}.channels[{index}].waveform",
            )


def _parse_binding_wrap(value: Any, *, context: str) -> tuple[bool, str | None]:
    if value is None:
        return False, None
    if isinstance(value, bool):
        return value, None

    wrap = _ensure_mapping(value, context=context)
    enabled = bool(wrap.get("enabled", True))
    color = wrap.get("color")
    if color is None:
        return enabled, None
    color_text = str(color).strip()
    if not color_text:
        raise TemplateValidationError(f"{context}.color must be non-empty.")
    return enabled, color_text


def _parse_binding_raster_colorbar(
    value: Any, *, context: str
) -> tuple[bool, str | None, str]:
    if value is None:
        return False, None, "right"
    if isinstance(value, bool):
        return value, None, "right"

    colorbar = _ensure_mapping(value, context=context)
    enabled = bool(colorbar.get("enabled", True))
    position = str(colorbar.get("position", "right")).strip().lower()
    if position not in {"right", "header"}:
        raise TemplateValidationError(f"{context}.position must be right or header.")
    label = colorbar.get("label")
    if label is None:
        return enabled, None, position
    label_text = str(label).strip()
    if not label_text:
        raise TemplateValidationError(f"{context}.label must be non-empty.")
    return enabled, label_text, position


def _parse_binding_raster_sample_axis(
    value: Any, *, context: str
) -> tuple[bool, str | None, str | None, int, float | None, float | None]:
    if value is None:
        return False, None, None, 5, None, None
    if isinstance(value, bool):
        return value, None, None, 5, None, None

    sample_axis = _ensure_mapping(value, context=context)
    enabled = bool(sample_axis.get("enabled", True))
    label = sample_axis.get("label")
    label_text: str | None = None
    if label is not None:
        label_text = str(label).strip()
        if not label_text:
            raise TemplateValidationError(f"{context}.label must be non-empty.")
    unit = sample_axis.get("unit")
    unit_text: str | None = None
    if unit is not None:
        unit_text = str(unit).strip()
        if not unit_text:
            raise TemplateValidationError(f"{context}.unit must be non-empty.")
    ticks = int(sample_axis.get("ticks", 5))
    if ticks < 2:
        raise TemplateValidationError(f"{context}.ticks must be at least 2.")
    min_value = sample_axis.get("min")
    max_value = sample_axis.get("max")
    if (min_value is None) != (max_value is None):
        raise TemplateValidationError(f"{context}.min and {context}.max must be set together.")
    axis_min = float(min_value) if min_value is not None else None
    axis_max = float(max_value) if max_value is not None else None
    if axis_min is not None and axis_max is not None and np.isclose(axis_min, axis_max):
        raise TemplateValidationError(f"{context}.min and {context}.max must differ.")
    return enabled, label_text, unit_text, ticks, axis_min, axis_max


def _parse_binding_raster_waveform(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if value is None:
        return {"enabled": False}
    if isinstance(value, bool):
        return {"enabled": value}

    waveform = _ensure_mapping(value, context=context)
    payload: dict[str, Any] = {"enabled": bool(waveform.get("enabled", True))}
    if "stride" in waveform:
        stride = int(waveform["stride"])
        if stride <= 0:
            raise TemplateValidationError(f"{context}.stride must be positive.")
        payload["stride"] = stride
    if "amplitude_scale" in waveform:
        amplitude_scale = float(waveform["amplitude_scale"])
        if amplitude_scale <= 0:
            raise TemplateValidationError(f"{context}.amplitude_scale must be positive.")
        payload["amplitude_scale"] = amplitude_scale
    if "color" in waveform:
        color = str(waveform["color"]).strip()
        if not color:
            raise TemplateValidationError(f"{context}.color must be non-empty.")
        payload["color"] = color
    if "line_width" in waveform:
        line_width = float(waveform["line_width"])
        if line_width <= 0:
            raise TemplateValidationError(f"{context}.line_width must be positive.")
        payload["line_width"] = line_width
    if "fill" in waveform:
        payload["fill"] = bool(waveform["fill"])
    if "positive_fill_color" in waveform:
        color = str(waveform["positive_fill_color"]).strip()
        if not color:
            raise TemplateValidationError(f"{context}.positive_fill_color must be non-empty.")
        payload["positive_fill_color"] = color
    if "negative_fill_color" in waveform:
        color = str(waveform["negative_fill_color"]).strip()
        if not color:
            raise TemplateValidationError(f"{context}.negative_fill_color must be non-empty.")
        payload["negative_fill_color"] = color
    if "invert_fill_polarity" in waveform:
        payload["invert_fill_polarity"] = bool(waveform["invert_fill_polarity"])
    if "max_traces" in waveform:
        max_traces = int(waveform["max_traces"])
        if max_traces <= 0:
            raise TemplateValidationError(f"{context}.max_traces must be positive.")
        payload["max_traces"] = max_traces
    return payload


def logfile_from_mapping(data: dict[str, Any]) -> LogFileSpec:
    validate_logfile_mapping(data)
    root = _ensure_mapping(data, context="logfile")
    try:
        _ = int(root["version"])
        name = str(root["name"])

        data_source_path: str | None = None
        data_source_format = "auto"
        data_section_value = root.get("data")
        if data_section_value is not None:
            data_section = _ensure_mapping(data_section_value, context="data")
            source_path = data_section.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise TemplateValidationError("data.source_path must be a non-empty string.")
            data_source_path = source_path
            data_source_format = _normalized_source_format(
                data_section.get("source_format", "auto"),
                context="data.source_format",
            )

        render_section = _ensure_mapping(root["render"], context="render")
        render_backend = str(render_section.get("backend", "matplotlib")).strip().lower()
        render_output_path = str(render_section["output_path"])
        render_dpi = int(render_section["dpi"])
        if render_dpi <= 0:
            raise TemplateValidationError("render.dpi must be positive.")
        render_matplotlib = deepcopy(
            _ensure_mapping(
                render_section.get("matplotlib", {}),
                context="render.matplotlib",
            )
        )
        if "style" in render_matplotlib:
            _ensure_mapping(render_matplotlib["style"], context="render.matplotlib.style")
        render_continuous_strip_page_height_mm_value = render_section.get(
            "continuous_strip_page_height_mm"
        )
        render_continuous_strip_page_height_mm = None
        if render_continuous_strip_page_height_mm_value is not None:
            render_continuous_strip_page_height_mm = float(
                render_continuous_strip_page_height_mm_value
            )
            if render_continuous_strip_page_height_mm <= 0:
                raise TemplateValidationError(
                    "render.continuous_strip_page_height_mm must be positive."
                )

        document = deepcopy(_ensure_mapping(root["document"], context="document"))
        layout = _ensure_mapping(document.get("layout"), context="document.layout")
        bindings = _ensure_mapping(document.get("bindings"), context="document.bindings")
        _validate_document_layout(layout, context="document.layout")
        section_ids = {
            str(_ensure_mapping(section, context="document.layout.log_sections item")["id"])
            for section in _ensure_sequence(
                layout["log_sections"], context="document.layout.log_sections"
            )
        }
        _validate_document_bindings(
            bindings,
            context="document.bindings",
            available_sections=section_ids,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TemplateValidationError("Invalid logfile configuration.") from exc

    return LogFileSpec(
        name=name,
        data_source_path=data_source_path,
        data_source_format=data_source_format,
        render_backend=render_backend,
        render_output_path=render_output_path,
        render_dpi=render_dpi,
        render_continuous_strip_page_height_mm=render_continuous_strip_page_height_mm,
        render_matplotlib=render_matplotlib,
        document=document,
    )


def load_logfile(path: str | Path) -> LogFileSpec:
    file_path = Path(path).resolve()
    payload = _load_yaml_mapping(file_path, context="Log file")
    resolved_payload = _resolve_template_inheritance(payload, base_dir=file_path.parent)
    return logfile_from_mapping(resolved_payload)


def _safe_format(template: str, values: dict[str, Any]) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_SafeDict(values))


def _resolve_text_tokens(document: dict[str, Any], dataset: WellDataset, source_path: Path) -> None:
    tokens = {str(key): value for key, value in dataset.well_metadata.items()}
    tokens["DATASET_NAME"] = dataset.name
    tokens["SOURCE_FILENAME"] = source_path.name
    tokens["SOURCE_PATH"] = str(source_path)

    name = document.get("name")
    if isinstance(name, str):
        document["name"] = _safe_format(name, tokens)

    header = document.get("header")
    if isinstance(header, dict):
        for key in ("title", "subtitle"):
            value = header.get(key)
            if isinstance(value, str):
                header[key] = _safe_format(value, tokens)
    footer = document.get("footer")
    if isinstance(footer, dict):
        lines = footer.get("lines")
        if isinstance(lines, list):
            footer["lines"] = [
                _safe_format(str(line), tokens) if isinstance(line, str) else line for line in lines
            ]


def _resolve_data_source_path(source_path: str, *, base_dir: Path | None = None) -> Path:
    resolved_path = Path(source_path)
    if not resolved_path.is_absolute():
        resolved_path = (base_dir or Path.cwd()) / resolved_path
    return resolved_path.resolve()


def _load_dataset_from_source(
    source_path: str,
    source_format: str,
    *,
    base_dir: Path | None = None,
) -> tuple[WellDataset, Path]:
    resolved_path = _resolve_data_source_path(source_path, base_dir=base_dir)
    resolved_format = _normalized_source_format(source_format, context="source_format")
    if resolved_format == "auto":
        resolved_format = resolved_path.suffix.lower().lstrip(".")

    if resolved_format == "las":
        return load_las(resolved_path), resolved_path
    if resolved_format == "dlis":
        return load_dlis(resolved_path), resolved_path
    raise TemplateValidationError(f"Unsupported data source format {resolved_format!r}.")


def _section_data_sources_for_logfile(
    spec: LogFileSpec,
) -> dict[str, tuple[str, str]]:
    layout = _ensure_mapping(spec.document["layout"], context="document.layout")
    sections = _layout_sections(layout, context="document.layout.log_sections")

    default_path = spec.data_source_path
    default_format = (
        _normalized_source_format(spec.data_source_format, context="data.source_format")
        if default_path is not None
        else "auto"
    )
    section_sources: dict[str, tuple[str, str]] = {}
    for index, section in enumerate(sections):
        section_id = str(section["id"])
        source_path: str
        source_format: str
        if "data" in section:
            section_data = _ensure_mapping(
                section["data"],
                context=f"document.layout.log_sections[{index}].data",
            )
            source_path = str(section_data["source_path"])
            source_format = _normalized_source_format(
                section_data.get("source_format", "auto"),
                context=f"document.layout.log_sections[{index}].data.source_format",
            )
        elif default_path is not None:
            source_path = default_path
            source_format = default_format
        else:
            raise TemplateValidationError(
                "Each document.layout.log_sections[*] must define data.source_path "
                "when top-level data.source_path is not set."
            )
        section_sources[section_id] = (source_path, source_format)
    return section_sources


def load_datasets_for_logfile(
    spec: LogFileSpec,
    *,
    base_dir: Path | None = None,
) -> tuple[dict[str, WellDataset], dict[str, Path]]:
    section_sources = _section_data_sources_for_logfile(spec)
    cache: dict[tuple[Path, str], tuple[WellDataset, Path]] = {}
    datasets_by_section: dict[str, WellDataset] = {}
    source_paths_by_section: dict[str, Path] = {}

    for section_id, (source_path, source_format) in section_sources.items():
        resolved_path = _resolve_data_source_path(source_path, base_dir=base_dir)
        resolved_format = _normalized_source_format(
            source_format,
            context=f"document.layout.log_sections[{section_id}].data.source_format",
        )
        if resolved_format == "auto":
            resolved_format = resolved_path.suffix.lower().lstrip(".")
        cache_key = (resolved_path, resolved_format)
        if cache_key not in cache:
            dataset, loaded_path = _load_dataset_from_source(
                str(resolved_path),
                resolved_format,
                base_dir=base_dir,
            )
            cache[cache_key] = (dataset, loaded_path)
        dataset, loaded_path = cache[cache_key]
        datasets_by_section[section_id] = dataset
        source_paths_by_section[section_id] = loaded_path

    return datasets_by_section, source_paths_by_section


def load_dataset_for_logfile(
    spec: LogFileSpec, *, base_dir: Path | None = None
) -> tuple[WellDataset, Path]:
    if spec.data_source_path is not None:
        return _load_dataset_from_source(
            spec.data_source_path,
            spec.data_source_format,
            base_dir=base_dir,
        )

    section_sources = _section_data_sources_for_logfile(spec)
    unique_sources = {source for source in section_sources.values()}
    if len(unique_sources) != 1:
        raise TemplateValidationError(
            "This logfile defines section-specific data sources. "
            "Use load_datasets_for_logfile(...) instead."
        )
    source_path, source_format = next(iter(unique_sources))
    return _load_dataset_from_source(
        source_path,
        source_format,
        base_dir=base_dir,
    )


def _build_scale(values: np.ndarray, scale_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(scale_cfg or {})
    fixed_bounds = "min" in cfg or "max" in cfg
    if fixed_bounds:
        if "min" not in cfg or "max" not in cfg:
            raise TemplateValidationError(
                "Track scale requires both min and max when fixed bounds are used."
            )
        kind = str(cfg.get("kind", "linear")).strip().lower()
        if kind == "logarithmic":
            kind = "log"
        if kind == "tangent":
            kind = "tangential"
        if kind == "auto":
            kind = "linear"
        if kind not in {"linear", "log", "tangential"}:
            raise TemplateValidationError(
                "Track scale kind must be linear, log, tangential, or auto."
            )
        scale: dict[str, Any] = {
            "kind": kind,
            "min": float(cfg["min"]),
            "max": float(cfg["max"]),
        }
        if "reverse" in cfg:
            scale["reverse"] = bool(cfg["reverse"])
        return scale

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {"kind": "linear", "min": 0.0, "max": 1.0}

    low = float(cfg.get("percentile_low", 2.0))
    high = float(cfg.get("percentile_high", 98.0))
    min_positive = float(cfg.get("min_positive", 1e-6))
    ratio_threshold = float(cfg.get("log_ratio_threshold", 200.0))
    if low >= high:
        raise TemplateValidationError("Track percentile_low must be lower than percentile_high.")

    lower, upper = np.nanpercentile(finite, [low, high])
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.1, 1.0)
        lower -= pad
        upper += pad

    requested_kind = str(cfg.get("kind", "auto")).strip().lower()
    if requested_kind == "logarithmic":
        requested_kind = "log"
    if requested_kind == "tangent":
        requested_kind = "tangential"
    if requested_kind not in {"auto", "linear", "log", "tangential"}:
        raise TemplateValidationError("Track scale kind must be auto, linear, log, or tangential.")

    scale_kind = "linear"
    if requested_kind == "log":
        scale_kind = "log"
    elif requested_kind == "tangential":
        scale_kind = "tangential"
    elif requested_kind == "auto":
        if lower > 0 and upper / max(lower, min_positive) >= ratio_threshold:
            scale_kind = "log"
    if lower > 0 and upper / max(lower, min_positive) >= ratio_threshold:
        if requested_kind != "linear":
            scale_kind = "log"
    if scale_kind == "log":
        lower = max(lower, min_positive)

    scale: dict[str, Any] = {"kind": scale_kind, "min": float(lower), "max": float(upper)}
    if "reverse" in cfg:
        scale["reverse"] = bool(cfg["reverse"])
    return scale


def _normalized_track_kind(kind: str) -> str:
    lowered = kind.strip().lower()
    if lowered == "depth":
        return "reference"
    if lowered == "curve":
        return "normal"
    if lowered == "image":
        return "array"
    return lowered


def _ordered_layout_tracks(section: dict[str, Any], *, context: str) -> list[dict[str, Any]]:
    track_items = _ensure_sequence(section["tracks"], context=f"{context}.tracks")
    indexed: list[tuple[int | None, int, dict[str, Any]]] = []
    for index, item in enumerate(track_items):
        track = deepcopy(_ensure_mapping(item, context=f"{context}.tracks[{index}]"))
        position = int(track["position"]) if "position" in track else None
        indexed.append((position, index, track))

    count = len(indexed)
    ordered: list[dict[str, Any] | None] = [None] * count
    explicit = sorted(
        (item for item in indexed if item[0] is not None),
        key=lambda item: (int(item[0] or 0), item[1]),
    )
    for position, _, track in explicit:
        assert position is not None
        slot = min(max(position, 1), count) - 1
        while slot < count and ordered[slot] is not None:
            slot += 1
        if slot >= count:
            slot = next(index for index, current in enumerate(ordered) if current is None)
        ordered[slot] = track

    for _, _, track in sorted(
        (item for item in indexed if item[0] is None),
        key=lambda item: item[1],
    ):
        slot = next(index for index, current in enumerate(ordered) if current is None)
        ordered[slot] = track

    return [track for track in ordered if track is not None]


def _layout_sections(layout: dict[str, Any], *, context: str) -> list[dict[str, Any]]:
    section_items = _ensure_sequence(layout["log_sections"], context=context)
    if not section_items:
        raise TemplateValidationError(f"{context} cannot be empty.")

    sections: list[dict[str, Any]] = []
    seen_section_ids: set[str] = set()
    for index, item in enumerate(section_items):
        section = _ensure_mapping(item, context=f"{context}[{index}]")
        section_id = str(section["id"])
        if section_id in seen_section_ids:
            raise TemplateValidationError(f"{context}[{index}].id {section_id!r} must be unique.")
        seen_section_ids.add(section_id)
        sections.append(section)
    return sections


def _build_empty_tracks_for_section(
    section: dict[str, Any], *, context: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tracks: list[dict[str, Any]] = []
    tracks_by_id: dict[str, dict[str, Any]] = {}
    for index, track_data in enumerate(_ordered_layout_tracks(section, context=context)):
        track_id = str(track_data["id"])
        if track_id in tracks_by_id:
            raise TemplateValidationError(f"{context}.tracks[{index}].id must be unique.")
        kind = _normalized_track_kind(str(track_data.get("kind", "normal")))
        built = {
            "id": track_id,
            "title": str(track_data.get("title", track_id)),
            "kind": kind,
            "width_mm": float(track_data["width_mm"]),
            "elements": [],
            "track_header": deepcopy(track_data.get("track_header", {})),
            "grid": deepcopy(track_data.get("grid", {})),
        }
        if "x_scale" in track_data:
            built["x_scale"] = deepcopy(track_data["x_scale"])
        if kind == "reference":
            built["reference"] = deepcopy(track_data.get("reference", {}))
        tracks.append(built)
        tracks_by_id[track_id] = built
    return tracks, tracks_by_id


def _binding_target_section(
    binding: dict[str, Any],
    *,
    binding_context: str,
    section_track_maps: dict[str, dict[str, dict[str, Any]]],
    track_to_sections: dict[str, list[str]],
) -> str:
    track_id = str(binding["track_id"])
    explicit_section = binding.get("section")
    if explicit_section is not None:
        section_id = str(explicit_section)
        section_tracks = section_track_maps.get(section_id)
        if section_tracks is None:
            raise TemplateValidationError(
                f"{binding_context}.section {section_id!r} does not exist in "
                "document.layout.log_sections."
            )
        if track_id not in section_tracks:
            raise TemplateValidationError(
                f"{binding_context}.track_id {track_id!r} was not found in section {section_id!r}."
            )
        return section_id

    candidate_sections = track_to_sections.get(track_id, [])
    if not candidate_sections:
        raise TemplateValidationError(
            f"{binding_context}.track_id {track_id!r} was not found in "
            "document.layout.log_sections."
        )
    if len(candidate_sections) > 1:
        joined = ", ".join(candidate_sections)
        raise TemplateValidationError(
            f"{binding_context}.track_id {track_id!r} exists in multiple sections ({joined}). "
            "Set document.bindings.channels[].section explicitly."
        )
    return candidate_sections[0]


def _build_tracks_from_layout_bindings(
    datasets_by_section: dict[str, WellDataset],
    document: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    layout = _ensure_mapping(document["layout"], context="document.layout")
    bindings = _ensure_mapping(document["bindings"], context="document.bindings")

    sections = _layout_sections(layout, context="document.layout.log_sections")
    tracks_by_section: dict[str, list[dict[str, Any]]] = {}
    section_track_maps: dict[str, dict[str, dict[str, Any]]] = {}
    track_to_sections: dict[str, list[str]] = {}
    for section_index, section in enumerate(sections):
        section_id = str(section["id"])
        section_context = f"document.layout.log_sections[{section_index}]"
        section_tracks, section_tracks_by_id = _build_empty_tracks_for_section(
            section,
            context=section_context,
        )
        tracks_by_section[section_id] = section_tracks
        section_track_maps[section_id] = section_tracks_by_id
        for track_id in section_tracks_by_id:
            track_to_sections.setdefault(track_id, []).append(section_id)

    on_missing = str(bindings.get("on_missing", "skip")).strip().lower()
    channel_items = _ensure_sequence(bindings["channels"], context="document.bindings.channels")
    channels_by_section_upper: dict[str, dict[str, Any]] = {}
    for section_id, section_dataset in datasets_by_section.items():
        channels_by_section_upper[section_id] = {
            channel.mnemonic.upper(): channel for channel in section_dataset.channels.values()
        }

    for index, item in enumerate(channel_items):
        binding = _ensure_mapping(item, context=f"document.bindings.channels[{index}]")
        binding_context = f"document.bindings.channels[{index}]"
        section_id = _binding_target_section(
            binding,
            binding_context=binding_context,
            section_track_maps=section_track_maps,
            track_to_sections=track_to_sections,
        )
        track_id = str(binding["track_id"])
        track = section_track_maps[section_id][track_id]
        by_upper = channels_by_section_upper.get(section_id)
        if by_upper is None:
            raise TemplateValidationError(f"Missing dataset for section {section_id!r}.")

        channel_name = str(binding["channel"])
        channel = by_upper.get(channel_name.upper())
        required = bool(binding.get("required", False))
        if channel is None:
            if required or on_missing == "error":
                raise TemplateValidationError(
                    f"Configured channel {channel_name!r} was not found in dataset."
                )
            continue

        track_kind = _normalized_track_kind(str(track.get("kind", "normal")))
        element_kind = str(binding.get("kind", "curve")).strip().lower()
        if element_kind == "curve":
            if not isinstance(channel, ScalarChannel):
                raise TemplateValidationError(
                    f"Binding channel {channel_name!r} is not scalar and cannot be used as curve."
                )
            if track_kind == "annotation":
                raise TemplateValidationError(
                    f"Track {track_id!r} is annotation and does not accept curve bindings."
                )
            style = deepcopy(
                _ensure_mapping(binding.get("style", {}), context=f"{binding_context}.style")
            )
            scale = _build_scale(
                channel.masked_values(),
                _ensure_mapping(binding.get("scale", {}), context=f"{binding_context}.scale"),
            )
            wrap_enabled, wrap_color = _parse_binding_wrap(
                binding.get("wrap"),
                context=f"{binding_context}.wrap",
            )
            wrap_value: bool | dict[str, Any] = wrap_enabled
            if wrap_color is not None:
                wrap_value = {"enabled": wrap_enabled, "color": wrap_color}
            element: dict[str, Any] = {
                "kind": "curve",
                "channel": channel.mnemonic,
                "label": str(binding.get("label", channel.mnemonic)),
                "style": style,
                "scale": scale,
                "wrap": wrap_value,
                "render_mode": str(binding.get("render_mode", "line")),
                "value_labels": deepcopy(
                    _ensure_mapping(
                        binding.get("value_labels", {}),
                        context=f"{binding_context}.value_labels",
                    )
                ),
                "header_display": deepcopy(
                    _ensure_mapping(
                        binding.get("header_display", {}),
                        context=f"{binding_context}.header_display",
                    )
                ),
            }
            track["elements"].append(element)
            continue

        if element_kind == "raster":
            if track_kind != "array":
                raise TemplateValidationError(
                    f"Track {track_id!r} must be array kind to accept raster bindings."
                )
            if not isinstance(channel, RasterChannel):
                raise TemplateValidationError(
                    f"Binding channel {channel_name!r} is not raster-compatible."
                )
            style = deepcopy(
                _ensure_mapping(binding.get("style", {}), context=f"{binding_context}.style")
            )
            colorbar_enabled, colorbar_label, colorbar_position = _parse_binding_raster_colorbar(
                binding.get("colorbar"),
                context=f"{binding_context}.colorbar",
            )
            (
                sample_axis_enabled,
                sample_axis_label,
                sample_axis_unit,
                sample_axis_ticks,
                sample_axis_min,
                sample_axis_max,
            ) = (
                _parse_binding_raster_sample_axis(
                    binding.get("sample_axis"),
                    context=f"{binding_context}.sample_axis",
                )
            )
            profile = str(binding.get("profile", "generic")).strip().lower()
            waveform_input = binding.get("waveform")
            waveform = _parse_binding_raster_waveform(
                waveform_input,
                context=f"{binding_context}.waveform",
            )
            if profile == "waveform" and waveform_input is None:
                waveform = {"enabled": True}
            show_raster = bool(binding.get("show_raster", profile != "waveform"))
            element = {
                "kind": "raster",
                "channel": channel.mnemonic,
                "label": str(binding.get("label", channel.mnemonic)),
                "style": style,
                "profile": profile,
                "normalization": str(binding.get("normalization", "auto")),
                "waveform_normalization": str(binding.get("waveform_normalization", "auto")),
                "interpolation": str(binding.get("interpolation", "nearest")),
                "show_raster": show_raster,
                "raster_alpha": float(binding.get("raster_alpha", 1.0)),
                "colorbar": (
                    {
                        "enabled": colorbar_enabled,
                        "label": colorbar_label,
                        "position": colorbar_position,
                    }
                    if colorbar_label is not None or colorbar_position != "right"
                    else colorbar_enabled
                ),
                "sample_axis": (
                    {
                        "enabled": sample_axis_enabled,
                        "label": sample_axis_label,
                        "unit": sample_axis_unit,
                        "ticks": sample_axis_ticks,
                        "min": sample_axis_min,
                        "max": sample_axis_max,
                    }
                    if (
                        sample_axis_label is not None
                        or sample_axis_unit is not None
                        or sample_axis_ticks != 5
                        or sample_axis_min is not None
                        or sample_axis_max is not None
                    )
                    else sample_axis_enabled
                ),
                "waveform": waveform
                if (
                    len(waveform) > 1
                    or waveform.get("enabled", False)
                )
                else waveform["enabled"],
            }
            if "color_limits" in binding:
                limits = _ensure_sequence(
                    binding["color_limits"],
                    context=f"{binding_context}.color_limits",
                )
                if len(limits) != 2:
                    raise TemplateValidationError(
                        f"{binding_context}.color_limits must contain two values."
                    )
                element["color_limits"] = [float(limits[0]), float(limits[1])]
            if "clip_percentiles" in binding:
                percentiles = _ensure_sequence(
                    binding["clip_percentiles"],
                    context=f"{binding_context}.clip_percentiles",
                )
                if len(percentiles) != 2:
                    raise TemplateValidationError(
                        f"{binding_context}.clip_percentiles must contain two values."
                    )
                element["clip_percentiles"] = [float(percentiles[0]), float(percentiles[1])]
            track["elements"].append(element)
            continue

        raise TemplateValidationError(f"Unsupported binding kind {element_kind!r}.")

    for section_tracks in tracks_by_section.values():
        for track in section_tracks:
            track_kind = _normalized_track_kind(str(track.get("kind", "normal")))
            if track_kind not in {"reference", "normal", "array"}:
                continue
            if "x_scale" in track and track["x_scale"] is not None:
                continue
            curves = [element for element in track["elements"] if element.get("kind") == "curve"]
            if len(curves) == 1:
                track["x_scale"] = deepcopy(curves[0].get("scale"))
            elif len(curves) > 1:
                track["x_scale"] = None

    return tracks_by_section


def _apply_layout_section_placeholders(document: dict[str, Any]) -> None:
    layout_data = document.get("layout")
    if layout_data is None:
        return
    layout = _ensure_mapping(layout_data, context="document.layout")
    heading = layout.get("heading")
    if isinstance(heading, dict):
        header = dict(_ensure_mapping(document.get("header", {}), context="document.header"))
        for key in ("title", "subtitle", "fields"):
            if key in heading and key not in header:
                header[key] = deepcopy(heading[key])
        document["header"] = header

    tail = layout.get("tail")
    if isinstance(tail, dict):
        footer = dict(_ensure_mapping(document.get("footer", {}), context="document.footer"))
        if "lines" in tail and "lines" not in footer:
            footer["lines"] = deepcopy(tail["lines"])
        document["footer"] = footer

    metadata = dict(_ensure_mapping(document.get("metadata", {}), context="document.metadata"))
    log_sections = _ensure_sequence(
        layout.get("log_sections", []), context="document.layout.log_sections"
    )
    active_section: dict[str, Any] = {}
    if log_sections:
        first = _ensure_mapping(log_sections[0], context="document.layout.log_sections[0]")
        active_section = {
            "id": str(first.get("id", "")),
            "title": str(first.get("title", "")),
            "subtitle": str(first.get("subtitle", "")),
        }
    metadata["layout_sections"] = {
        "heading": deepcopy(layout.get("heading", {})),
        "comments": deepcopy(layout.get("comments", [])),
        "log_sections": deepcopy(log_sections),
        "tail": deepcopy(layout.get("tail", {})),
        "active_section": active_section,
    }
    document["metadata"] = metadata


def _set_active_layout_section(document: dict[str, Any], section: dict[str, Any]) -> None:
    metadata = dict(_ensure_mapping(document.get("metadata", {}), context="document.metadata"))
    layout_sections_data = metadata.get("layout_sections", {})
    layout_sections = dict(
        _ensure_mapping(
            layout_sections_data,
            context="document.metadata.layout_sections",
        )
    )
    layout_sections["active_section"] = {
        "id": str(section.get("id", "")),
        "title": str(section.get("title", "")),
        "subtitle": str(section.get("subtitle", "")),
    }
    metadata["layout_sections"] = layout_sections
    document["metadata"] = metadata


def _apply_reference_layout_overrides(document: dict[str, Any]) -> None:
    tracks = document.get("tracks")
    if not isinstance(tracks, list):
        return

    depth_data: dict[str, Any]
    existing_depth = document.get("depth")
    if existing_depth is None:
        depth_data = {}
        document["depth"] = depth_data
    else:
        depth_data = dict(existing_depth)
        document["depth"] = depth_data

    for index, item in enumerate(tracks):
        track = _ensure_mapping(item, context=f"document.tracks[{index}]")
        kind = str(track.get("kind", "")).strip().lower()
        if kind not in {"reference", "depth"}:
            continue
        reference_data = track.get("reference")
        if reference_data is None:
            continue
        reference = _ensure_mapping(reference_data, context=f"document.tracks[{index}].reference")
        if not bool(reference.get("define_layout", True)):
            continue

        if "unit" in reference:
            depth_data["unit"] = str(reference["unit"])
        if "scale_ratio" in reference:
            depth_data["scale"] = int(reference["scale_ratio"])
        if "major_step" in reference:
            depth_data["major_step"] = float(reference["major_step"])
        if "minor_step" in reference:
            depth_data["minor_step"] = float(reference["minor_step"])
        elif "major_step" in reference:
            secondary_grid = _ensure_mapping(
                reference.get("secondary_grid", {}),
                context=f"document.tracks[{index}].reference.secondary_grid",
            )
            if bool(secondary_grid.get("display", True)):
                line_count = int(secondary_grid.get("line_count", 4))
                if line_count > 0:
                    depth_data["minor_step"] = float(reference["major_step"]) / line_count
        break


def build_documents_for_logfile(
    spec: LogFileSpec,
    dataset: WellDataset | dict[str, WellDataset],
    *,
    source_path: Path | dict[str, Path],
) -> tuple[LogDocument, ...]:
    base_document = deepcopy(spec.document)
    if "name" not in base_document:
        base_document["name"] = spec.name
    _apply_layout_section_placeholders(base_document)

    layout = _ensure_mapping(base_document["layout"], context="document.layout")
    sections = _layout_sections(layout, context="document.layout.log_sections")
    section_ids = [str(section["id"]) for section in sections]

    if isinstance(dataset, WellDataset):
        datasets_by_section = {section_id: dataset for section_id in section_ids}
    else:
        datasets_by_section = {}
        for section_id in section_ids:
            section_dataset = dataset.get(section_id)
            if section_dataset is None:
                raise TemplateValidationError(f"Missing dataset for section {section_id!r}.")
            datasets_by_section[section_id] = section_dataset

    if isinstance(source_path, Path):
        source_paths_by_section = {section_id: source_path for section_id in section_ids}
    else:
        source_paths_by_section = {}
        for section_id in section_ids:
            section_source_path = source_path.get(section_id)
            if section_source_path is None:
                raise TemplateValidationError(f"Missing source path for section {section_id!r}.")
            source_paths_by_section[section_id] = section_source_path

    tracks_by_section = _build_tracks_from_layout_bindings(datasets_by_section, base_document)

    documents: list[LogDocument] = []
    for section in sections:
        section_id = str(section["id"])
        section_document = deepcopy(base_document)
        _resolve_text_tokens(
            section_document,
            datasets_by_section[section_id],
            source_paths_by_section[section_id],
        )
        section_document["tracks"] = deepcopy(tracks_by_section[section_id])
        _set_active_layout_section(section_document, section)
        _apply_reference_layout_overrides(section_document)
        documents.append(document_from_mapping(section_document))
    return tuple(documents)


def build_document_for_logfile(
    spec: LogFileSpec,
    dataset: WellDataset,
    *,
    source_path: Path,
) -> LogDocument:
    documents = build_documents_for_logfile(spec, dataset, source_path=source_path)
    return documents[0]
