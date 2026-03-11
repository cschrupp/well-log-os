from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .errors import TemplateValidationError
from .model import (
    CurveElement,
    CurveHeaderDisplaySpec,
    CurveValueLabelsSpec,
    DepthAxisSpec,
    FooterSpec,
    GridDisplayMode,
    GridScaleKind,
    GridSpacingMode,
    GridSpec,
    HeaderField,
    HeaderSpec,
    LogDocument,
    MarkerSpec,
    NumberFormatKind,
    PageSpec,
    RasterElement,
    ReferenceAxisKind,
    ReferenceTrackSpec,
    ScaleKind,
    ScaleSpec,
    StyleSpec,
    TrackHeaderObjectKind,
    TrackHeaderObjectSpec,
    TrackHeaderSpec,
    TrackKind,
    TrackSpec,
    ZoneSpec,
)


def _ensure_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TemplateValidationError(
            f"Expected a mapping for {context}, got {type(value).__name__}."
        )
    return value


def _ensure_sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TemplateValidationError(
            f"Expected a sequence for {context}, got {type(value).__name__}."
        )
    return value


def _build_style(data: Mapping[str, Any] | None) -> StyleSpec:
    if not data:
        return StyleSpec()
    style_data = dict(data)
    return StyleSpec(
        color=style_data.get("color", "black"),
        line_width=float(style_data.get("line_width", 0.8)),
        line_style=style_data.get("line_style", "-"),
        opacity=float(style_data.get("opacity", 1.0)),
        fill_color=style_data.get("fill_color"),
        fill_alpha=float(style_data.get("fill_alpha", 0.2)),
        colormap=style_data.get("colormap", "viridis"),
    )


def _parse_scale_ratio(value: str | int | float | None) -> int:
    if value is None:
        return 200
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if ":" in text:
        _, right = text.split(":", 1)
        return int(float(right.strip()))
    return int(float(text))


def _build_scale(data: Mapping[str, Any] | None) -> ScaleSpec | None:
    if not data:
        return None
    scale_data = dict(data)
    kind_raw = str(scale_data.get("kind", "linear")).strip().lower()
    if kind_raw == "logarithmic":
        kind_raw = "log"
    if kind_raw == "tangent":
        kind_raw = "tangential"
    kind = ScaleKind(kind_raw)
    return ScaleSpec(
        kind=kind,
        minimum=float(scale_data.get("min", scale_data.get("minimum", 0.0))),
        maximum=float(scale_data.get("max", scale_data.get("maximum", 1.0))),
        reverse=bool(scale_data.get("reverse", False)),
    )


def _parse_curve_wrap_config(
    value: Any,
    *,
    context: str,
) -> tuple[bool, str | None]:
    if value is None:
        return False, None
    if isinstance(value, bool):
        return value, None

    wrap_data = _ensure_mapping(value, context=context)
    enabled = bool(wrap_data.get("enabled", True))
    color = wrap_data.get("color")
    if color is None:
        return enabled, None
    color_text = str(color).strip()
    if not color_text:
        raise TemplateValidationError(f"{context}.color must be non-empty when provided.")
    return enabled, color_text


def _parse_grid_scale_kind(value: Any, *, context: str) -> GridScaleKind:
    text = str(value or "linear").strip().lower()
    alias_map = {
        "linear": GridScaleKind.LINEAR,
        "log": GridScaleKind.LOGARITHMIC,
        "logarithmic": GridScaleKind.LOGARITHMIC,
        "exponential": GridScaleKind.LOGARITHMIC,
        "tangent": GridScaleKind.TANGENTIAL,
        "tangential": GridScaleKind.TANGENTIAL,
    }
    kind = alias_map.get(text)
    if kind is None:
        raise TemplateValidationError(
            f"{context} must be linear, logarithmic/exponential, or tangential."
        )
    return kind


def _parse_grid_display_mode(
    value: Any,
    *,
    context: str,
    default: GridDisplayMode = GridDisplayMode.BELOW,
) -> GridDisplayMode:
    if value is None:
        return default
    if isinstance(value, bool):
        return GridDisplayMode.BELOW if value else GridDisplayMode.NONE
    text = str(value).strip().lower()
    alias_map = {
        "below": GridDisplayMode.BELOW,
        "under": GridDisplayMode.BELOW,
        "above": GridDisplayMode.ABOVE,
        "over": GridDisplayMode.ABOVE,
        "none": GridDisplayMode.NONE,
        "off": GridDisplayMode.NONE,
        "hidden": GridDisplayMode.NONE,
        "false": GridDisplayMode.NONE,
    }
    mode = alias_map.get(text)
    if mode is None:
        raise TemplateValidationError(f"{context} must be below, above, or none.")
    return mode


def _build_grid_spec(data: Any, *, context: str) -> GridSpec:
    grid_data = _ensure_mapping(data or {}, context=f"{context}.grid")
    horizontal_data = _ensure_mapping(
        grid_data.get("horizontal", {}),
        context=f"{context}.grid.horizontal",
    )
    vertical_data = _ensure_mapping(
        grid_data.get("vertical", {}),
        context=f"{context}.grid.vertical",
    )
    horizontal_main = _ensure_mapping(
        horizontal_data.get("main", {}),
        context=f"{context}.grid.horizontal.main",
    )
    horizontal_secondary = _ensure_mapping(
        horizontal_data.get("secondary", {}),
        context=f"{context}.grid.horizontal.secondary",
    )
    vertical_main = _ensure_mapping(
        vertical_data.get("main", {}),
        context=f"{context}.grid.vertical.main",
    )
    vertical_secondary = _ensure_mapping(
        vertical_data.get("secondary", {}),
        context=f"{context}.grid.vertical.secondary",
    )

    def _parse_spacing_mode(value: Any, *, field_context: str) -> GridSpacingMode:
        text = str(value or "count").strip().lower()
        alias_map = {
            "count": GridSpacingMode.COUNT,
            "manual": GridSpacingMode.COUNT,
            "scale": GridSpacingMode.SCALE,
            "auto": GridSpacingMode.SCALE,
        }
        mode = alias_map.get(text)
        if mode is None:
            raise TemplateValidationError(f"{field_context} must be count/manual or scale/auto.")
        return mode

    major_visible = bool(horizontal_main.get("visible", grid_data.get("major", True)))
    minor_visible = bool(horizontal_secondary.get("visible", grid_data.get("minor", True)))
    major_alpha = float(grid_data.get("major_alpha", 0.35))
    minor_alpha = float(grid_data.get("minor_alpha", 0.15))

    global_display_raw = grid_data.get("display")
    global_display = _parse_grid_display_mode(
        global_display_raw,
        context=f"{context}.grid.display",
    )
    horizontal_display = _parse_grid_display_mode(
        horizontal_data.get("display", global_display),
        context=f"{context}.grid.horizontal.display",
    )
    vertical_display = _parse_grid_display_mode(
        vertical_data.get("display", global_display),
        context=f"{context}.grid.vertical.display",
    )

    horizontal_major_thickness = (
        float(horizontal_main["thickness"]) if "thickness" in horizontal_main else None
    )
    horizontal_minor_thickness = (
        float(horizontal_secondary["thickness"]) if "thickness" in horizontal_secondary else None
    )
    horizontal_major_color = str(horizontal_main["color"]) if "color" in horizontal_main else None
    horizontal_minor_color = (
        str(horizontal_secondary["color"]) if "color" in horizontal_secondary else None
    )
    horizontal_major_alpha = float(horizontal_main.get("alpha", major_alpha))
    horizontal_minor_alpha = float(horizontal_secondary.get("alpha", minor_alpha))

    vertical_main_thickness = (
        float(vertical_main["thickness"]) if "thickness" in vertical_main else None
    )
    vertical_secondary_thickness = (
        float(vertical_secondary["thickness"]) if "thickness" in vertical_secondary else None
    )
    vertical_main_color = str(vertical_main["color"]) if "color" in vertical_main else None
    vertical_secondary_color = (
        str(vertical_secondary["color"]) if "color" in vertical_secondary else None
    )
    vertical_main_alpha = float(vertical_main.get("alpha", major_alpha))
    vertical_secondary_alpha = float(vertical_secondary.get("alpha", minor_alpha))

    try:
        return GridSpec(
            major=major_visible,
            minor=minor_visible,
            major_alpha=major_alpha,
            minor_alpha=minor_alpha,
            horizontal_display=horizontal_display,
            horizontal_major_visible=major_visible,
            horizontal_minor_visible=minor_visible,
            horizontal_major_color=horizontal_major_color,
            horizontal_minor_color=horizontal_minor_color,
            horizontal_major_thickness=horizontal_major_thickness,
            horizontal_minor_thickness=horizontal_minor_thickness,
            horizontal_major_alpha=horizontal_major_alpha,
            horizontal_minor_alpha=horizontal_minor_alpha,
            vertical_display=vertical_display,
            vertical_main_visible=bool(vertical_main.get("visible", major_visible)),
            vertical_main_line_count=int(vertical_main.get("line_count", 4)),
            vertical_main_thickness=vertical_main_thickness,
            vertical_main_color=vertical_main_color,
            vertical_main_alpha=vertical_main_alpha,
            vertical_main_scale=_parse_grid_scale_kind(
                vertical_main.get("scale", "linear"),
                context=f"{context}.grid.vertical.main.scale",
            ),
            vertical_main_spacing_mode=_parse_spacing_mode(
                vertical_main.get("spacing_mode", "count"),
                field_context=f"{context}.grid.vertical.main.spacing_mode",
            ),
            vertical_secondary_visible=bool(vertical_secondary.get("visible", minor_visible)),
            vertical_secondary_line_count=int(vertical_secondary.get("line_count", 4)),
            vertical_secondary_thickness=vertical_secondary_thickness,
            vertical_secondary_color=vertical_secondary_color,
            vertical_secondary_alpha=vertical_secondary_alpha,
            vertical_secondary_scale=_parse_grid_scale_kind(
                vertical_secondary.get("scale", "linear"),
                context=f"{context}.grid.vertical.secondary.scale",
            ),
            vertical_secondary_spacing_mode=_parse_spacing_mode(
                vertical_secondary.get("spacing_mode", "count"),
                field_context=f"{context}.grid.vertical.secondary.spacing_mode",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise TemplateValidationError("Invalid track grid configuration.") from exc


def _parse_track_kind(raw_kind: Any) -> TrackKind:
    text = str(raw_kind or "normal").strip().lower()
    alias_map = {
        "reference": TrackKind.REFERENCE,
        "depth": TrackKind.REFERENCE,
        "normal": TrackKind.NORMAL,
        "curve": TrackKind.NORMAL,
        "array": TrackKind.ARRAY,
        "image": TrackKind.ARRAY,
        "annotation": TrackKind.ANNOTATION,
    }
    kind = alias_map.get(text)
    if kind is None:
        raise TemplateValidationError(f"Unsupported track kind {text!r}.")
    return kind


def _build_reference_track(data: Any) -> ReferenceTrackSpec:
    if data is None:
        return ReferenceTrackSpec()
    ref_data = _ensure_mapping(data, context="track.reference")

    secondary_grid_data = _ensure_mapping(
        ref_data.get("secondary_grid", {}),
        context="track.reference.secondary_grid",
    )
    header_data = _ensure_mapping(
        ref_data.get("header", {}),
        context="track.reference.header",
    )
    number_format_data = _ensure_mapping(
        ref_data.get("number_format", {}),
        context="track.reference.number_format",
    )

    axis_text = str(ref_data.get("axis", "depth")).strip().lower()
    number_format_text = str(number_format_data.get("format", "automatic")).strip().lower()
    try:
        axis = ReferenceAxisKind(axis_text)
        number_format = NumberFormatKind(number_format_text)
    except ValueError as exc:
        raise TemplateValidationError("Invalid reference track axis or number format.") from exc

    try:
        return ReferenceTrackSpec(
            axis=axis,
            define_layout=bool(ref_data.get("define_layout", True)),
            unit=ref_data.get("unit"),
            scale_ratio=(int(ref_data["scale_ratio"]) if "scale_ratio" in ref_data else None),
            major_step=(float(ref_data["major_step"]) if "major_step" in ref_data else None),
            minor_step=(float(ref_data["minor_step"]) if "minor_step" in ref_data else None),
            secondary_grid_display=bool(secondary_grid_data.get("display", True)),
            secondary_grid_line_count=int(secondary_grid_data.get("line_count", 4)),
            display_unit_in_header=bool(header_data.get("display_unit", True)),
            display_scale_in_header=bool(header_data.get("display_scale", True)),
            display_annotations_in_header=bool(header_data.get("display_annotations", True)),
            number_format=number_format,
            precision=int(number_format_data.get("precision", 2)),
            values_orientation=str(ref_data.get("values_orientation", "horizontal")),
        )
    except (TypeError, ValueError) as exc:
        raise TemplateValidationError("Invalid reference track configuration.") from exc


def _build_curve_value_labels(data: Any) -> CurveValueLabelsSpec:
    if data is None:
        return CurveValueLabelsSpec()
    label_data = _ensure_mapping(data, context="curve.value_labels")
    format_text = str(label_data.get("format", "automatic")).strip().lower()
    try:
        number_format = NumberFormatKind(format_text)
    except ValueError as exc:
        raise TemplateValidationError("Invalid curve.value_labels.format.") from exc

    try:
        return CurveValueLabelsSpec(
            step=float(label_data.get("step", 5.0)),
            number_format=number_format,
            precision=int(label_data.get("precision", 2)),
            color=label_data.get("color"),
            font_size=float(label_data.get("font_size", 5.5)),
            font_family=label_data.get("font_family"),
            font_weight=str(label_data.get("font_weight", "normal")),
            font_style=str(label_data.get("font_style", "normal")),
            horizontal_alignment=str(label_data.get("horizontal_alignment", "center"))
            .strip()
            .lower(),
            vertical_alignment=str(label_data.get("vertical_alignment", "center")).strip().lower(),
        )
    except (TypeError, ValueError) as exc:
        raise TemplateValidationError("Invalid curve.value_labels configuration.") from exc


def _build_curve_header_display(data: Any) -> CurveHeaderDisplaySpec:
    if data is None:
        return CurveHeaderDisplaySpec()
    display_data = _ensure_mapping(data, context="curve.header_display")
    return CurveHeaderDisplaySpec(
        show_name=bool(display_data.get("show_name", True)),
        show_unit=bool(display_data.get("show_unit", True)),
        show_limits=bool(display_data.get("show_limits", True)),
        show_color=bool(display_data.get("show_color", True)),
    )


def _build_header(data: Mapping[str, Any] | None) -> HeaderSpec:
    if not data:
        return HeaderSpec()
    fields = tuple(
        HeaderField(
            label=str(item["label"]),
            source_key=str(item["source_key"]),
            default=str(item.get("default", "")),
        )
        for item in data.get("fields", [])
    )
    return HeaderSpec(
        title=data.get("title"),
        subtitle=data.get("subtitle"),
        fields=fields,
    )


def _build_footer(data: Mapping[str, Any] | None) -> FooterSpec:
    if not data:
        return FooterSpec()
    lines = tuple(str(item) for item in data.get("lines", []))
    return FooterSpec(lines=lines)


def _build_track_header(data: Any) -> TrackHeaderSpec:
    if data is None:
        return TrackHeaderSpec()
    header_data = _ensure_mapping(data, context="track_header")

    objects_data = header_data.get("objects")
    if objects_data is not None:
        object_items = _ensure_sequence(objects_data, context="track_header.objects")
        objects = []
        for index, item in enumerate(object_items):
            object_data = _ensure_mapping(item, context=f"track_header.objects[{index}]")
            try:
                objects.append(
                    TrackHeaderObjectSpec(
                        kind=TrackHeaderObjectKind(str(object_data["kind"])),
                        enabled=bool(object_data.get("enabled", True)),
                        reserve_space=bool(object_data.get("reserve_space", True)),
                        line_units=int(object_data.get("line_units", 1)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TemplateValidationError(
                    f"Invalid track header object at index {index}."
                ) from exc
        try:
            return TrackHeaderSpec(objects=tuple(objects))
        except ValueError as exc:
            raise TemplateValidationError("Invalid track_header.objects configuration.") from exc

    defaults = TrackHeaderSpec().objects
    object_map = {obj.kind: obj for obj in defaults}
    for kind in TrackHeaderObjectKind:
        key = kind.value
        if key not in header_data:
            continue
        object_data = _ensure_mapping(header_data[key], context=f"track_header.{key}")
        default = object_map[kind]
        try:
            object_map[kind] = TrackHeaderObjectSpec(
                kind=kind,
                enabled=bool(object_data.get("enabled", default.enabled)),
                reserve_space=bool(object_data.get("reserve_space", default.reserve_space)),
                line_units=int(object_data.get("line_units", default.line_units)),
            )
        except (TypeError, ValueError) as exc:
            raise TemplateValidationError(f"Invalid track_header.{key} configuration.") from exc
    ordered = tuple(object_map[item.kind] for item in defaults)
    return TrackHeaderSpec(objects=ordered)


def _build_markers(data: Any) -> tuple[MarkerSpec, ...]:
    if data is None:
        return ()

    markers_data = _ensure_sequence(data, context="markers")
    markers = []
    for index, item in enumerate(markers_data):
        marker_data = _ensure_mapping(item, context=f"markers[{index}]")
        try:
            markers.append(
                MarkerSpec(
                    depth=float(marker_data["depth"]),
                    label=str(marker_data.get("label", "")),
                    color=str(marker_data.get("color", "#666666")),
                    line_style=str(marker_data.get("line_style", "--")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TemplateValidationError(f"Invalid marker at index {index}.") from exc
    return tuple(markers)


def _build_zones(data: Any) -> tuple[ZoneSpec, ...]:
    if data is None:
        return ()

    zones_data = _ensure_sequence(data, context="zones")
    zones = []
    for index, item in enumerate(zones_data):
        zone_data = _ensure_mapping(item, context=f"zones[{index}]")
        try:
            zones.append(
                ZoneSpec(
                    top=float(zone_data["top"]),
                    base=float(zone_data["base"]),
                    label=str(zone_data.get("label", "")),
                    fill_color=str(zone_data.get("fill_color", "#d9d9d9")),
                    alpha=float(zone_data.get("alpha", 0.25)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TemplateValidationError(f"Invalid zone at index {index}.") from exc
    return tuple(zones)


def _build_track(track_data: Mapping[str, Any]) -> TrackSpec:
    kind = _parse_track_kind(track_data.get("kind", "normal"))
    elements = []
    for item in track_data.get("elements", []):
        element_data = _ensure_mapping(item, context=f"track {track_data.get('id', '')} element")
        element_kind = element_data.get("kind", "curve")
        if element_kind == "curve":
            wrap_enabled, wrap_color = _parse_curve_wrap_config(
                element_data.get("wrap"),
                context=f"track {track_data.get('id', '')} element.wrap",
            )
            if "wrap_color" in element_data and element_data["wrap_color"] is not None:
                wrap_color_text = str(element_data["wrap_color"]).strip()
                if not wrap_color_text:
                    raise TemplateValidationError(
                        f"track {track_data.get('id', '')} element.wrap_color must be non-empty."
                    )
                wrap_color = wrap_color_text
            elements.append(
                CurveElement(
                    channel=str(element_data["channel"]),
                    label=element_data.get("label"),
                    style=_build_style(element_data.get("style")),
                    scale=_build_scale(element_data.get("scale")),
                    wrap=wrap_enabled,
                    wrap_color=wrap_color,
                    render_mode=str(element_data.get("render_mode", "line")),
                    value_labels=_build_curve_value_labels(element_data.get("value_labels")),
                    header_display=_build_curve_header_display(element_data.get("header_display")),
                )
            )
        elif element_kind in {"raster", "image"}:
            limits = element_data.get("color_limits")
            color_limits = None
            if limits is not None:
                if not isinstance(limits, Sequence) or len(limits) != 2:
                    raise TemplateValidationError(
                        "Raster color_limits must contain two numeric values."
                    )
                color_limits = (float(limits[0]), float(limits[1]))
            elements.append(
                RasterElement(
                    channel=str(element_data["channel"]),
                    style=_build_style(element_data.get("style")),
                    interpolation=str(element_data.get("interpolation", "nearest")),
                    color_limits=color_limits,
                )
            )
        else:
            raise TemplateValidationError(f"Unsupported element kind {element_kind!r}.")
    track_context = f"track {track_data.get('id', '')}"
    reference = (
        _build_reference_track(track_data.get("reference")) if kind == TrackKind.REFERENCE else None
    )
    return TrackSpec(
        id=str(track_data["id"]),
        title=str(track_data.get("title", track_data["id"])),
        kind=kind,
        width_mm=float(track_data["width_mm"]),
        elements=tuple(elements),
        x_scale=_build_scale(track_data.get("x_scale")),
        header=_build_track_header(track_data.get("track_header")),
        grid=_build_grid_spec(track_data.get("grid"), context=track_context),
        reference=reference,
    )


def _resolve_depth_axis_from_reference_tracks(
    depth_axis: DepthAxisSpec,
    tracks: tuple[TrackSpec, ...],
) -> DepthAxisSpec:
    for track in tracks:
        if track.kind != TrackKind.REFERENCE:
            continue
        reference = track.reference or ReferenceTrackSpec()
        if not reference.define_layout:
            continue

        major_step = (
            reference.major_step if reference.major_step is not None else depth_axis.major_step
        )
        if reference.minor_step is not None:
            minor_step = reference.minor_step
        elif reference.secondary_grid_display and reference.secondary_grid_line_count > 0:
            minor_step = major_step / reference.secondary_grid_line_count
        else:
            minor_step = depth_axis.minor_step

        return DepthAxisSpec(
            unit=str(reference.unit or depth_axis.unit),
            scale_ratio=reference.scale_ratio or depth_axis.scale_ratio,
            major_step=major_step,
            minor_step=minor_step,
        )
    return depth_axis


def document_from_mapping(data: Mapping[str, Any]) -> LogDocument:
    root = _ensure_mapping(data, context="document")
    page_data = _ensure_mapping(root.get("page", {}), context="page")
    if "size" in page_data:
        page = PageSpec.from_name(
            str(page_data["size"]),
            orientation=str(page_data.get("orientation", "portrait")),
            continuous=bool(page_data.get("continuous", False)),
            margin_left_mm=float(page_data.get("margin_left_mm", 0.0)),
            margin_right_mm=float(page_data.get("margin_right_mm", 10.0)),
            margin_top_mm=float(page_data.get("margin_top_mm", 10.0)),
            margin_bottom_mm=float(page_data.get("margin_bottom_mm", 10.0)),
            header_height_mm=float(page_data.get("header_height_mm", 18.0)),
            track_header_height_mm=float(page_data.get("track_header_height_mm", 8.0)),
            footer_height_mm=float(page_data.get("footer_height_mm", 10.0)),
            track_gap_mm=float(page_data.get("track_gap_mm", 0.0)),
        )
    else:
        page = PageSpec(
            width_mm=float(page_data["width_mm"]),
            height_mm=float(page_data["height_mm"]),
            continuous=bool(page_data.get("continuous", False)),
            margin_left_mm=float(page_data.get("margin_left_mm", 0.0)),
            margin_right_mm=float(page_data.get("margin_right_mm", 10.0)),
            margin_top_mm=float(page_data.get("margin_top_mm", 10.0)),
            margin_bottom_mm=float(page_data.get("margin_bottom_mm", 10.0)),
            header_height_mm=float(page_data.get("header_height_mm", 18.0)),
            track_header_height_mm=float(page_data.get("track_header_height_mm", 8.0)),
            footer_height_mm=float(page_data.get("footer_height_mm", 10.0)),
            track_gap_mm=float(page_data.get("track_gap_mm", 0.0)),
        )

    depth_data = _ensure_mapping(root.get("depth", {}), context="depth")
    depth_axis = DepthAxisSpec(
        unit=str(depth_data.get("unit", "m")),
        scale_ratio=_parse_scale_ratio(depth_data.get("scale", depth_data.get("scale_ratio", 200))),
        major_step=float(depth_data.get("major_step", 10.0)),
        minor_step=float(depth_data.get("minor_step", 2.0)),
    )

    track_items = root.get("tracks", [])
    if not isinstance(track_items, Sequence):
        raise TemplateValidationError("tracks must be a sequence.")
    tracks = tuple(_build_track(_ensure_mapping(item, context="track")) for item in track_items)
    depth_axis = _resolve_depth_axis_from_reference_tracks(depth_axis, tracks)

    depth_range_data = root.get("depth_range")
    depth_range = None
    if depth_range_data is not None:
        if not isinstance(depth_range_data, Sequence) or len(depth_range_data) != 2:
            raise TemplateValidationError("depth_range must contain two numeric values.")
        depth_range = (float(depth_range_data[0]), float(depth_range_data[1]))

    return LogDocument(
        name=str(root.get("name", "well-log")),
        page=page,
        depth_axis=depth_axis,
        tracks=tracks,
        depth_range=depth_range,
        header=_build_header(_ensure_mapping(root.get("header", {}), context="header")),
        footer=_build_footer(_ensure_mapping(root.get("footer", {}), context="footer")),
        markers=_build_markers(root.get("markers")),
        zones=_build_zones(root.get("zones")),
        metadata=dict(root.get("metadata", {})),
    )


def load_document(path: str | Path) -> LogDocument:
    template_path = Path(path)
    with template_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return document_from_mapping(payload)
