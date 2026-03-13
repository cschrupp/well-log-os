from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..errors import DependencyUnavailableError, TemplateValidationError
from ..layout import LayoutEngine
from ..model import (
    CurveElement,
    CurveFillKind,
    FooterSpec,
    GridDisplayMode,
    GridScaleKind,
    GridSpacingMode,
    HeaderSpec,
    LogDocument,
    NumberFormatKind,
    RasterChannel,
    RasterColorbarPosition,
    RasterElement,
    RasterNormalizationKind,
    RasterProfileKind,
    RasterWaveformSpec,
    ReferenceTrackSpec,
    ScalarChannel,
    ScaleKind,
    TrackHeaderObjectKind,
    TrackKind,
    WellDataset,
)
from ..units import DEFAULT_UNITS, SimpleUnitRegistry
from .base import Renderer, RenderResult

DEFAULT_MPL_STYLE_PATH = Path(__file__).with_name("matplotlib_defaults.yaml")


def _load_default_mpl_style(path: Path = DEFAULT_MPL_STYLE_PATH) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise RuntimeError(f"Unable to load matplotlib defaults from {path}.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Matplotlib defaults file must contain a mapping: {path}.")
    return payload


DEFAULT_MPL_STYLE = _load_default_mpl_style()


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class _CurvePlotData:
    depth: np.ndarray
    raw_values: np.ndarray
    plot_values: np.ndarray
    valid_mask: np.ndarray
    wrapped_mask: np.ndarray
    scale: Any


class MatplotlibRenderer(Renderer):
    def __init__(
        self,
        registry: SimpleUnitRegistry = DEFAULT_UNITS,
        *,
        dpi: int = 200,
        continuous_strip_page_height_mm: float | None = None,
        style: dict[str, Any] | None = None,
    ) -> None:
        if dpi <= 0:
            raise ValueError("Renderer dpi must be positive.")
        if continuous_strip_page_height_mm is not None and continuous_strip_page_height_mm <= 0:
            raise ValueError("continuous_strip_page_height_mm must be positive when provided.")
        if style is not None and not isinstance(style, dict):
            raise ValueError("Renderer style overrides must be a mapping when provided.")
        self.registry = registry
        self.layout_engine = LayoutEngine(registry)
        self.dpi = dpi
        self.continuous_strip_page_height_mm = continuous_strip_page_height_mm
        self.style = _deep_merge_dicts(DEFAULT_MPL_STYLE, style or {})

    def _style_section(self, section: str) -> dict[str, Any]:
        values = self.style.get(section, {})
        return values if isinstance(values, dict) else {}

    def _style_value(self, section: str, key: str) -> Any:
        return self._style_section(section)[key]

    def _is_reference_track(self, track) -> bool:
        return track.kind == TrackKind.REFERENCE

    def _is_annotation_track(self, track) -> bool:
        return track.kind == TrackKind.ANNOTATION

    def _reference_spec(self, track) -> ReferenceTrackSpec:
        reference = track.reference
        if reference is None:
            return ReferenceTrackSpec()
        return reference

    def _reference_scale_text(self, track, document) -> str:
        reference = self._reference_spec(track)
        parts: list[str] = []
        if reference.display_unit_in_header:
            parts.append(document.depth_axis.unit)
        if reference.display_scale_in_header:
            parts.append(f"1:{document.depth_axis.scale_ratio}")
        if reference.display_annotations_in_header and document.markers:
            parts.append(f"ANN {len(document.markers)}")
        if not parts:
            return "Reference"
        return " ".join(parts)

    def _resolve_reference_steps(self, track, document) -> tuple[float, float, bool]:
        reference = self._reference_spec(track)
        major_step = reference.major_step or document.depth_axis.major_step
        if reference.minor_step is not None:
            minor_step = reference.minor_step
        elif reference.secondary_grid_display and reference.secondary_grid_line_count > 0:
            minor_step = major_step / reference.secondary_grid_line_count
        else:
            minor_step = document.depth_axis.minor_step
        return major_step, minor_step, reference.secondary_grid_display

    def _format_reference_value(self, value: float, reference: ReferenceTrackSpec) -> str:
        precision = reference.precision
        return self._format_number(value, reference.number_format, precision)

    def _format_number(self, value: float, number_format: NumberFormatKind, precision: int) -> str:
        if number_format == NumberFormatKind.FIXED:
            return f"{value:.{precision}f}"
        if number_format == NumberFormatKind.SCIENTIFIC:
            return f"{value:.{precision}e}"
        if number_format == NumberFormatKind.CONCISE:
            return f"{value:.{precision}g}"

        # Automatic mode: integers stay clean, otherwise use concise notation.
        rounded = round(value)
        if np.isclose(value, rounded):
            return f"{int(rounded)}"
        return f"{value:.{precision}g}"

    def _draw_reference_values_inside(
        self, ax, track, document, window, *, major_step: float
    ) -> None:
        from matplotlib.transforms import blended_transform_factory

        track_style = self._style_section("track")
        reference = self._reference_spec(track)
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        x = float(track_style["reference_label_x"])
        align = str(track_style.get("reference_label_align", "center")).lower()
        if align not in {"left", "center", "right"}:
            align = "center"
        font_family = track_style.get("reference_label_fontfamily")
        font_weight = str(track_style.get("reference_label_fontweight", "normal"))
        font_style = str(track_style.get("reference_label_fontstyle", "normal"))
        start = np.floor(window.start / major_step) * major_step
        epsilon = max(abs(major_step) * 1e-6, 1e-8)
        value = start
        while value <= window.stop + epsilon:
            if value >= window.start - epsilon:
                text = self._format_reference_value(float(value), reference)
                rotation = 90 if reference.values_orientation == "vertical" else 0
                kwargs = {
                    "transform": transform,
                    "ha": align,
                    "va": "center",
                    "fontsize": float(track_style["reference_label_fontsize"]),
                    "color": str(track_style["reference_label_color"]),
                    "rotation": rotation,
                    "clip_on": True,
                    "fontweight": font_weight,
                    "fontstyle": font_style,
                }
                if isinstance(font_family, str) and font_family.strip():
                    kwargs["fontfamily"] = font_family
                ax.text(
                    x,
                    float(value),
                    text,
                    **kwargs,
                )
            value += major_step

    def _draw_reference_edge_ticks(
        self,
        ax,
        window,
        *,
        major_step: float,
        minor_step: float,
        draw_minor: bool,
    ) -> None:
        from matplotlib.transforms import blended_transform_factory

        track_style = self._style_section("track")
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        major_len = min(max(float(track_style["reference_major_tick_length_ratio"]), 0.0), 0.5)
        minor_len = min(max(float(track_style["reference_minor_tick_length_ratio"]), 0.0), 0.5)
        tick_color = str(track_style.get("reference_tick_color", "#5f5f5f"))
        tick_linewidth = float(track_style.get("reference_tick_linewidth", 0.65))

        def _draw_ticks_for_step(step: float, tick_len: float) -> None:
            start = np.floor(window.start / step) * step
            epsilon = max(abs(step) * 1e-6, 1e-8)
            value = start
            while value <= window.stop + epsilon:
                if value >= window.start - epsilon:
                    y = float(value)
                    ax.plot(
                        [0.0, tick_len],
                        [y, y],
                        transform=transform,
                        color=tick_color,
                        linewidth=tick_linewidth,
                    )
                    ax.plot(
                        [1.0 - tick_len, 1.0],
                        [y, y],
                        transform=transform,
                        color=tick_color,
                        linewidth=tick_linewidth,
                    )
                value += step

        _draw_ticks_for_step(major_step, major_len)
        if draw_minor and minor_step > 0:
            _draw_ticks_for_step(minor_step, minor_len)

    def _curve_count(self, track) -> int:
        return sum(1 for element in track.elements if isinstance(element, CurveElement))

    def _document_curve_row_capacity(self, document: LogDocument) -> int:
        return max((self._curve_count(track) for track in document.tracks), default=0)

    def _header_property_group_capacity(self, document: LogDocument) -> int:
        return max(1, self._document_curve_row_capacity(document))

    def _curve_header_row_count(self, document: LogDocument, track) -> int:
        count = self._curve_count(track)
        if count <= 0:
            return 0
        return max(count, self._header_property_group_capacity(document))

    def _effective_header_line_units(self, track, header_item) -> int:
        if not header_item.enabled or not header_item.reserve_space:
            return header_item.line_units
        if header_item.kind in {TrackHeaderObjectKind.LEGEND, TrackHeaderObjectKind.SCALE}:
            return max(header_item.line_units, self._curve_count(track))
        return header_item.line_units

    def _auto_adjust_track_header_height(self, document: LogDocument) -> LogDocument:
        base_height = document.page.track_header_height_mm
        if base_height <= 0:
            return document

        section_title_height = self._section_title_height_mm(document)
        required_height = float(base_height)
        for track in document.tracks:
            reserved = track.header.reserved_objects()
            if not reserved:
                continue
            configured_units = sum(item.line_units for item in reserved)
            if configured_units <= 0:
                continue
            effective_units = sum(
                self._effective_header_line_units(track, item) for item in reserved
            )
            required_height = max(
                required_height,
                base_height * (effective_units / configured_units),
            )

        if section_title_height > 0:
            required_height += section_title_height

        if required_height <= base_height + 1e-9:
            return document
        return replace(
            document,
            page=replace(document.page, track_header_height_mm=required_height),
        )

    def _active_section_title(self, document: LogDocument) -> tuple[str, str]:
        metadata = document.metadata
        if not isinstance(metadata, dict):
            return "", ""
        sections_data = metadata.get("layout_sections")
        if not isinstance(sections_data, dict):
            return "", ""

        active = sections_data.get("active_section")
        if isinstance(active, dict):
            title = str(active.get("title", "")).strip()
            subtitle = str(active.get("subtitle", "")).strip()
            return title, subtitle

        sections = sections_data.get("log_sections")
        if isinstance(sections, list) and sections:
            first = sections[0]
            if isinstance(first, dict):
                title = str(first.get("title", "")).strip()
                subtitle = str(first.get("subtitle", "")).strip()
                return title, subtitle
        return "", ""

    def _section_title_height_mm(self, document: LogDocument) -> float:
        style = self._style_section("section_title")
        if not bool(style.get("enabled", True)):
            return 0.0
        title, _ = self._active_section_title(document)
        if not title:
            return 0.0
        return max(float(style.get("height_mm", 0.0)), 0.0)

    def _draw_section_title_box(self, fig, document, page_layout) -> float:
        if page_layout.page_number != 1 or not page_layout.track_header_top_frames:
            return 0.0

        section_height_mm = self._section_title_height_mm(document)
        if section_height_mm <= 0:
            return 0.0

        header_frame = page_layout.track_header_top_frames[0].frame
        section_height_mm = min(section_height_mm, max(header_frame.height_mm - 0.2, 0.0))
        if section_height_mm <= 0:
            return 0.0

        section_style = self._style_section("section_title")
        left_mm = min(frame.frame.x_mm for frame in page_layout.track_header_top_frames)
        right_mm = max(
            frame.frame.x_mm + frame.frame.width_mm for frame in page_layout.track_header_top_frames
        )
        width_mm = max(right_mm - left_mm, 0.0)
        if width_mm <= 0:
            return 0.0

        box_frame = replace(
            header_frame,
            x_mm=left_mm,
            width_mm=width_mm,
            height_mm=section_height_mm,
        )
        ax = fig.add_axes(self._normalize_frame(page_layout.page, box_frame))
        ax.set_facecolor(str(section_style["background_color"]))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(str(section_style["border_color"]))
            spine.set_linewidth(float(section_style["border_linewidth"]))

        title, subtitle = self._active_section_title(document)
        ax.text(
            0.5,
            float(section_style["title_y"]),
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=float(section_style["title_fontsize"]),
            fontweight="bold",
            color=str(section_style["title_color"]),
            clip_on=True,
        )
        if subtitle:
            ax.text(
                0.5,
                float(section_style["subtitle_y"]),
                subtitle,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=float(section_style["subtitle_fontsize"]),
                color=str(section_style["subtitle_color"]),
                clip_on=True,
            )
        return section_height_mm

    def _build_continuous_strip_document(self, document: LogDocument) -> LogDocument:
        if self.continuous_strip_page_height_mm is None:
            return document
        return self._build_continuous_strip_document_with_height(
            document,
            self.continuous_strip_page_height_mm,
        )

    def _build_continuous_strip_document_with_height(
        self,
        document: LogDocument,
        page_height_mm: float,
    ) -> LogDocument:
        if page_height_mm <= 0:
            raise ValueError("Continuous strip page height must be positive.")
        strip_page = replace(
            document.page,
            continuous=False,
            height_mm=page_height_mm,
            margin_top_mm=0.0,
            margin_bottom_mm=0.0,
            header_height_mm=0.0,
            footer_height_mm=0.0,
        )
        return replace(
            document,
            page=strip_page,
            header=HeaderSpec(),
            footer=FooterSpec(),
        )

    def render(
        self,
        document: LogDocument,
        dataset: WellDataset,
        *,
        output_path: str | Path | None = None,
    ) -> RenderResult:
        return self.render_documents((document,), dataset, output_path=output_path)

    def render_documents(
        self,
        documents: tuple[LogDocument, ...] | list[LogDocument],
        dataset: WellDataset | tuple[WellDataset, ...] | list[WellDataset],
        *,
        output_path: str | Path | None = None,
    ) -> RenderResult:
        normalized_documents = tuple(documents)
        if not normalized_documents:
            raise ValueError("render_documents requires at least one document.")
        if isinstance(dataset, WellDataset):
            normalized_datasets = tuple(dataset for _ in normalized_documents)
        else:
            normalized_datasets = tuple(dataset)
            if len(normalized_datasets) != len(normalized_documents):
                raise ValueError("render_documents requires one dataset per document.")
        document_dataset_pairs = tuple(zip(normalized_documents, normalized_datasets, strict=True))
        return self._render_documents(document_dataset_pairs, output_path=output_path)

    def _render_documents(
        self,
        document_dataset_pairs: tuple[tuple[LogDocument, WellDataset], ...],
        *,
        output_path: str | Path | None = None,
    ) -> RenderResult:
        output = Path(output_path) if output_path is not None else None

        try:
            import matplotlib

            # Use a non-GUI backend when writing files or when no display is available.
            if output is not None or not os.environ.get("DISPLAY"):
                matplotlib.use("Agg", force=True)
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            matplotlib.rcParams["path.simplify"] = False
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Matplotlib is required for static rendering. Install well-log-os[pdf]."
            ) from exc

        figures = []
        total_pages = 0

        # Keep PDF output crisp in standard viewers:
        # - Embed TrueType fonts instead of Type3 glyphs.
        # - Avoid path simplification artifacts on dense log curves.
        rc_overrides = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": self.dpi,
            "figure.dpi": self.dpi,
            "path.simplify": False,
        }

        with matplotlib.rc_context(rc_overrides):
            pdf = None
            if output is not None and output.suffix.lower() == ".pdf":
                pdf = PdfPages(output)
            try:
                for source_document, section_dataset in document_dataset_pairs:
                    render_document = source_document
                    draw_header = True
                    draw_track_header = True
                    draw_footer = True
                    auto_multisection_strip = (
                        output is not None
                        and output.suffix.lower() == ".pdf"
                        and len(document_dataset_pairs) > 1
                        and source_document.page.continuous
                        and self.continuous_strip_page_height_mm is None
                    )
                    if (
                        output is not None
                        and output.suffix.lower() == ".pdf"
                        and source_document.page.continuous
                        and (
                            self.continuous_strip_page_height_mm is not None
                            or auto_multisection_strip
                        )
                    ):
                        strip_height_mm = (
                            float(self.continuous_strip_page_height_mm)
                            if self.continuous_strip_page_height_mm is not None
                            else float(source_document.page.height_mm)
                        )
                        render_document = self._build_continuous_strip_document_with_height(
                            source_document,
                            strip_height_mm,
                        )
                        draw_header = False
                        draw_footer = False

                    render_document = self._auto_adjust_track_header_height(render_document)
                    layouts = self.layout_engine.layout(render_document, section_dataset)

                    for local_page_number, page_layout in enumerate(layouts, start=1):
                        global_page_number = total_pages + local_page_number
                        fig = plt.figure(
                            figsize=(
                                page_layout.page.width_mm / 25.4,
                                page_layout.page.height_mm / 25.4,
                            ),
                            dpi=self.dpi,
                        )
                        if draw_header:
                            self._draw_header(fig, render_document, section_dataset, page_layout)
                        if draw_footer:
                            self._draw_footer(
                                fig,
                                render_document,
                                page_layout,
                                page_number=global_page_number,
                            )
                        if draw_track_header:
                            top_section_title_height_mm = self._draw_section_title_box(
                                fig, render_document, page_layout
                            )
                            for track_header in page_layout.track_header_top_frames:
                                header_frame = track_header.frame
                                if top_section_title_height_mm > 0:
                                    header_frame = replace(
                                        header_frame,
                                        y_mm=header_frame.y_mm + top_section_title_height_mm,
                                        height_mm=(
                                            header_frame.height_mm - top_section_title_height_mm
                                        ),
                                    )
                                if header_frame.width_mm <= 0 or header_frame.height_mm <= 0:
                                    continue
                                frame = self._normalize_frame(page_layout.page, header_frame)
                                ax = fig.add_axes(frame)
                                self._draw_track_header(
                                    ax, track_header.track, render_document, section_dataset
                                )
                            for track_header in page_layout.track_header_bottom_frames:
                                header_frame = track_header.frame
                                if header_frame.width_mm <= 0 or header_frame.height_mm <= 0:
                                    continue
                                frame = self._normalize_frame(page_layout.page, header_frame)
                                ax = fig.add_axes(frame)
                                self._draw_track_header(
                                    ax, track_header.track, render_document, section_dataset
                                )
                        for track_frame in page_layout.track_frames:
                            if track_frame.frame.width_mm <= 0 or track_frame.frame.height_mm <= 0:
                                continue
                            frame = self._normalize_frame(page_layout.page, track_frame.frame)
                            ax = fig.add_axes(frame)
                            self._draw_track(
                                ax,
                                track_frame.track,
                                render_document,
                                section_dataset,
                                page_layout,
                            )
                        if pdf is not None:
                            pdf.savefig(fig, dpi=self.dpi)
                            plt.close(fig)
                        else:
                            figures.append(fig)

                    total_pages += len(layouts)
            finally:
                if pdf is not None:
                    pdf.close()

        artifact = str(output) if output is not None else figures
        return RenderResult(
            backend="matplotlib",
            page_count=total_pages,
            artifact=artifact,
            output_path=output,
        )

    def _normalize_frame(self, page, frame) -> list[float]:
        left = frame.x_mm / page.width_mm
        bottom = 1.0 - (frame.y_mm + frame.height_mm) / page.height_mm
        width = frame.width_mm / page.width_mm
        height = frame.height_mm / page.height_mm
        return [left, bottom, width, height]

    def _draw_header(self, fig, document, dataset, page_layout) -> None:
        header_style = self._style_section("header")
        header = document.header
        if header.title:
            fig.text(
                float(header_style["title_x"]),
                float(header_style["title_y"]),
                header.title,
                ha="center",
                va="top",
                fontsize=float(header_style["title_fontsize"]),
                fontweight="bold",
            )
        if header.subtitle:
            fig.text(
                float(header_style["subtitle_x"]),
                float(header_style["subtitle_y"]),
                header.subtitle,
                ha="center",
                va="top",
                fontsize=float(header_style["subtitle_fontsize"]),
            )
        if not header.fields:
            return
        start_y = float(header_style["field_start_y"])
        step_y = float(header_style["field_step_y"])
        for index, field in enumerate(header.fields):
            value = dataset.header_value(field.source_key, field.default)
            fig.text(
                float(header_style["field_x"]),
                start_y - index * step_y,
                f"{field.label}: {value}",
                ha="left",
                va="top",
                fontsize=float(header_style["field_fontsize"]),
            )

    def _draw_footer(self, fig, document, page_layout, *, page_number: int | None = None) -> None:
        footer_style = self._style_section("footer")
        if not document.footer.lines:
            return
        for index, line in enumerate(document.footer.lines):
            fig.text(
                float(footer_style["line_x"]),
                float(footer_style["line_start_y"]) + index * float(footer_style["line_step_y"]),
                line,
                ha="left",
                va="bottom",
                fontsize=float(footer_style["line_fontsize"]),
            )
        resolved_page_number = page_layout.page_number if page_number is None else page_number
        fig.text(
            float(footer_style["page_x"]),
            float(footer_style["page_y"]),
            f"Page {resolved_page_number}",
            ha="right",
            va="bottom",
            fontsize=float(footer_style["page_fontsize"]),
        )

    def _draw_track_header(self, ax, track, document, dataset) -> None:
        track_header_style = self._style_section("track_header")
        ax.set_facecolor(str(track_header_style["background_color"]))
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_track_frame(ax)
        slots = self._track_header_slots(track)
        paired_slot = self._curve_header_pair_slot(track, slots)
        raster_triplet_slot = None
        if paired_slot is None:
            raster_triplet_slot = self._raster_header_triplet_slot(track, slots, dataset)
        for index, (item, slot_top, slot_bottom) in enumerate(slots):
            if paired_slot is not None and index == paired_slot[1]:
                continue
            if (
                raster_triplet_slot is not None
                and raster_triplet_slot[0] < index <= raster_triplet_slot[1]
            ):
                continue
            if index > 0:
                ax.plot(
                    [0.0, 1.0],
                    [slot_top, slot_top],
                    transform=ax.transAxes,
                    color=str(track_header_style["separator_color"]),
                    linewidth=float(track_header_style["separator_linewidth"]),
                )
            if not item.enabled:
                continue
            if paired_slot is not None and index == paired_slot[0]:
                self._draw_track_header_curve_pairs(
                    ax,
                    track,
                    document,
                    dataset,
                    paired_slot[2],
                    paired_slot[3],
                )
                continue
            if raster_triplet_slot is not None and index == raster_triplet_slot[0]:
                self._draw_track_header_raster_triplet(
                    ax,
                    track,
                    document,
                    dataset,
                    raster_triplet_slot[2],
                    raster_triplet_slot[3],
                )
                continue
            if item.kind == TrackHeaderObjectKind.TITLE:
                self._draw_track_header_title(ax, track, slot_top, slot_bottom)
            elif item.kind == TrackHeaderObjectKind.SCALE:
                self._draw_track_header_scale(ax, track, document, dataset, slot_top, slot_bottom)
            elif item.kind == TrackHeaderObjectKind.LEGEND:
                self._draw_track_header_legend(ax, track, document, dataset, slot_top, slot_bottom)
            elif item.kind == TrackHeaderObjectKind.DIVISIONS:
                self._draw_track_header_divisions(ax, track, dataset, slot_top, slot_bottom)

    def _curve_header_pair_slot(self, track, slots) -> tuple[int, int, float, float] | None:
        if self._curve_count(track) <= 0:
            return None
        scale_index = None
        legend_index = None
        for index, (item, _, _) in enumerate(slots):
            if not item.enabled or not item.reserve_space:
                continue
            if item.kind == TrackHeaderObjectKind.SCALE:
                scale_index = index
            elif item.kind == TrackHeaderObjectKind.LEGEND:
                legend_index = index
        if scale_index is None or legend_index is None:
            return None
        if abs(scale_index - legend_index) != 1:
            return None
        first = min(scale_index, legend_index)
        second = max(scale_index, legend_index)
        top = slots[first][1]
        bottom = slots[second][2]
        return first, second, top, bottom

    def _raster_header_triplet_slot(
        self,
        track,
        slots,
        dataset: WellDataset,
    ) -> tuple[int, int, float, float] | None:
        if self._curve_count(track) > 0:
            return None
        if self._header_raster_colorbar_target(track, dataset) is None:
            return None

        scale_index = None
        legend_index = None
        division_index = None
        for index, (item, _, _) in enumerate(slots):
            if not item.enabled or not item.reserve_space:
                continue
            if item.kind == TrackHeaderObjectKind.SCALE:
                scale_index = index
            elif item.kind == TrackHeaderObjectKind.LEGEND:
                legend_index = index
            elif item.kind == TrackHeaderObjectKind.DIVISIONS:
                division_index = index

        if scale_index is None or legend_index is None:
            return None

        indices = [scale_index, legend_index]
        if division_index is not None:
            indices.append(division_index)
        first = min(indices)
        last = max(indices)
        top = slots[first][1]
        bottom = slots[last][2]
        return first, last, top, bottom

    def _track_header_slots(self, track) -> tuple[tuple[object, float, float], ...]:
        reserved = track.header.reserved_objects()
        if not reserved:
            return ()
        top = float(self._style_value("track_header", "slot_top"))
        bottom = float(self._style_value("track_header", "slot_bottom"))
        span = top - bottom
        total_units = sum(self._effective_header_line_units(track, item) for item in reserved)
        cursor = top
        slots = []
        for item in reserved:
            item_height = span * (self._effective_header_line_units(track, item) / total_units)
            slot_top = cursor
            slot_bottom = cursor - item_height
            slots.append((item, slot_top, slot_bottom))
            cursor = slot_bottom
        return tuple(slots)

    def _slot_font_size(
        self,
        ax,
        slot_top: float,
        slot_bottom: float,
        *,
        min_pt: float,
        max_pt: float,
    ) -> float:
        slot_height_px = max((slot_top - slot_bottom) * ax.bbox.height, 1.0)
        scale_factor = float(self._style_value("track_header", "font_scale_factor"))
        return max(min_pt, min(max_pt, slot_height_px * scale_factor))

    def _draw_track_header_title(self, ax, track, slot_top: float, slot_bottom: float) -> None:
        track_header_style = self._style_section("track_header")
        fontsize = self._slot_font_size(
            ax,
            slot_top,
            slot_bottom,
            min_pt=float(track_header_style["title_min_pt"]),
            max_pt=float(track_header_style["title_max_pt"]),
        )
        title_align = str(track_header_style.get("title_align", "left")).lower()
        if title_align not in {"left", "center", "right"}:
            title_align = "left"
        title_x = float(track_header_style.get("title_x", track_header_style["text_x"]))
        ax.text(
            title_x,
            0.5 * (slot_top + slot_bottom),
            track.title,
            transform=ax.transAxes,
            ha=title_align,
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=True,
        )

    def _draw_track_header_scale(
        self,
        ax,
        track,
        document,
        dataset,
        slot_top: float,
        slot_bottom: float,
    ) -> None:
        track_header_style = self._style_section("track_header")
        if self._is_reference_track(track):
            scale_text = self._reference_scale_text(track, document)
            fontsize = self._slot_font_size(
                ax,
                slot_top,
                slot_bottom,
                min_pt=float(track_header_style["scale_min_pt"]),
                max_pt=float(track_header_style["scale_max_pt"]),
            )
            ax.text(
                float(track_header_style["text_x"]),
                0.5 * (slot_top + slot_bottom),
                scale_text,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                clip_on=True,
            )
            return

        curves = self._curve_elements(track)
        if not curves:
            rasters = self._raster_elements(track)
            if rasters:
                target = rasters[0]
                channel = dataset.get_channel(target.channel)
                if isinstance(channel, RasterChannel):
                    left_value, unit_text, right_value = self._raster_scale_text_triplet(
                        track,
                        target,
                        channel,
                    )
                    fontsize = self._slot_font_size(
                        ax,
                        slot_top,
                        slot_bottom,
                        min_pt=float(track_header_style["scale_min_pt"]),
                        max_pt=float(track_header_style["scale_max_pt"]),
                    )
                    ax.text(
                        float(track_header_style["scale_left_x"]),
                        0.5 * (slot_top + slot_bottom),
                        left_value,
                        transform=ax.transAxes,
                        ha="left",
                        va="center",
                        fontsize=fontsize,
                        clip_on=True,
                    )
                    ax.text(
                        float(track_header_style["scale_unit_x"]),
                        0.5 * (slot_top + slot_bottom),
                        unit_text,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        clip_on=True,
                    )
                    ax.text(
                        float(track_header_style["scale_right_x"]),
                        0.5 * (slot_top + slot_bottom),
                        right_value,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=fontsize,
                        clip_on=True,
                    )
                    return
            fontsize = self._slot_font_size(
                ax,
                slot_top,
                slot_bottom,
                min_pt=float(track_header_style["scale_min_pt"]),
                max_pt=float(track_header_style["scale_max_pt"]),
            )
            ax.text(
                float(track_header_style["text_x"]),
                0.5 * (slot_top + slot_bottom),
                "Scale: auto",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                clip_on=True,
            )
            return

        row_count = len(curves)
        rows = self._curve_row_bounds(slot_top, slot_bottom, row_count)
        for index, (element, (row_top, row_bottom)) in enumerate(zip(curves, rows, strict=False)):
            y_center = 0.5 * (row_top + row_bottom)
            fontsize = self._slot_font_size(
                ax,
                row_top,
                row_bottom,
                min_pt=float(track_header_style["scale_min_pt"]),
                max_pt=float(track_header_style["scale_max_pt"]),
            )
            left_value, unit_text, right_value = self._curve_scale_text_triplet(
                track,
                element,
                dataset,
            )
            row_color = self._curve_header_color(element)
            ax.plot(
                [0.0, 1.0],
                [row_top, row_top],
                transform=ax.transAxes,
                color=row_color,
                linewidth=max(
                    float(track_header_style["separator_linewidth"]),
                    self._curve_header_line_width(element),
                ),
                linestyle=self._curve_header_line_style(element),
            )
            ax.text(
                float(track_header_style["scale_left_x"]),
                y_center,
                left_value,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                color=row_color,
                clip_on=True,
            )
            ax.text(
                float(track_header_style["scale_unit_x"]),
                y_center,
                unit_text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=row_color,
                clip_on=True,
            )
            ax.text(
                float(track_header_style["scale_right_x"]),
                y_center,
                right_value,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=fontsize,
                color=row_color,
                clip_on=True,
            )
            if index == row_count - 1:
                ax.plot(
                    [0.0, 1.0],
                    [row_bottom, row_bottom],
                    transform=ax.transAxes,
                    color=row_color,
                    linewidth=max(
                        float(track_header_style["separator_linewidth"]),
                        self._curve_header_line_width(element),
                    ),
                    linestyle=self._curve_header_line_style(element),
                )

    def _curve_elements(self, track) -> list[CurveElement]:
        return [element for element in track.elements if isinstance(element, CurveElement)]

    def _raster_elements(self, track) -> list[RasterElement]:
        return [element for element in track.elements if isinstance(element, RasterElement)]

    def _raster_header_label(self, element: RasterElement, channel: RasterChannel) -> str:
        if element.label:
            return element.label
        description = str(channel.description or "").strip()
        if description:
            return description
        return channel.mnemonic

    def _raster_axis_limits(
        self,
        track,
        element: RasterElement,
        channel: RasterChannel,
    ) -> tuple[float, float, str | None]:
        sample_axis = self._resolved_raster_sample_axis(channel, element)
        if element.sample_axis_min is not None and element.sample_axis_max is not None:
            axis_min = float(element.sample_axis_min)
            axis_max = float(element.sample_axis_max)
        elif track.x_scale is not None:
            axis_min = float(track.x_scale.minimum)
            axis_max = float(track.x_scale.maximum)
        else:
            axis_min = float(sample_axis[0])
            axis_max = float(sample_axis[-1])
        if np.isclose(axis_min, axis_max):
            axis_min = float(sample_axis[0])
            axis_max = float(sample_axis[-1])
        unit_text = element.sample_axis_unit or channel.sample_unit or None
        return axis_min, axis_max, unit_text

    def _resolved_raster_sample_axis(
        self,
        channel: RasterChannel,
        element: RasterElement,
    ) -> np.ndarray:
        sample_count = channel.values.shape[1]
        if (
            element.sample_axis_source_origin is not None
            and element.sample_axis_source_step is not None
        ):
            return (
                float(element.sample_axis_source_origin)
                + float(element.sample_axis_source_step) * np.arange(sample_count, dtype=float)
            )
        if channel.sample_axis.shape[0] == sample_count:
            return np.asarray(channel.sample_axis, dtype=float)
        return np.arange(sample_count, dtype=float)

    def _clip_raster_columns_to_window(
        self,
        sample_axis: np.ndarray,
        values: np.ndarray,
        *,
        axis_min: float,
        axis_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = min(axis_min, axis_max)
        upper = max(axis_min, axis_max)
        ascending = sample_axis[0] <= sample_axis[-1]
        working_axis = sample_axis if ascending else sample_axis[::-1]
        working_values = values if ascending else values[:, ::-1]

        left = int(np.searchsorted(working_axis, lower, side="left"))
        right = int(np.searchsorted(working_axis, upper, side="right"))

        if right - left < 2:
            left = max(0, min(left, working_axis.size - 2))
            right = min(working_axis.size, max(right, left + 2))
        if left == 0 and right == working_axis.size and (
            lower > float(working_axis[-1]) or upper < float(working_axis[0])
        ):
            return sample_axis, values

        clipped_axis = working_axis[left:right]
        clipped_values = working_values[:, left:right]
        if ascending:
            return clipped_axis, clipped_values
        return clipped_axis[::-1], clipped_values[:, ::-1]

    def _raster_scale_text_triplet(
        self,
        track,
        element: RasterElement,
        channel: RasterChannel,
    ) -> tuple[str, str, str]:
        axis_min, axis_max, unit_text = self._raster_axis_limits(track, element, channel)
        reverse = bool(track.x_scale.reverse) if track.x_scale is not None else False
        left = axis_max if reverse else axis_min
        right = axis_min if reverse else axis_max
        return f"{left:g}", unit_text or "", f"{right:g}"

    def _resolve_raster_normalization(
        self,
        element: RasterElement,
        *,
        target: str,
    ) -> RasterNormalizationKind:
        if target == "waveform":
            requested = element.waveform_normalization
        else:
            requested = element.normalization
        if requested != RasterNormalizationKind.AUTO:
            return requested

        if element.profile == RasterProfileKind.VDL:
            if target == "waveform":
                return RasterNormalizationKind.TRACE_MAXABS
            return RasterNormalizationKind.GLOBAL_MAXABS
        if element.profile == RasterProfileKind.WAVEFORM:
            if target == "waveform":
                return RasterNormalizationKind.TRACE_MAXABS
            return RasterNormalizationKind.NONE
        return RasterNormalizationKind.NONE

    def _normalize_raster_values(
        self,
        values: np.ndarray,
        mode: RasterNormalizationKind,
    ) -> np.ndarray:
        normalized = np.asarray(values, dtype=float)
        if mode == RasterNormalizationKind.NONE:
            return normalized
        if mode == RasterNormalizationKind.GLOBAL_MAXABS:
            denominator = float(np.nanmax(np.abs(normalized)))
            if np.isfinite(denominator) and not np.isclose(denominator, 0.0):
                return normalized / denominator
            return normalized

        # TRACE_MAXABS mode: normalize each waveform independently, a standard VDL behavior.
        denominators = np.nanmax(np.abs(normalized), axis=1, keepdims=True)
        valid = np.isfinite(denominators) & ~np.isclose(denominators, 0.0)
        safe_denominators = np.where(valid, denominators, 1.0)
        return normalized / safe_denominators

    def _prepare_raster_display_data(
        self,
        depth: np.ndarray,
        values: np.ndarray,
        element: RasterElement,
        *,
        target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        ordered_depth = np.asarray(depth, dtype=float)
        ordered_values = np.asarray(values, dtype=float)
        order = np.argsort(ordered_depth, kind="mergesort")
        ordered_depth = ordered_depth[order]
        ordered_values = ordered_values[order, :]

        if element.profile == RasterProfileKind.VDL:
            masked = np.ma.masked_invalid(ordered_values)
            medians = np.ma.median(masked, axis=1).filled(0.0)
            ordered_values = ordered_values - medians[:, None]

        normalization = self._resolve_raster_normalization(element, target=target)
        return ordered_depth, self._normalize_raster_values(ordered_values, normalization)

    def _resolve_raster_color_limits(
        self,
        values: np.ndarray,
        element: RasterElement,
    ) -> tuple[float, float] | None:
        if element.color_limits is not None:
            return float(element.color_limits[0]), float(element.color_limits[1])

        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return None

        if element.profile == RasterProfileKind.VDL:
            if element.clip_percentiles is not None:
                _, high = element.clip_percentiles
                clip = float(np.nanpercentile(np.abs(finite), high))
            else:
                clip = float(np.nanpercentile(np.abs(finite), 99.0))
            if not np.isfinite(clip) or np.isclose(clip, 0.0):
                clip = float(np.nanmax(np.abs(finite)))
            if not np.isfinite(clip) or np.isclose(clip, 0.0):
                clip = 1.0
            return -clip, clip

        if element.clip_percentiles is not None:
            low, high = element.clip_percentiles
            lower, upper = np.nanpercentile(finite, [low, high])
            if np.isclose(lower, upper):
                lower = float(np.nanmin(finite))
                upper = float(np.nanmax(finite))
            return float(lower), float(upper)

        return None

    def _resolved_raster_colormap(self, element: RasterElement) -> str:
        if (
            element.profile == RasterProfileKind.VDL
            and str(element.style.colormap).strip().lower() == "viridis"
        ):
            # VDL convention: positive amplitudes black, negative amplitudes white.
            return "gray_r"
        return element.style.colormap

    def _draw_raster_waveforms(
        self,
        ax,
        *,
        depth: np.ndarray,
        x_axis: np.ndarray,
        values: np.ndarray,
        waveform: RasterWaveformSpec,
        opacity: float,
    ) -> None:
        if not waveform.enabled:
            return

        y_limits = ax.get_ylim()
        window_top = min(y_limits)
        window_base = max(y_limits)
        selected = self._select_waveform_indices(
            depth,
            window_top=window_top,
            window_base=window_base,
            waveform=waveform,
        )
        if selected.size == 0:
            return

        depth_step = float(np.nanmedian(np.abs(np.diff(depth))))
        if not np.isfinite(depth_step) or np.isclose(depth_step, 0.0):
            depth_step = 1.0
        amplitude = waveform.amplitude_scale * depth_step * waveform.stride

        for depth_index in selected:
            trace = values[depth_index]
            finite = np.isfinite(trace)
            if np.count_nonzero(finite) < 2:
                continue
            x = x_axis[finite]
            trace_values = trace[finite]
            baseline = np.full(trace_values.shape, depth[depth_index], dtype=float)
            y = baseline + trace_values * amplitude
            signed = -trace_values if waveform.invert_fill_polarity else trace_values
            x_fill, y_fill, baseline_fill, signed_fill = self._trace_fill_series(
                x,
                y,
                baseline,
                signed,
            )
            if waveform.fill:
                positive = signed_fill > 0
                negative = signed_fill < 0
                if np.any(positive):
                    ax.fill_between(
                        x_fill,
                        baseline_fill,
                        y_fill,
                        where=positive,
                        facecolor=waveform.positive_fill_color,
                        linewidth=0.0,
                        alpha=opacity,
                        zorder=4.0,
                        interpolate=False,
                    )
                if np.any(negative):
                    ax.fill_between(
                        x_fill,
                        baseline_fill,
                        y_fill,
                        where=negative,
                        facecolor=waveform.negative_fill_color,
                        linewidth=0.0,
                        alpha=opacity,
                        zorder=4.0,
                        interpolate=False,
                    )
            ax.plot(
                x,
                y,
                color=waveform.color,
                linewidth=waveform.line_width,
                alpha=opacity,
                zorder=4.2,
                clip_on=True,
            )

    def _select_waveform_indices(
        self,
        depth: np.ndarray,
        *,
        window_top: float,
        window_base: float,
        waveform: RasterWaveformSpec,
    ) -> np.ndarray:
        finite_indices = np.where(np.isfinite(depth))[0]
        if finite_indices.size == 0:
            return np.asarray([], dtype=int)
        # Anchor sampling from top-of-log depth order, independent of channel storage order.
        ordered = finite_indices[np.argsort(depth[finite_indices], kind="mergesort")]
        selected = ordered[:: waveform.stride]
        if selected.size == 0:
            return selected
        if waveform.max_traces is not None and selected.size > waveform.max_traces:
            downsample = int(np.ceil(selected.size / waveform.max_traces))
            selected = selected[::downsample]
        selected = np.sort(selected)
        depth_values = depth[selected]
        visible = (depth_values >= window_top) & (depth_values <= window_base)
        return selected[visible]

    def _trace_fill_series(
        self,
        x: np.ndarray,
        y: np.ndarray,
        baseline: np.ndarray,
        signed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if x.size < 2:
            return x, y, baseline, signed

        x_out: list[float] = [float(x[0])]
        y_out: list[float] = [float(y[0])]
        baseline_out: list[float] = [float(baseline[0])]
        signed_out: list[float] = [float(signed[0])]

        for index in range(x.size - 1):
            x0 = float(x[index])
            x1 = float(x[index + 1])
            y0 = float(y[index])
            y1 = float(y[index + 1])
            b0 = float(baseline[index])
            b1 = float(baseline[index + 1])
            s0 = float(signed[index])
            s1 = float(signed[index + 1])

            if s0 == 0.0:
                x_out.append(x0)
                y_out.append(y0)
                baseline_out.append(b0)
                signed_out.append(0.0)

            if s0 * s1 < 0.0:
                fraction = abs(s0) / (abs(s0) + abs(s1))
                xc = x0 + (x1 - x0) * fraction
                bc = b0 + (b1 - b0) * fraction
                x_out.append(xc)
                y_out.append(bc)
                baseline_out.append(bc)
                signed_out.append(0.0)

            x_out.append(x1)
            y_out.append(y1)
            baseline_out.append(b1)
            signed_out.append(s1)

        return (
            np.asarray(x_out, dtype=float),
            np.asarray(y_out, dtype=float),
            np.asarray(baseline_out, dtype=float),
            np.asarray(signed_out, dtype=float),
        )

    def _curve_row_bounds(
        self,
        slot_top: float,
        slot_bottom: float,
        row_count: int,
    ) -> tuple[tuple[float, float], ...]:
        if row_count <= 0:
            return ()
        slot_height = slot_top - slot_bottom
        return tuple(
            (
                slot_top - (index * slot_height / row_count),
                slot_top - ((index + 1) * slot_height / row_count),
            )
            for index in range(row_count)
        )

    def _curve_scale_text_triplet(
        self,
        track,
        element: CurveElement,
        dataset: WellDataset,
    ) -> tuple[str, str, str]:
        show_limits = element.header_display.show_limits
        show_unit = element.header_display.show_unit

        channel = dataset.get_channel(element.channel)
        unit_text = ""
        if show_unit and isinstance(channel, ScalarChannel):
            unit_text = channel.value_unit or ""

        if not show_limits:
            return "", unit_text, ""

        scale = element.scale or track.x_scale
        if scale is not None:
            left = scale.maximum if scale.reverse else scale.minimum
            right = scale.minimum if scale.reverse else scale.maximum
            return f"{left:g}", unit_text, f"{right:g}"

        if isinstance(channel, ScalarChannel):
            values = channel.masked_values()
            finite = values[np.isfinite(values)]
            if finite.size >= 1:
                left = float(np.nanmin(finite))
                right = float(np.nanmax(finite))
                return f"{left:g}", unit_text, f"{right:g}"

        return "auto", unit_text, "auto"

    def _curve_header_color(self, element: CurveElement) -> str:
        if element.header_display.show_color:
            return element.style.color
        return "#111111"

    def _curve_header_label(self, element: CurveElement) -> str:
        if not element.header_display.show_name:
            return ""
        return element.label or element.channel

    def _curve_header_line_style(self, element: CurveElement) -> str:
        if not element.header_display.show_color:
            return "-"
        return element.style.line_style

    def _curve_header_line_width(self, element: CurveElement) -> float:
        if not element.header_display.show_color:
            return 0.7
        return max(0.7, element.style.line_width)

    def _header_division_scale(
        self, track, dataset: WellDataset
    ) -> tuple[float, float, ScaleKind] | None:
        if self._uses_independent_curve_scales(track):
            return None

        scale = track.x_scale
        if scale is not None:
            left = scale.maximum if scale.reverse else scale.minimum
            right = scale.minimum if scale.reverse else scale.maximum
            return left, right, scale.kind

        for element in self._curve_elements(track):
            if element.scale is not None:
                left = element.scale.maximum if element.scale.reverse else element.scale.minimum
                right = element.scale.minimum if element.scale.reverse else element.scale.maximum
                return left, right, element.scale.kind

            channel = dataset.get_channel(element.channel)
            if isinstance(channel, ScalarChannel):
                values = channel.masked_values()
                finite = values[np.isfinite(values)]
                if finite.size >= 2:
                    return float(np.nanmin(finite)), float(np.nanmax(finite)), ScaleKind.LINEAR
        return None

    def _header_raster_colorbar_target(
        self,
        track,
        dataset: WellDataset,
    ) -> tuple[RasterElement, RasterChannel] | None:
        for element in self._raster_elements(track):
            if not element.colorbar_enabled:
                continue
            if element.colorbar_position != RasterColorbarPosition.HEADER:
                continue
            if not element.show_raster:
                continue
            channel = dataset.get_channel(element.channel)
            if isinstance(channel, RasterChannel):
                return element, channel
        return None

    def _draw_track_header_raster_colorbar(
        self,
        ax,
        track,
        element: RasterElement,
        channel: RasterChannel,
        slot_top: float,
        slot_bottom: float,
    ) -> None:
        track_header_style = self._style_section("track_header")
        raster_style = self._style_section("raster")
        slot_height = slot_top - slot_bottom
        label_fontsize = self._slot_font_size(
            ax,
            slot_top,
            slot_bottom,
            min_pt=float(track_header_style["scale_min_pt"]),
            max_pt=float(track_header_style["scale_max_pt"]),
        )
        tick_fontsize = max(label_fontsize - 0.6, 2.5)
        bar_left = float(track_header_style["scale_left_x"])
        bar_right = float(track_header_style["scale_right_x"])
        bar_center = slot_bottom + slot_height * float(
            raster_style["header_colorbar_bar_center_y_ratio"]
        )
        bar_height = slot_height * float(raster_style["header_colorbar_bar_height_ratio"])
        bar_bottom = max(slot_bottom, bar_center - 0.5 * bar_height)
        bar_top = min(slot_top, bar_center + 0.5 * bar_height)
        label_y = slot_bottom + slot_height * float(raster_style["header_colorbar_label_y_ratio"])

        _, normalized_values = self._prepare_raster_display_data(
            channel.depth,
            channel.values,
            element,
            target="raster",
        )
        sample_axis = self._resolved_raster_sample_axis(channel, element)
        axis_min, axis_max, _ = self._raster_axis_limits(track, element, channel)
        _, normalized_values = self._clip_raster_columns_to_window(
            sample_axis,
            normalized_values,
            axis_min=axis_min,
            axis_max=axis_max,
        )
        limits = self._resolve_raster_color_limits(normalized_values, element)
        if limits is None:
            finite = normalized_values[np.isfinite(normalized_values)]
            if finite.size >= 2:
                limits = (float(np.nanmin(finite)), float(np.nanmax(finite)))
            else:
                limits = (0.0, 1.0)
        left_value, center_text, right_value = self._raster_header_colorbar_text_triplet(
            element,
            channel,
            limits=limits,
        )

        gradient = np.linspace(0.0, 1.0, 128, dtype=float)[None, :]
        ax.imshow(
            gradient,
            cmap=self._resolved_raster_colormap(element),
            extent=[bar_left, bar_right, bar_bottom, bar_top],
            transform=ax.transAxes,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            zorder=1.0,
        )
        ax.plot(
            [bar_left, bar_right, bar_right, bar_left, bar_left],
            [bar_bottom, bar_bottom, bar_top, bar_top, bar_bottom],
            transform=ax.transAxes,
            color=str(raster_style["header_colorbar_border_color"]),
            linewidth=float(raster_style["header_colorbar_border_linewidth"]),
            zorder=1.2,
        )
        text_color = str(raster_style["header_colorbar_text_color"])
        ax.text(
            bar_left,
            label_y,
            left_value,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=tick_fontsize,
            color=text_color,
            clip_on=True,
        )
        ax.text(
            0.5 * (bar_left + bar_right),
            label_y,
            center_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color=text_color,
            clip_on=True,
        )
        ax.text(
            bar_right,
            label_y,
            right_value,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=tick_fontsize,
            color=text_color,
            clip_on=True,
        )

    def _raster_header_colorbar_text_triplet(
        self,
        element: RasterElement,
        channel: RasterChannel,
        *,
        limits: tuple[float, float],
    ) -> tuple[str, str, str]:
        center_text = element.colorbar_label or channel.value_unit or "Amplitude"
        if element.profile == RasterProfileKind.VDL:
            return "Min", center_text, "Max"
        return f"{limits[0]:g}", center_text, f"{limits[1]:g}"

    def _draw_track_header_divisions(
        self,
        ax,
        track,
        dataset: WellDataset,
        slot_top: float,
        slot_bottom: float,
    ) -> None:
        track_header_style = self._style_section("track_header")
        division_scale = self._header_division_scale(track, dataset)
        if division_scale is None:
            raster_target = self._header_raster_colorbar_target(track, dataset)
            if raster_target is not None:
                element, channel = raster_target
                self._draw_track_header_raster_colorbar(
                    ax,
                    track,
                    element,
                    channel,
                    slot_top,
                    slot_bottom,
                )
                return
            fontsize = self._slot_font_size(
                ax,
                slot_top,
                slot_bottom,
                min_pt=float(track_header_style["scale_min_pt"]),
                max_pt=float(track_header_style["scale_max_pt"]),
            )
            ax.text(
                float(track_header_style["text_x"]),
                0.5 * (slot_top + slot_bottom),
                "-",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                clip_on=True,
            )
            return

        left, right, scale_kind = division_scale
        if np.isclose(left, right):
            return

        tick_count = int(track_header_style.get("division_tick_count", 5))
        tick_count = max(2, tick_count)
        if scale_kind == ScaleKind.LOG:
            if left <= 0 or right <= 0:
                return
            tick_values = np.geomspace(left, right, tick_count)
            left_log = np.log10(left)
            right_log = np.log10(right)
            if np.isclose(left_log, right_log):
                return
            tick_fractions = (np.log10(tick_values) - left_log) / (right_log - left_log)
        elif scale_kind == ScaleKind.TANGENTIAL:
            tick_values = np.linspace(left, right, tick_count)
            tick_fractions = self._grid_segment_positions(tick_count - 1, GridScaleKind.TANGENTIAL)
        else:
            tick_values = np.linspace(left, right, tick_count)
            tick_fractions = np.linspace(0.0, 1.0, tick_count)

        left_x = float(track_header_style["scale_left_x"])
        right_x = float(track_header_style["scale_right_x"])
        tick_x = left_x + (right_x - left_x) * tick_fractions

        slot_height = slot_top - slot_bottom
        axis_y = slot_bottom + slot_height * float(track_header_style["division_axis_y_ratio"])
        label_y = slot_bottom + slot_height * float(track_header_style["division_label_y_ratio"])
        tick_half = 0.5 * slot_height * float(track_header_style["division_tick_length_ratio"])
        tick_color = str(track_header_style["division_tick_color"])
        tick_linewidth = float(track_header_style["division_tick_linewidth"])
        label_fontsize = self._slot_font_size(
            ax,
            slot_top,
            slot_bottom,
            min_pt=float(track_header_style["scale_min_pt"]),
            max_pt=float(track_header_style["scale_max_pt"]),
        )

        ax.plot(
            [left_x, right_x],
            [axis_y, axis_y],
            transform=ax.transAxes,
            color=tick_color,
            linewidth=tick_linewidth,
        )
        for index, (x, value) in enumerate(zip(tick_x, tick_values, strict=False)):
            ax.plot(
                [x, x],
                [axis_y - tick_half, axis_y + tick_half],
                transform=ax.transAxes,
                color=tick_color,
                linewidth=tick_linewidth,
            )
            align = "center"
            if index == 0:
                align = "left"
            elif index == tick_count - 1:
                align = "right"
            ax.text(
                x,
                label_y,
                self._format_number(float(value), NumberFormatKind.AUTOMATIC, 2),
                transform=ax.transAxes,
                ha=align,
                va="center",
                fontsize=label_fontsize,
                color=tick_color,
                clip_on=True,
            )

    def _draw_track_header_raster_triplet(
        self,
        ax,
        track,
        document,
        dataset: WellDataset,
        slot_top: float,
        slot_bottom: float,
    ) -> None:
        target = self._header_raster_colorbar_target(track, dataset)
        if target is None:
            self._draw_track_header_legend(ax, track, document, dataset, slot_top, slot_bottom)
            return

        element, channel = target
        property_group_capacity = self._header_property_group_capacity(document)
        rows = self._curve_row_bounds(slot_top, slot_bottom, property_group_capacity * 3)
        if len(rows) < 3:
            return
        self._draw_track_header_raster_colorbar(
            ax,
            track,
            element,
            channel,
            rows[0][0],
            rows[0][1],
        )
        self._draw_track_header_legend(
            ax,
            track,
            document,
            dataset,
            rows[1][0],
            rows[1][1],
        )
        self._draw_track_header_scale(
            ax,
            track,
            document,
            dataset,
            rows[2][0],
            rows[2][1],
        )

    def _draw_track_header_legend(
        self,
        ax,
        track,
        document,
        dataset: WellDataset,
        slot_top: float,
        slot_bottom: float,
    ) -> None:
        track_header_style = self._style_section("track_header")
        curves = self._curve_elements(track)
        if not curves:
            rasters = self._raster_elements(track)
            if rasters:
                target = rasters[0]
                channel = dataset.get_channel(target.channel)
                if isinstance(channel, RasterChannel):
                    fontsize = self._slot_font_size(
                        ax,
                        slot_top,
                        slot_bottom,
                        min_pt=float(track_header_style["legend_row_min_pt"]),
                        max_pt=float(track_header_style["legend_row_max_pt"]),
                    )
                    label = self._raster_header_label(target, channel)
                    available_px = max(ax.bbox.width * 0.9, 1.0)
                    approx_char_px = max(
                        fontsize * float(track_header_style["legend_char_width_ratio"]),
                        1.0,
                    )
                    max_chars = max(
                        int(track_header_style["legend_min_chars"]),
                        int(available_px / approx_char_px),
                    )
                    if len(label) > max_chars:
                        label = f"{label[: max_chars - 3]}..."
                    ax.text(
                        0.5,
                        0.5 * (slot_top + slot_bottom),
                        label,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        clip_on=True,
                    )
                    return
            fontsize = self._slot_font_size(
                ax,
                slot_top,
                slot_bottom,
                min_pt=float(track_header_style["legend_empty_min_pt"]),
                max_pt=float(track_header_style["legend_empty_max_pt"]),
            )
            ax.text(
                float(track_header_style["text_x"]),
                0.5 * (slot_top + slot_bottom),
                "-",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                clip_on=True,
            )
            return

        row_count = self._curve_header_row_count(document, track)
        rows = self._curve_row_bounds(slot_top, slot_bottom, row_count)
        for element, (_, row_bottom) in zip(curves, rows, strict=False):
            row_color = self._curve_header_color(element)
            ax.plot(
                [0.0, 1.0],
                [row_bottom, row_bottom],
                transform=ax.transAxes,
                color=row_color,
                linewidth=max(
                    float(track_header_style["separator_linewidth"]),
                    self._curve_header_line_width(element),
                ),
                linestyle=self._curve_header_line_style(element),
            )
        for element, (row_top, row_bottom) in zip(curves, rows, strict=False):
            y_center = 0.5 * (row_top + row_bottom)
            fontsize = self._slot_font_size(
                ax,
                row_top,
                row_bottom,
                min_pt=float(track_header_style["legend_row_min_pt"]),
                max_pt=float(track_header_style["legend_row_max_pt"]),
            )
            label = self._curve_header_label(element)
            available_px = max(ax.bbox.width * 0.9, 1.0)
            approx_char_px = max(
                fontsize * float(track_header_style["legend_char_width_ratio"]), 1.0
            )
            max_chars = max(
                int(track_header_style["legend_min_chars"]), int(available_px / approx_char_px)
            )
            if len(label) > max_chars:
                label = f"{label[: max_chars - 3]}..."

            row_color = self._curve_header_color(element)
            ax.text(
                0.5,
                y_center,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=row_color,
                clip_on=True,
            )

    def _draw_track_header_curve_pairs(
        self,
        ax,
        track,
        document,
        dataset: WellDataset,
        slot_top: float,
        slot_bottom: float,
    ) -> None:
        track_header_style = self._style_section("track_header")
        curves = self._curve_elements(track)
        if not curves:
            return

        row_count = self._curve_header_row_count(document, track)
        rows = self._curve_row_bounds(slot_top, slot_bottom, row_count * 2)
        for curve_index, element in enumerate(curves):
            name_top, name_bottom = rows[curve_index * 2]
            scale_top, scale_bottom = rows[curve_index * 2 + 1]
            row_color = self._curve_header_color(element)

            name_fontsize = self._slot_font_size(
                ax,
                name_top,
                name_bottom,
                min_pt=float(track_header_style["legend_row_min_pt"]),
                max_pt=float(track_header_style["legend_row_max_pt"]),
            )
            label = self._curve_header_label(element)
            available_px = max(ax.bbox.width * 0.9, 1.0)
            approx_char_px = max(
                name_fontsize * float(track_header_style["legend_char_width_ratio"]), 1.0
            )
            max_chars = max(
                int(track_header_style["legend_min_chars"]),
                int(available_px / approx_char_px),
            )
            if len(label) > max_chars:
                label = f"{label[: max_chars - 3]}..."
            ax.text(
                0.5,
                0.5 * (name_top + name_bottom),
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=name_fontsize,
                color=row_color,
                clip_on=True,
            )

            # The curve style sample line sits between the name and scale rows.
            ax.plot(
                [
                    float(track_header_style["scale_left_x"]),
                    float(track_header_style["scale_right_x"]),
                ],
                [name_bottom, name_bottom],
                transform=ax.transAxes,
                color=row_color,
                linewidth=max(
                    float(track_header_style["separator_linewidth"]),
                    self._curve_header_line_width(element),
                ),
                linestyle=self._curve_header_line_style(element),
            )

            scale_fontsize = self._slot_font_size(
                ax,
                scale_top,
                scale_bottom,
                min_pt=float(track_header_style["scale_min_pt"]),
                max_pt=float(track_header_style["scale_max_pt"]),
            )
            left_value, unit_text, right_value = self._curve_scale_text_triplet(
                track, element, dataset
            )
            scale_row_height = scale_top - scale_bottom
            y_offset = scale_row_height * float(
                track_header_style.get("paired_scale_text_offset_ratio", 0.08)
            )
            y_center = 0.5 * (scale_top + scale_bottom) - y_offset
            ax.text(
                float(track_header_style["scale_left_x"]),
                y_center,
                left_value,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=scale_fontsize,
                color=row_color,
                clip_on=True,
            )
            ax.text(
                float(track_header_style["scale_unit_x"]),
                y_center,
                unit_text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=scale_fontsize,
                color=row_color,
                clip_on=True,
            )
            ax.text(
                float(track_header_style["scale_right_x"]),
                y_center,
                right_value,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=scale_fontsize,
                color=row_color,
                clip_on=True,
            )

    def _grid_zorder(self, mode: GridDisplayMode) -> float:
        if mode == GridDisplayMode.ABOVE:
            return 3.0
        return 0.25

    def _grid_segment_positions(self, line_count: int, scale: GridScaleKind) -> np.ndarray:
        points = np.linspace(0.0, 1.0, max(line_count, 1) + 1)
        if scale == GridScaleKind.LINEAR:
            return points
        if scale == GridScaleKind.LOGARITHMIC:
            factor = 3.0
            return np.expm1(factor * points) / np.expm1(factor)
        spread = 1.2
        denominator = np.tan(0.5 * spread)
        return 0.5 + np.tan((points - 0.5) * spread) / (2.0 * denominator)

    def _log_scale_grid_fractions(
        self,
        track,
        dataset: WellDataset,
    ) -> tuple[list[float], list[float]] | None:
        reference = self._header_division_scale(track, dataset)
        if reference is None:
            return None
        left, right, scale_kind = reference
        if scale_kind != ScaleKind.LOG:
            return None
        low = min(left, right)
        high = max(left, right)
        if low <= 0 or high <= 0 or np.isclose(low, high):
            return None
        reverse = left > right
        low_log = float(np.log10(low))
        high_log = float(np.log10(high))
        if np.isclose(low_log, high_log):
            return None

        min_decade = int(np.floor(low_log))
        max_decade = int(np.ceil(high_log))
        major_values: list[float] = []
        minor_values: list[float] = []
        for decade in range(min_decade, max_decade + 1):
            decade_base = 10.0**decade
            major_values.append(decade_base)
            for multiplier in range(2, 10):
                minor_values.append(multiplier * decade_base)

        def _to_fractions(values: list[float]) -> list[float]:
            fractions: list[float] = []
            for value in values:
                if value <= low or value >= high:
                    continue
                fraction = (np.log10(value) - low_log) / (high_log - low_log)
                normalized = float(1.0 - fraction) if reverse else float(fraction)
                if 1e-6 < normalized < 1 - 1e-6:
                    fractions.append(normalized)
            return fractions

        return _to_fractions(major_values), _to_fractions(minor_values)

    def _vertical_grid_fractions(
        self, track, dataset: WellDataset
    ) -> tuple[list[float], list[float]]:
        main_points = self._grid_segment_positions(
            track.grid.vertical_main_line_count,
            track.grid.vertical_main_scale,
        )
        main_lines = [float(value) for value in main_points[1:-1]]

        secondary_lines: list[float] = []
        if track.grid.vertical_secondary_line_count > 1:
            secondary_points = self._grid_segment_positions(
                track.grid.vertical_secondary_line_count,
                track.grid.vertical_secondary_scale,
            )[1:-1]
            for start, stop in zip(main_points[:-1], main_points[1:], strict=True):
                segment = stop - start
                for fraction in secondary_points:
                    secondary_lines.append(float(start + segment * fraction))

        def _dedupe(values: list[float], *, blocked: list[float]) -> list[float]:
            deduped: list[float] = []
            for value in values:
                if value <= 1e-6 or value >= 1 - 1e-6:
                    continue
                if any(np.isclose(value, item, atol=1e-6) for item in blocked):
                    continue
                if any(np.isclose(value, item, atol=1e-6) for item in deduped):
                    continue
                deduped.append(value)
            return deduped

        log_fractions = self._log_scale_grid_fractions(track, dataset)
        if log_fractions is not None:
            major_log, minor_log = log_fractions
            if (
                track.grid.vertical_main_spacing_mode == GridSpacingMode.SCALE
                and track.grid.vertical_main_scale == GridScaleKind.LOGARITHMIC
            ):
                main_lines = major_log
            if (
                track.grid.vertical_secondary_spacing_mode == GridSpacingMode.SCALE
                and track.grid.vertical_secondary_scale == GridScaleKind.LOGARITHMIC
            ):
                secondary_lines = minor_log

        return _dedupe(main_lines, blocked=[]), _dedupe(secondary_lines, blocked=main_lines)

    def _draw_vertical_grid_lines(self, ax, track, window, dataset: WellDataset) -> None:
        from matplotlib.transforms import blended_transform_factory

        if track.grid.vertical_display == GridDisplayMode.NONE:
            return

        x_limits = ax.get_xlim()
        reverse_axis = x_limits[0] > x_limits[1]
        main_lines, secondary_lines = self._vertical_grid_fractions(track, dataset)
        if reverse_axis:
            main_lines = [1.0 - value for value in main_lines]
            secondary_lines = [1.0 - value for value in secondary_lines]

        transform = blended_transform_factory(ax.transAxes, ax.transData)
        y_top = float(window.start)
        y_base = float(window.stop)
        zorder = self._grid_zorder(track.grid.vertical_display)
        grid_style = self._style_section("grid")

        if track.grid.vertical_secondary_visible:
            secondary_color = track.grid.vertical_secondary_color or str(
                grid_style["depth_minor_color"]
            )
            secondary_linewidth = (
                track.grid.vertical_secondary_thickness
                if track.grid.vertical_secondary_thickness is not None
                else float(grid_style["x_minor_linewidth"])
            )
            for fraction in secondary_lines:
                ax.plot(
                    [fraction, fraction],
                    [y_top, y_base],
                    transform=transform,
                    color=secondary_color,
                    linewidth=secondary_linewidth,
                    alpha=track.grid.vertical_secondary_alpha,
                    zorder=zorder,
                    clip_on=True,
                )

        if track.grid.vertical_main_visible:
            main_color = track.grid.vertical_main_color or str(grid_style["depth_major_color"])
            main_linewidth = (
                track.grid.vertical_main_thickness
                if track.grid.vertical_main_thickness is not None
                else float(grid_style["x_major_linewidth"])
            )
            for fraction in main_lines:
                ax.plot(
                    [fraction, fraction],
                    [y_top, y_base],
                    transform=transform,
                    color=main_color,
                    linewidth=main_linewidth,
                    alpha=track.grid.vertical_main_alpha,
                    zorder=zorder,
                    clip_on=True,
                )

    def _draw_horizontal_grid_lines(
        self,
        ax,
        track,
        window,
        *,
        major_step: float,
        minor_step: float,
        draw_minor: bool,
    ) -> None:
        from matplotlib.transforms import blended_transform_factory

        if track.grid.horizontal_display == GridDisplayMode.NONE:
            return

        transform = blended_transform_factory(ax.transAxes, ax.transData)
        zorder = self._grid_zorder(track.grid.horizontal_display)
        grid_style = self._style_section("grid")
        y_top = float(window.start)
        y_base = float(window.stop)

        def _depth_values(step: float) -> list[float]:
            start = np.floor(y_top / step) * step
            epsilon = max(abs(step) * 1e-6, 1e-8)
            values: list[float] = []
            value = start
            while value <= y_base + epsilon:
                if value >= y_top - epsilon:
                    values.append(float(value))
                value += step
            return values

        if (
            track.grid.horizontal_minor_visible
            and track.grid.minor
            and draw_minor
            and minor_step > 0
        ):
            minor_color = track.grid.horizontal_minor_color or str(grid_style["depth_minor_color"])
            minor_linewidth = (
                track.grid.horizontal_minor_thickness
                if track.grid.horizontal_minor_thickness is not None
                else float(grid_style["depth_minor_linewidth"])
            )
            minor_alpha = (
                track.grid.horizontal_minor_alpha
                if track.grid.horizontal_minor_alpha is not None
                else float(grid_style["depth_minor_alpha"])
            )
            major_multiple = max(round(major_step / minor_step), 1)
            for value in _depth_values(minor_step):
                tick_index = round(value / minor_step)
                if tick_index % major_multiple == 0:
                    continue
                ax.plot(
                    [0.0, 1.0],
                    [value, value],
                    transform=transform,
                    color=minor_color,
                    linewidth=minor_linewidth,
                    alpha=minor_alpha,
                    zorder=zorder,
                    clip_on=True,
                )

        if track.grid.horizontal_major_visible and track.grid.major and major_step > 0:
            major_color = track.grid.horizontal_major_color or str(grid_style["depth_major_color"])
            major_linewidth = (
                track.grid.horizontal_major_thickness
                if track.grid.horizontal_major_thickness is not None
                else float(grid_style["depth_major_linewidth"])
            )
            major_alpha = (
                track.grid.horizontal_major_alpha
                if track.grid.horizontal_major_alpha is not None
                else float(grid_style["depth_major_alpha"])
            )
            for value in _depth_values(major_step):
                ax.plot(
                    [0.0, 1.0],
                    [value, value],
                    transform=transform,
                    color=major_color,
                    linewidth=major_linewidth,
                    alpha=major_alpha,
                    zorder=zorder,
                    clip_on=True,
                )

    def _draw_track(self, ax, track, document, dataset, page_layout) -> None:
        track_style = self._style_section("track")
        is_reference_track = self._is_reference_track(track)
        window = page_layout.depth_window
        ax.set_ylim(window.stop, window.start)
        ax.set_facecolor(str(track_style["background_color"]))
        ax.set_axisbelow(True)
        self._style_track_frame(ax)
        major_step = max(document.depth_axis.major_step, document.depth_axis.minor_step)
        minor_step = document.depth_axis.minor_step
        draw_minor_grid = True
        if is_reference_track:
            major_step, minor_step, draw_minor_grid = self._resolve_reference_steps(track, document)
        self._configure_depth_axis(
            ax,
            document,
            show_labels=False,
            major_step=major_step,
            minor_step=minor_step,
        )
        reference_grid_mode = str(track_style.get("reference_grid_mode", "edge_ticks")).lower()
        if is_reference_track and reference_grid_mode == "edge_ticks":
            self._draw_reference_edge_ticks(
                ax,
                window,
                major_step=major_step,
                minor_step=minor_step,
                draw_minor=draw_minor_grid,
            )
        else:
            self._draw_horizontal_grid_lines(
                ax,
                track,
                window,
                major_step=major_step,
                minor_step=minor_step,
                draw_minor=draw_minor_grid,
            )

        for zone in document.zones:
            if zone.base < window.start or zone.top > window.stop:
                continue
            zone_top = max(zone.top, window.start)
            zone_base = min(zone.base, window.stop)
            ax.axhspan(zone_top, zone_base, color=zone.fill_color, alpha=zone.alpha, linewidth=0)
        for marker in document.markers:
            if marker.depth < window.start or marker.depth > window.stop:
                continue
            ax.axhline(
                marker.depth,
                color=marker.color,
                linestyle=marker.line_style,
                linewidth=float(track_style["marker_linewidth"]),
            )

        if is_reference_track:
            ax.set_facecolor(str(track_style["depth_background_color"]))
            if track.elements:
                for element in track.elements:
                    if isinstance(element, CurveElement):
                        self._draw_curve(ax, track, element, document, dataset)
                    elif isinstance(element, RasterElement):
                        self._draw_raster(ax, track, element, document, dataset)
                self._configure_x_axis(ax, track)
                self._apply_scale(ax, track)
                self._draw_vertical_grid_lines(ax, track, window, dataset)
                ax.tick_params(
                    axis="x",
                    which="both",
                    top=False,
                    bottom=False,
                    labeltop=False,
                    labelbottom=False,
                )
            else:
                ax.set_xlim(0, 1)
                ax.set_xticks([])
            ax.tick_params(axis="y", length=0, labelleft=False)
            self._draw_reference_values_inside(
                ax,
                track,
                document,
                window,
                major_step=major_step,
            )
            self._draw_marker_callouts(ax, document, window)
            return

        if self._is_annotation_track(track):
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.tick_params(axis="y", length=0, labelleft=False)
            self._draw_marker_callouts(ax, document, window)
            return

        independent_curve_scales = self._uses_independent_curve_scales(track)
        for element in track.elements:
            if isinstance(element, CurveElement):
                self._draw_curve(
                    ax,
                    track,
                    element,
                    document,
                    dataset,
                    independent_curve_scales=independent_curve_scales,
                )
            elif isinstance(element, RasterElement):
                self._draw_raster(ax, track, element, document, dataset)

        if independent_curve_scales:
            self._configure_independent_curve_axis(ax)
        else:
            self._configure_x_axis(ax, track)
            self._apply_scale(ax, track)
        self._draw_vertical_grid_lines(ax, track, window, dataset)
        show_sample_axis = False
        if track.kind == TrackKind.ARRAY:
            show_sample_axis = self._draw_array_sample_axis(ax, track, dataset)
        ax.tick_params(
            axis="x",
            which="both",
            top=False,
            bottom=show_sample_axis,
            labeltop=False,
            labelbottom=show_sample_axis,
        )
        ax.tick_params(axis="y", length=0, labelleft=False)

    def _uses_independent_curve_scales(self, track) -> bool:
        if self._is_reference_track(track) or self._is_annotation_track(track):
            return False
        return self._curve_count(track) > 1

    def _configure_independent_curve_axis(self, ax) -> None:
        import matplotlib.ticker as mticker

        ax.set_xscale("linear")
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    def _draw_array_sample_axis(self, ax, track, dataset: WellDataset) -> bool:
        raster_elements = [
            element for element in track.elements if isinstance(element, RasterElement)
        ]
        if not raster_elements:
            return False
        target = next((element for element in raster_elements if element.sample_axis_enabled), None)
        if target is None:
            return False

        channel = dataset.get_channel(target.channel)
        if not isinstance(channel, RasterChannel):
            return False

        tick_count = max(target.sample_axis_tick_count, 2)
        axis_min, axis_max, axis_unit = self._raster_axis_limits(track, target, channel)
        if np.isclose(axis_min, axis_max):
            return False

        ticks = np.linspace(axis_min, axis_max, tick_count)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])

        raster_style = self._style_section("raster")
        ax.tick_params(
            axis="x",
            which="major",
            length=2.0,
            pad=1.2,
            labelsize=float(raster_style["sample_axis_tick_labelsize"]),
            colors=str(raster_style["sample_axis_tick_color"]),
        )
        label = target.sample_axis_label
        if not label:
            sample_label = str(channel.sample_label or "sample")
            if axis_unit:
                label = f"{sample_label} ({axis_unit})"
            else:
                label = sample_label
        ax.set_xlabel(
            label,
            fontsize=float(raster_style["sample_axis_label_fontsize"]),
            color=str(raster_style["sample_axis_label_color"]),
            labelpad=float(raster_style["sample_axis_label_pad"]),
        )
        return True

    def _tangential_transform_values(self, values: np.ndarray, scale) -> np.ndarray:
        spread = float(self._style_section("track").get("tangential_spread", 1.2))
        spread = min(max(spread, 0.05), 2.6)
        denominator = np.tan(0.5 * spread)
        unit = (values - scale.minimum) / (scale.maximum - scale.minimum)
        transformed = 0.5 + np.tan((unit - 0.5) * spread) / (2.0 * denominator)
        return np.clip(transformed, 0.0, 1.0)

    def _wrap_curve_values(
        self,
        values: np.ndarray,
        scale,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wrapped = np.array(values, dtype=float, copy=True)
        valid_mask = np.isfinite(wrapped)
        wrapped_mask = np.zeros(values.shape, dtype=bool)

        lower = min(scale.minimum, scale.maximum)
        upper = max(scale.minimum, scale.maximum)
        if np.isclose(lower, upper):
            return wrapped, valid_mask & False, wrapped_mask

        if scale.kind == ScaleKind.LOG:
            valid_mask &= wrapped > 0
            if not np.any(valid_mask):
                return wrapped, valid_mask, wrapped_mask
            if lower <= 0 or upper <= 0:
                return wrapped, valid_mask & False, wrapped_mask

            outside = valid_mask & ((wrapped < lower) | (wrapped > upper))
            wrapped_mask = outside
            if not np.any(outside):
                return wrapped, valid_mask, wrapped_mask

            low = float(np.log(lower))
            high = float(np.log(upper))
            period = high - low
            if np.isclose(period, 0.0):
                return wrapped, valid_mask & False, wrapped_mask & False
            outside_log = np.log(wrapped[outside])
            wrapped_log = np.mod(outside_log - low, period) + low
            wrapped[outside] = np.exp(wrapped_log)
            return wrapped, valid_mask, wrapped_mask

        outside = valid_mask & ((wrapped < lower) | (wrapped > upper))
        wrapped_mask = outside
        if not np.any(outside):
            return wrapped, valid_mask, wrapped_mask

        period = upper - lower
        wrapped[outside] = np.mod(wrapped[outside] - lower, period) + lower
        return wrapped, valid_mask, wrapped_mask

    def _normalize_curve_values(
        self,
        values: np.ndarray,
        scale,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(values)
        normalized = np.full(values.shape, np.nan, dtype=float)
        if scale is None:
            finite = values[mask]
            if finite.size < 2:
                return normalized, mask & False
            min_value = float(np.nanmin(finite))
            max_value = float(np.nanmax(finite))
            if np.isclose(min_value, max_value):
                return normalized, mask & False
            scaled = (values[mask] - min_value) / (max_value - min_value)
            normalized[mask] = np.clip(scaled, 0.0, 1.0)
            return normalized, mask

        if scale.kind == ScaleKind.LOG:
            if scale.minimum <= 0 or scale.maximum <= 0 or np.isclose(scale.minimum, scale.maximum):
                return normalized, mask & False
            positive_mask = mask & (values > 0)
            if not np.any(positive_mask):
                return normalized, mask & False
            low = np.log(scale.minimum)
            high = np.log(scale.maximum)
            scaled = (np.log(values[positive_mask]) - low) / (high - low)
            if scale.reverse:
                scaled = 1.0 - scaled
            normalized[positive_mask] = np.clip(scaled, 0.0, 1.0)
            return normalized, positive_mask

        if scale.kind == ScaleKind.TANGENTIAL:
            if np.isclose(scale.minimum, scale.maximum):
                return normalized, mask & False
            scaled = self._tangential_transform_values(values[mask], scale)
            if scale.reverse:
                scaled = 1.0 - scaled
            normalized[mask] = scaled
            return normalized, mask

        if np.isclose(scale.minimum, scale.maximum):
            return normalized, mask & False
        scaled = (values[mask] - scale.minimum) / (scale.maximum - scale.minimum)
        if scale.reverse:
            scaled = 1.0 - scaled
        normalized[mask] = np.clip(scaled, 0.0, 1.0)
        return normalized, mask

    def _curve_plot_data(
        self,
        track,
        element: CurveElement,
        document,
        dataset,
        *,
        independent_curve_scales: bool,
    ) -> _CurvePlotData:
        channel = dataset.get_channel(element.channel)
        if not isinstance(channel, ScalarChannel):
            raise TypeError(f"Curve element {element.channel} requires a scalar channel.")

        depth = channel.depth_in(document.depth_axis.unit, self.registry)
        values = channel.masked_values()
        scale = element.scale or track.x_scale
        plot_values = values
        valid_mask = np.isfinite(values)
        wrapped_mask = np.zeros(values.shape, dtype=bool)

        if element.wrap and scale is not None:
            plot_values, valid_mask, wrapped_mask = self._wrap_curve_values(values, scale)

        if scale is not None and scale.kind == ScaleKind.TANGENTIAL:
            plot_values, normalized_mask = self._normalize_curve_values(plot_values, scale)
            valid_mask &= normalized_mask
            wrapped_mask &= valid_mask
            return _CurvePlotData(depth, values, plot_values, valid_mask, wrapped_mask, scale)

        if independent_curve_scales:
            plot_values, normalized_mask = self._normalize_curve_values(plot_values, scale)
            valid_mask &= normalized_mask
            wrapped_mask &= valid_mask
            return _CurvePlotData(depth, values, plot_values, valid_mask, wrapped_mask, scale)

        if scale is not None and scale.kind == ScaleKind.LOG:
            valid_mask &= plot_values > 0
            wrapped_mask &= valid_mask

        return _CurvePlotData(depth, values, plot_values, valid_mask, wrapped_mask, scale)

    def _scales_match(self, first, second) -> bool:
        if first is None and second is None:
            return True
        if (first is None) != (second is None):
            return False
        assert first is not None and second is not None
        return (
            first.kind == second.kind
            and np.isclose(first.minimum, second.minimum)
            and np.isclose(first.maximum, second.maximum)
            and first.reverse == second.reverse
        )

    def _curves_share_fill_axis(
        self,
        track,
        primary: CurveElement,
        secondary: CurveElement,
        *,
        independent_curve_scales: bool,
    ) -> bool:
        primary_scale = primary.scale or track.x_scale
        secondary_scale = secondary.scale or track.x_scale
        if independent_curve_scales and (primary_scale is None or secondary_scale is None):
            return False
        return self._scales_match(primary_scale, secondary_scale)

    def _find_track_curves(
        self,
        track,
        channel_name: str,
        *,
        exclude: CurveElement,
    ) -> list[CurveElement]:
        target = channel_name.strip().upper()
        matches: list[CurveElement] = []
        for element in self._curve_elements(track):
            if element is exclude:
                continue
            if element.channel.upper() == target:
                matches.append(element)
        return matches

    def _find_track_curve_by_id(
        self,
        track,
        element_id: str,
        *,
        exclude: CurveElement,
    ) -> CurveElement | None:
        target = element_id.strip()
        for element in self._curve_elements(track):
            if element is exclude:
                continue
            if element.id == target:
                return element
        return None

    def _align_curve_fill_values(
        self,
        reference_depth: np.ndarray,
        reference_mask: np.ndarray,
        source_data: _CurvePlotData,
    ) -> np.ndarray:
        if (
            source_data.depth.shape == reference_depth.shape
            and np.allclose(source_data.depth, reference_depth, equal_nan=True)
        ):
            return np.where(source_data.valid_mask, source_data.plot_values, np.nan)

        aligned = np.full(reference_depth.shape, np.nan, dtype=float)
        mask = source_data.valid_mask & np.isfinite(source_data.depth) & np.isfinite(
            source_data.plot_values
        )
        if np.count_nonzero(mask) < 2:
            return aligned

        source_depth = source_data.depth[mask]
        source_values = source_data.plot_values[mask]
        order = np.argsort(source_depth, kind="mergesort")
        source_depth = source_depth[order]
        source_values = source_values[order]
        unique_depth, unique_indices = np.unique(source_depth, return_index=True)
        if unique_depth.size < 2:
            return aligned
        unique_values = source_values[unique_indices]
        interpolated = np.interp(
            reference_depth,
            unique_depth,
            unique_values,
            left=np.nan,
            right=np.nan,
        )
        return np.where(reference_mask, interpolated, np.nan)

    def _resolved_curve_fill_color(self, element: CurveElement) -> str:
        assert element.fill is not None
        if element.fill.color is not None:
            return element.fill.color
        if element.style.fill_color is not None:
            return element.style.fill_color
        return element.style.color

    def _resolved_curve_fill_alpha(self, element: CurveElement) -> float:
        assert element.fill is not None
        if element.fill.alpha is not None:
            return element.fill.alpha
        return element.style.fill_alpha

    def _draw_curve_fill(
        self,
        ax,
        track,
        element: CurveElement,
        document,
        dataset,
        *,
        independent_curve_scales: bool,
    ) -> None:
        if element.fill is None:
            return
        if element.wrap:
            raise TemplateValidationError(
                f"Curve fill for {element.channel!r} does not support wrapped curves yet."
            )
        if element.fill.kind == CurveFillKind.BETWEEN_CURVES:
            assert element.fill.other_channel is not None
            candidates = self._find_track_curves(
                track,
                element.fill.other_channel,
                exclude=element,
            )
            if not candidates:
                raise TemplateValidationError(
                    f"Curve fill for {element.channel!r} references missing channel "
                    f"{element.fill.other_channel!r} in track {track.id!r}."
                )
            compatible = [
                candidate
                for candidate in candidates
                if self._curves_share_fill_axis(
                    track,
                    element,
                    candidate,
                    independent_curve_scales=independent_curve_scales,
                )
            ]
            if not compatible:
                raise TemplateValidationError(
                    f"Curve fill between {element.channel!r} and {element.fill.other_channel!r} "
                    "requires matching effective scales."
                )
            other = compatible[0]
        elif element.fill.kind == CurveFillKind.BETWEEN_INSTANCES:
            assert element.fill.other_element_id is not None
            other = self._find_track_curve_by_id(
                track,
                element.fill.other_element_id,
                exclude=element,
            )
            if other is None:
                raise TemplateValidationError(
                    f"Curve fill for {element.channel!r} references missing curve id "
                    f"{element.fill.other_element_id!r} in track {track.id!r}."
                )
        else:
            raise TemplateValidationError(
                f"Curve fill kind {element.fill.kind!s} is not implemented yet."
            )
        if other.wrap:
            raise TemplateValidationError(
                f"Curve fill for {element.channel!r} cannot target wrapped curve "
                f"{other.channel!r} yet."
            )

        primary_data = self._curve_plot_data(
            track,
            element,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )
        secondary_data = self._curve_plot_data(
            track,
            other,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )
        secondary_values = self._align_curve_fill_values(
            primary_data.depth,
            primary_data.valid_mask,
            secondary_data,
        )
        valid_mask = (
            primary_data.valid_mask
            & np.isfinite(primary_data.plot_values)
            & np.isfinite(secondary_values)
        )
        if not np.any(valid_mask):
            return

        alpha = self._resolved_curve_fill_alpha(element)
        if element.fill.crossover.enabled:
            left_color = element.fill.crossover.left_color or self._resolved_curve_fill_color(
                element
            )
            right_color = element.fill.crossover.right_color or self._resolved_curve_fill_color(
                element
            )
            crossover_alpha = (
                element.fill.crossover.alpha
                if element.fill.crossover.alpha is not None
                else alpha
            )
            left_mask = valid_mask & (primary_data.plot_values < secondary_values)
            right_mask = valid_mask & (primary_data.plot_values > secondary_values)
            if np.any(left_mask):
                ax.fill_betweenx(
                    primary_data.depth,
                    primary_data.plot_values,
                    secondary_values,
                    where=left_mask,
                    facecolor=left_color,
                    alpha=crossover_alpha,
                    linewidth=0.0,
                    interpolate=True,
                )
            if np.any(right_mask):
                ax.fill_betweenx(
                    primary_data.depth,
                    primary_data.plot_values,
                    secondary_values,
                    where=right_mask,
                    facecolor=right_color,
                    alpha=crossover_alpha,
                    linewidth=0.0,
                    interpolate=True,
                )
            return

        ax.fill_betweenx(
            primary_data.depth,
            primary_data.plot_values,
            secondary_values,
            where=valid_mask,
            facecolor=self._resolved_curve_fill_color(element),
            alpha=alpha,
            linewidth=0.0,
            interpolate=True,
        )

    def _plot_curve_with_wrap_segments(
        self,
        ax,
        depth: np.ndarray,
        x_values: np.ndarray,
        valid_mask: np.ndarray,
        wrapped_mask: np.ndarray,
        element,
    ) -> None:
        base_mask = valid_mask & ~wrapped_mask
        if np.any(base_mask):
            base_x = np.where(base_mask, x_values, np.nan)
            base_y = np.where(base_mask, depth, np.nan)
            ax.plot(
                base_x,
                base_y,
                color=element.style.color,
                linewidth=element.style.line_width,
                linestyle=element.style.line_style,
                alpha=element.style.opacity,
            )

        if not np.any(wrapped_mask):
            return
        wrap_color = element.wrap_color or element.style.color
        wrapped_x = np.where(wrapped_mask, x_values, np.nan)
        wrapped_y = np.where(wrapped_mask, depth, np.nan)
        ax.plot(
            wrapped_x,
            wrapped_y,
            color=wrap_color,
            linewidth=element.style.line_width,
            linestyle=element.style.line_style,
            alpha=element.style.opacity,
        )

    def _draw_curve(
        self,
        ax,
        track,
        element,
        document,
        dataset,
        *,
        independent_curve_scales: bool = False,
    ) -> None:
        plot_data = self._curve_plot_data(
            track,
            element,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )
        depth = plot_data.depth
        values = plot_data.raw_values
        scale = plot_data.scale

        if element.render_mode == "line":
            self._draw_curve_fill(
                ax,
                track,
                element,
                document,
                dataset,
                independent_curve_scales=independent_curve_scales,
            )

        if scale is not None and scale.kind == ScaleKind.TANGENTIAL:
            if element.render_mode == "value_labels":
                self._draw_curve_value_labels(
                    ax,
                    depth,
                    plot_data.plot_values,
                    element,
                    scale,
                    text_values=values,
                    value_mask=plot_data.valid_mask,
                )
            else:
                self._plot_curve_with_wrap_segments(
                    ax,
                    depth,
                    plot_data.plot_values,
                    plot_data.valid_mask,
                    plot_data.wrapped_mask,
                    element,
                )
            if scale.reverse:
                ax.set_xlim(1.0, 0.0)
            else:
                ax.set_xlim(0.0, 1.0)
            return

        if independent_curve_scales:
            if element.render_mode == "value_labels":
                self._draw_curve_value_labels(
                    ax,
                    depth,
                    plot_data.plot_values,
                    element,
                    scale,
                    text_values=values,
                    value_mask=plot_data.valid_mask,
                )
                return
            self._plot_curve_with_wrap_segments(
                ax,
                depth,
                plot_data.plot_values,
                plot_data.valid_mask,
                plot_data.wrapped_mask,
                element,
            )
            return

        if scale is None:
            xmin = float(np.nanmin(values))
            xmax = float(np.nanmax(values))
        else:
            xmin = scale.minimum
            xmax = scale.maximum
        if element.render_mode == "value_labels":
            self._draw_curve_value_labels(
                ax,
                depth,
                plot_data.plot_values,
                element,
                scale,
                text_values=values if element.wrap else None,
                value_mask=plot_data.valid_mask,
            )
        elif scale is not None and scale.kind == ScaleKind.LOG:
            ax.set_xscale("log")
            self._plot_curve_with_wrap_segments(
                ax,
                depth,
                plot_data.plot_values,
                plot_data.valid_mask,
                plot_data.wrapped_mask,
                element,
            )
        else:
            self._plot_curve_with_wrap_segments(
                ax,
                depth,
                plot_data.plot_values,
                plot_data.valid_mask,
                plot_data.wrapped_mask,
                element,
            )
        if scale is not None and scale.reverse:
            ax.set_xlim(xmax, xmin)
        else:
            ax.set_xlim(xmin, xmax)

    def _draw_curve_value_labels(
        self,
        ax,
        depth,
        values,
        element,
        scale,
        *,
        text_values=None,
        value_mask=None,
    ) -> None:
        labels = element.value_labels
        mask = np.isfinite(depth) & np.isfinite(values)
        if value_mask is not None:
            mask &= value_mask
        if scale is not None and scale.kind == ScaleKind.LOG:
            mask &= values > 0
        if not np.any(mask):
            return

        valid_depth = depth[mask]
        valid_plot_values = values[mask]
        if text_values is None:
            valid_text_values = valid_plot_values
        else:
            valid_text_values = text_values[mask]
        step = labels.step
        window_top = float(min(ax.get_ylim()))
        window_base = float(max(ax.get_ylim()))
        start = np.floor(window_top / step) * step
        epsilon = max(abs(step) * 1e-6, 1e-8)

        sample_indices: list[int] = []
        used: set[int] = set()
        target = start
        while target <= window_base + epsilon:
            if target >= window_top - epsilon:
                nearest = int(np.argmin(np.abs(valid_depth - target)))
                if nearest not in used:
                    used.add(nearest)
                    sample_indices.append(nearest)
            target += step

        for index in sample_indices:
            x_value = float(valid_plot_values[index])
            text_value = float(valid_text_values[index])
            y_value = float(valid_depth[index])
            text = self._format_number(text_value, labels.number_format, labels.precision)
            kwargs = {
                "ha": labels.horizontal_alignment,
                "va": labels.vertical_alignment,
                "fontsize": labels.font_size,
                "color": labels.color or element.style.color,
                "fontweight": labels.font_weight,
                "fontstyle": labels.font_style,
                "clip_on": True,
            }
            if labels.font_family:
                kwargs["fontfamily"] = labels.font_family
            ax.text(x_value, y_value, text, **kwargs)

    def _draw_raster(self, ax, track, element, document, dataset) -> None:
        channel = dataset.get_channel(element.channel)
        if not isinstance(channel, RasterChannel):
            raise TypeError(f"Raster element {element.channel} requires a raster channel.")
        axis_min, axis_max, _ = self._raster_axis_limits(track, element, channel)
        sample_axis = self._resolved_raster_sample_axis(channel, element)
        depth_source = channel.depth_in(document.depth_axis.unit, self.registry)
        raster_depth, raster_values = self._prepare_raster_display_data(
            depth_source,
            channel.values,
            element,
            target="raster",
        )
        waveform_depth, waveform_values = self._prepare_raster_display_data(
            depth_source,
            channel.values,
            element,
            target="waveform",
        )
        raster_axis, raster_values = self._clip_raster_columns_to_window(
            sample_axis,
            raster_values,
            axis_min=axis_min,
            axis_max=axis_max,
        )
        waveform_axis, waveform_values = self._clip_raster_columns_to_window(
            sample_axis,
            waveform_values,
            axis_min=axis_min,
            axis_max=axis_max,
        )
        extent = [
            float(np.nanmin(raster_axis)),
            float(np.nanmax(raster_axis)),
            float(np.nanmax(raster_depth)),
            float(np.nanmin(raster_depth)),
        ]
        image_kwargs = {
            "aspect": "auto",
            "extent": extent,
            "cmap": self._resolved_raster_colormap(element),
            "interpolation": element.interpolation,
            "origin": "upper",
        }
        show_raster = element.show_raster and element.profile != RasterProfileKind.WAVEFORM
        image = None
        if show_raster:
            resolved_limits = self._resolve_raster_color_limits(raster_values, element)
            if resolved_limits is not None:
                if element.profile == RasterProfileKind.VDL:
                    import matplotlib.colors as mcolors

                    image_kwargs["norm"] = mcolors.TwoSlopeNorm(
                        vmin=resolved_limits[0],
                        vcenter=0.0,
                        vmax=resolved_limits[1],
                    )
                else:
                    image_kwargs["vmin"], image_kwargs["vmax"] = resolved_limits
            image = ax.imshow(raster_values, alpha=element.raster_alpha, **image_kwargs)
        if (
            image is not None
            and element.colorbar_enabled
            and element.colorbar_position == RasterColorbarPosition.RIGHT
        ):
            self._draw_raster_colorbar(ax, image, element, channel)
        self._draw_raster_waveforms(
            ax,
            depth=waveform_depth,
            x_axis=waveform_axis,
            values=waveform_values,
            waveform=element.waveform,
            opacity=element.style.opacity,
        )
        if track.x_scale is not None:
            if track.x_scale.reverse:
                ax.set_xlim(track.x_scale.maximum, track.x_scale.minimum)
            else:
                ax.set_xlim(track.x_scale.minimum, track.x_scale.maximum)

    def _draw_raster_colorbar(self, ax, image, element, channel: RasterChannel) -> None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        raster_style = self._style_section("raster")
        width_ratio = max(float(raster_style["colorbar_width_ratio"]), 1e-3)
        pad_ratio = max(float(raster_style["colorbar_pad_ratio"]), 0.0)
        width_ratio = min(width_ratio, 0.4)
        x_anchor = max(1.0 - width_ratio - pad_ratio, 0.0)
        cax = inset_axes(
            ax,
            width=f"{width_ratio * 100:.3f}%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(x_anchor, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        colorbar = ax.figure.colorbar(image, cax=cax)
        colorbar.ax.tick_params(
            labelsize=float(raster_style["colorbar_tick_labelsize"]),
            colors=str(raster_style["colorbar_tick_color"]),
            length=1.8,
            pad=0.8,
        )
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor(str(raster_style["colorbar_tick_color"]))
        label = element.colorbar_label or channel.value_unit or ""
        if label:
            colorbar.set_label(
                label,
                fontsize=float(raster_style["colorbar_label_fontsize"]),
                color=str(raster_style["colorbar_label_color"]),
                labelpad=1.5,
            )

    def _apply_scale(self, ax, track) -> None:
        if track.x_scale is None:
            return
        scale = track.x_scale
        if scale.kind == ScaleKind.LOG:
            ax.set_xscale("log")
            if scale.reverse:
                ax.set_xlim(scale.maximum, scale.minimum)
            else:
                ax.set_xlim(scale.minimum, scale.maximum)
            return
        if scale.kind == ScaleKind.TANGENTIAL:
            ax.set_xscale("linear")
            if scale.reverse:
                ax.set_xlim(1.0, 0.0)
            else:
                ax.set_xlim(0.0, 1.0)
            return
        if scale.reverse:
            ax.set_xlim(scale.maximum, scale.minimum)
        else:
            ax.set_xlim(scale.minimum, scale.maximum)

    def _configure_depth_axis(
        self,
        ax,
        document,
        *,
        show_labels: bool,
        major_step: float | None = None,
        minor_step: float | None = None,
    ) -> None:
        import matplotlib.ticker as mticker

        resolved_major_step = (
            max(document.depth_axis.major_step, document.depth_axis.minor_step)
            if major_step is None
            else max(major_step, 1e-12)
        )
        resolved_minor_step = (
            document.depth_axis.minor_step if minor_step is None else max(minor_step, 1e-12)
        )
        ax.yaxis.set_major_locator(mticker.MultipleLocator(resolved_major_step))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(resolved_minor_step))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.tick_params(axis="y", which="major", length=0, labelleft=show_labels)
        ax.tick_params(axis="y", which="minor", length=0, labelleft=False)

    def _configure_x_axis(self, ax, track) -> None:
        import matplotlib.ticker as mticker

        if track.x_scale is not None and track.x_scale.kind == ScaleKind.LOG:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            return
        if track.x_scale is not None and track.x_scale.kind == ScaleKind.TANGENTIAL:
            ax.set_xscale("linear")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
            return

        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    def _draw_depth_grid(self, ax, *, show_minor: bool = True) -> None:
        grid_style = self._style_section("grid")
        ax.grid(
            True,
            axis="y",
            which="major",
            color=str(grid_style["depth_major_color"]),
            linewidth=float(grid_style["depth_major_linewidth"]),
            alpha=float(grid_style["depth_major_alpha"]),
        )
        if not show_minor:
            return
        ax.grid(
            True,
            axis="y",
            which="minor",
            color=str(grid_style["depth_minor_color"]),
            linewidth=float(grid_style["depth_minor_linewidth"]),
            alpha=float(grid_style["depth_minor_alpha"]),
        )

    def _style_track_frame(self, ax) -> None:
        track_style = self._style_section("track")
        for spine in ax.spines.values():
            spine.set_color(str(track_style["frame_color"]))
            spine.set_linewidth(float(track_style["frame_linewidth"]))

    def _draw_marker_callouts(self, ax, document, window) -> None:
        if not document.markers:
            return
        from matplotlib.transforms import blended_transform_factory

        marker_style = self._style_section("markers")
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        y_offset = max(document.depth_axis.minor_step * 0.4, 1.0)
        for marker in document.markers:
            if marker.depth < window.start or marker.depth > window.stop:
                continue
            if not marker.label:
                continue
            ax.annotate(
                marker.label,
                xy=(float(marker_style["callout_anchor_x"]), marker.depth),
                xycoords=transform,
                xytext=(float(marker_style["callout_text_x"]), marker.depth - y_offset),
                textcoords=transform,
                fontsize=float(marker_style["callout_fontsize"]),
                color=str(marker_style["callout_text_color"]),
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": str(marker_style["callout_arrow_style"]),
                    "color": marker.color,
                    "lw": float(marker_style["callout_arrow_linewidth"]),
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
