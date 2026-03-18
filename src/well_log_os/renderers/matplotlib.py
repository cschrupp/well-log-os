from __future__ import annotations

import math
import os
import re
import textwrap
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..errors import DependencyUnavailableError, TemplateValidationError
from ..layout import LayoutEngine
from ..model import (
    AnnotationArrowSpec,
    AnnotationGlyphSpec,
    AnnotationIntervalSpec,
    AnnotationLabelMode,
    AnnotationMarkerSpec,
    AnnotationTextSpec,
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
    ReferenceCurveOverlayMode,
    ReferenceCurveOverlaySpec,
    ReferenceCurveTickSide,
    ReferenceEventSpec,
    ReferenceTrackSpec,
    ReportBlockSpec,
    ReportDetailCellSpec,
    ReportDetailSpec,
    ReportServiceTitleSpec,
    ReportValueSpec,
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
    x_is_fractional: bool = False


@dataclass(slots=True)
class _CurveFillRenderData:
    primary: _CurvePlotData
    secondary_values: np.ndarray
    valid_mask: np.ndarray


@dataclass(slots=True)
class _CurveCalloutRenderRecord:
    label: str
    side: str
    allow_side_flip: bool
    curve_key: int
    anchor_x: float
    anchor_y: float
    text_x: float
    desired_text_y: float
    color: str
    font_size: float
    font_weight: str
    font_style: str
    arrow: bool
    arrow_style: str
    arrow_linewidth: float
    placed_side: str | None = None
    text_y: float | None = None


@dataclass(slots=True)
class _AnnotationLabelRecord:
    label: str
    anchor_x: float
    anchor_y: float
    preferred_x: float
    preferred_y: float
    color: str
    font_size: float
    font_weight: str
    font_style: str
    priority: int
    arrow: bool
    arrow_style: str
    arrow_linewidth: float
    rotation: float = 0.0
    label_mode: AnnotationLabelMode = AnnotationLabelMode.FREE
    label_lane_start: float | None = None
    label_lane_end: float | None = None
    side: str | None = None
    allow_side_flip: bool = False
    placed_side: str | None = None
    text_x: float | None = None
    text_y: float | None = None
    display_label: str | None = None


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

    def _resolved_reference_overlay(
        self,
        track,
        element: CurveElement,
    ) -> ReferenceCurveOverlaySpec | None:
        if not self._is_reference_track(track):
            return None
        if element.reference_overlay is not None:
            return element.reference_overlay
        return ReferenceCurveOverlaySpec()

    def _reference_overlay_lane(
        self,
        overlay: ReferenceCurveOverlaySpec,
    ) -> tuple[float, float]:
        if overlay.lane_start is not None and overlay.lane_end is not None:
            return float(overlay.lane_start), float(overlay.lane_end)
        track_style = self._style_section("track")
        if overlay.mode == ReferenceCurveOverlayMode.INDICATOR:
            return (
                float(track_style["reference_overlay_indicator_lane_start"]),
                float(track_style["reference_overlay_indicator_lane_end"]),
            )
        return (
            float(track_style["reference_overlay_curve_lane_start"]),
            float(track_style["reference_overlay_curve_lane_end"]),
        )

    def _reference_overlay_tick_length_ratio(
        self,
        overlay: ReferenceCurveOverlaySpec,
    ) -> float:
        if overlay.tick_length_ratio is not None:
            return float(overlay.tick_length_ratio)
        return float(self._style_section("track")["reference_overlay_tick_length_ratio"])

    def _reference_overlay_threshold(
        self,
        overlay: ReferenceCurveOverlaySpec,
    ) -> float:
        if overlay.threshold is not None:
            return float(overlay.threshold)
        return float(self._style_section("track")["reference_overlay_threshold"])

    def _reference_event_tick_length_ratio(self, event: ReferenceEventSpec) -> float:
        if event.tick_length_ratio is not None:
            return float(event.tick_length_ratio)
        return float(self._style_section("track")["reference_overlay_tick_length_ratio"])

    def _reference_event_segments(
        self,
        event: ReferenceEventSpec,
    ) -> tuple[tuple[float, float], ...]:
        if event.lane_start is not None and event.lane_end is not None:
            return ((float(event.lane_start), float(event.lane_end)),)
        tick_length = self._reference_event_tick_length_ratio(event)
        if event.tick_side == ReferenceCurveTickSide.LEFT:
            return ((0.0, tick_length),)
        if event.tick_side == ReferenceCurveTickSide.RIGHT:
            return ((1.0 - tick_length, 1.0),)
        return ((0.0, tick_length), (1.0 - tick_length, 1.0))

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

    def _reference_event_elements(self, track) -> tuple[ReferenceEventSpec, ...]:
        if not self._is_reference_track(track):
            return ()
        reference = track.reference
        if reference is None:
            return ()
        return tuple(reference.events)

    def _fill_header_elements(self, track) -> list[CurveElement]:
        return [
            element
            for element in track.elements
            if isinstance(element, CurveElement) and element.fill is not None
        ]

    def _fill_header_count(self, track) -> int:
        return len(self._fill_header_elements(track))

    def _document_curve_row_capacity(self, document: LogDocument) -> int:
        return max((self._curve_count(track) for track in document.tracks), default=0)

    def _document_fill_row_capacity(self, document: LogDocument) -> int:
        return max((self._fill_header_count(track) for track in document.tracks), default=0)

    def _header_property_group_capacity(self, document: LogDocument) -> int:
        return max(1, self._document_curve_row_capacity(document))

    def _curve_header_row_count(self, document: LogDocument, track) -> int:
        count = self._curve_count(track)
        if count <= 0:
            return 0
        return max(count, self._header_property_group_capacity(document))

    def _fill_header_row_count(self, document: LogDocument, track) -> int:
        capacity = self._document_fill_row_capacity(document)
        if capacity <= 0:
            return 0
        return max(self._fill_header_count(track), capacity)

    def _effective_header_line_units(self, track, header_item) -> int:
        if not header_item.enabled or not header_item.reserve_space:
            return header_item.line_units
        if header_item.kind == TrackHeaderObjectKind.SCALE:
            if self._is_reference_track(track):
                return header_item.line_units
            return max(header_item.line_units, self._curve_count(track))
        if header_item.kind == TrackHeaderObjectKind.LEGEND:
            if self._is_reference_track(track) and self._curve_count(track) > 0:
                return max(header_item.line_units, self._curve_count(track) * 2)
            return max(header_item.line_units, self._curve_count(track)) + self._fill_header_count(
                track
            )
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
                report_document = document_dataset_pairs[0][0] if document_dataset_pairs else None
                report_dataset = document_dataset_pairs[0][1] if document_dataset_pairs else None
                report_block = (
                    report_document.header.report
                    if report_document is not None and report_document.header.report is not None
                    else None
                )
                if (
                    report_document is not None
                    and report_dataset is not None
                    and report_block is not None
                    and report_block.enabled
                ):
                    report_fig = plt.figure(
                        figsize=self._report_page_size_inches(report_document.page),
                        dpi=self.dpi,
                    )
                    self._draw_report_page(
                        report_fig,
                        report_block,
                        report_dataset,
                        compact=False,
                        frame=(
                            float(self.style["report"]["heading_frame_x"]),
                            float(self.style["report"]["heading_frame_y"]),
                            float(self.style["report"]["heading_frame_width"]),
                            float(self.style["report"]["heading_frame_height"]),
                        ),
                    )
                    if pdf is not None:
                        pdf.savefig(report_fig, dpi=self.dpi)
                        plt.close(report_fig)
                    else:
                        figures.append(report_fig)
                    total_pages += 1
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
                if (
                    report_document is not None
                    and report_dataset is not None
                    and report_block is not None
                    and report_block.tail_enabled
                ):
                    report_tail_fig = plt.figure(
                        figsize=self._report_page_size_inches(report_document.page),
                        dpi=self.dpi,
                    )
                    self._draw_report_page(
                        report_tail_fig,
                        report_block,
                        report_dataset,
                        compact=True,
                        frame=(
                            float(self.style["report"]["tail_frame_x"]),
                            float(self.style["report"]["tail_frame_y"]),
                            float(self.style["report"]["tail_frame_width"]),
                            float(self.style["report"]["tail_frame_height"]),
                        ),
                    )
                    if pdf is not None:
                        pdf.savefig(report_tail_fig, dpi=self.dpi)
                        plt.close(report_tail_fig)
                    else:
                        figures.append(report_tail_fig)
                    total_pages += 1
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

    def _report_page_size_inches(self, page) -> tuple[float, float]:
        return float(page.width_mm) / 25.4, float(page.height_mm) / 25.4

    def _report_page_transform(self, ax, *, rotated: bool):
        if not rotated:
            return ax.transAxes
        from matplotlib.transforms import Affine2D

        return Affine2D().rotate_deg(-90).translate(0.0, 1.0) + ax.transAxes

    def _resolve_report_value(self, value: ReportValueSpec, dataset: WellDataset) -> str:
        if value.source_key is not None:
            fallback = value.value if value.value is not None else value.default
            return str(dataset.header_value(value.source_key, fallback))
        if value.value is not None:
            return str(value.value)
        return str(value.default)

    def _report_summary_fields(
        self,
        report: ReportBlockSpec,
        dataset: WellDataset,
    ) -> list[tuple[str, str]]:
        desired_keys = ("company", "well", "field", "county", "country")
        fields_by_key = {field.key: field for field in report.general_fields}
        resolved: list[tuple[str, str]] = []
        for key in desired_keys:
            field = fields_by_key.get(key)
            if field is None:
                continue
            value = self._resolve_report_value(field.value, dataset).strip()
            if not value:
                continue
            resolved.append((field.label, value))
        return resolved

    def _report_general_table_fields(
        self,
        report: ReportBlockSpec,
        dataset: WellDataset,
    ) -> list[tuple[str, str]]:
        summary_keys = {"company", "well", "field", "county", "country"}
        rows: list[tuple[str, str]] = []
        for field in report.general_fields:
            if field.key in summary_keys:
                continue
            value = self._resolve_report_value(field.value, dataset).strip()
            rows.append((field.label, value))
        return rows

    def _report_service_titles(
        self,
        report: ReportBlockSpec,
        dataset: WellDataset,
    ) -> list[tuple[ReportServiceTitleSpec, str]]:
        titles: list[tuple[ReportServiceTitleSpec, str]] = []
        for item in report.service_titles:
            value = self._resolve_report_value(item.value, dataset).strip()
            if value:
                titles.append((item, value))
        return titles

    def _measure_report_text_bbox(
        self,
        ax,
        *,
        renderer,
        text: str,
        text_x: float,
        text_y: float,
        transform,
        fontsize: float,
        fontweight: str,
        fontstyle: str,
        horizontal_alignment: str,
        vertical_alignment: str,
        rotation: float,
    ):
        artist = ax.text(
            text_x,
            text_y,
            text,
            transform=transform,
            fontsize=fontsize,
            fontweight=fontweight,
            fontstyle=fontstyle,
            ha=horizontal_alignment,
            va=vertical_alignment,
            rotation=rotation,
            rotation_mode="anchor",
            alpha=0.0,
        )
        try:
            return artist.get_window_extent(renderer=renderer)
        finally:
            artist.remove()

    def _fit_report_text_fontsize(
        self,
        ax,
        *,
        text: str,
        text_x: float,
        text_y: float,
        transform,
        base_fontsize: float,
        available_width_ratio: float,
        available_height_ratio: float,
        fontweight: str,
        fontstyle: str,
        horizontal_alignment: str,
        vertical_alignment: str,
        rotation: float,
        auto_adjust: bool,
        min_fontsize: float = 5.0,
    ) -> float:
        if not text:
            return base_fontsize
        if not auto_adjust:
            return base_fontsize

        renderer = self._curve_callout_renderer(ax)
        axes_bbox = ax.get_window_extent(renderer=renderer)
        max_width_px = max(float(axes_bbox.width) * max(available_width_ratio, 0.01), 1.0)
        max_height_px = max(float(axes_bbox.height) * max(available_height_ratio, 0.01), 1.0)
        normalized_rotation = abs(float(rotation)) % 180.0
        if np.isclose(normalized_rotation, 90.0):
            max_width_px, max_height_px = max_height_px, max_width_px
        fontsize = float(base_fontsize)
        lower_bound = min(float(min_fontsize), fontsize)
        while fontsize > lower_bound + 1e-6:
            bbox = self._measure_report_text_bbox(
                ax,
                renderer=renderer,
                text=text,
                text_x=text_x,
                text_y=text_y,
                transform=transform,
                fontsize=fontsize,
                fontweight=fontweight,
                fontstyle=fontstyle,
                horizontal_alignment=horizontal_alignment,
                vertical_alignment=vertical_alignment,
                rotation=rotation,
            )
            if bbox.width <= max_width_px and bbox.height <= max_height_px:
                return fontsize
            fontsize = max(lower_bound, fontsize - 0.5)
        return fontsize

    def _draw_report_service_title_lines(
        self,
        ax,
        titles: list[tuple[ReportServiceTitleSpec, str]],
        *,
        box: tuple[float, float, float, float],
        row_step: float,
        start_y: float,
        fallback_fontsize: float,
        transform,
        text_rotation: float,
    ) -> None:
        if not titles:
            return

        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"
        padding = min(0.02, 0.18 * box[2])
        available_width_ratio = max(box[2] - 2.0 * padding, 0.01)
        available_height_ratio = max(min(row_step * 0.82, box[3]), 0.03)
        for index, (title_spec, title) in enumerate(titles):
            alignment = title_spec.alignment
            if alignment == "center":
                text_x = box[0] + 0.5 * box[2]
            elif alignment == "right":
                text_x = box[0] + box[2] - padding
            else:
                text_x = box[0] + padding
            text_y = start_y - index * max(row_step, 0.055)
            base_fontsize = float(
                title_spec.font_size if title_spec.font_size is not None else fallback_fontsize
            )
            fontweight = "bold" if title_spec.bold else "normal"
            fontstyle = "italic" if title_spec.italic else "normal"
            fontsize = self._fit_report_text_fontsize(
                ax,
                text=title,
                text_x=text_x,
                text_y=text_y,
                transform=transform,
                base_fontsize=base_fontsize,
                available_width_ratio=available_width_ratio,
                available_height_ratio=available_height_ratio,
                fontweight=fontweight,
                fontstyle=fontstyle,
                horizontal_alignment=alignment,
                vertical_alignment="top",
                rotation=text_rotation,
                auto_adjust=title_spec.auto_adjust,
            )
            ax.text(
                text_x,
                text_y,
                title,
                ha=alignment,
                va="top",
                fontsize=fontsize,
                fontweight=fontweight,
                fontstyle=fontstyle,
                color="#111111",
                **text_kwargs,
            )

    def _report_field_map(
        self,
        report: ReportBlockSpec,
        dataset: WellDataset,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for field in report.general_fields:
            resolved[field.key] = self._resolve_report_value(field.value, dataset).strip()
        return resolved

    def _report_location_lines(self, value: str) -> tuple[str, str, str]:
        text = value.strip()
        if not text:
            return ("", "", "")
        match = re.search(
            r"Lat:\s*([^,]+?)(?:\s+Long:|\s+Lon:|,)\s*(.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return (
                f"Lat: {match.group(1).strip()}",
                f"Long: {match.group(2).strip()}",
                "",
            )
        return (text, "", "")

    def _draw_report_outer_frame(self, ax, report_style: dict[str, Any]) -> None:
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                facecolor=str(report_style["background_color"]),
                edgecolor=str(report_style["border_color"]),
                linewidth=float(report_style["border_linewidth"]),
                transform=ax.transAxes,
                zorder=0.1,
            )
        )

    def _draw_report_cover_section(
        self,
        ax,
        report: ReportBlockSpec,
        dataset: WellDataset,
        report_style: dict[str, Any],
        *,
        transform,
        text_rotation: float,
    ) -> None:
        from matplotlib.patches import Rectangle

        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"

        def draw_box(x: float, y: float, width: float, height: float) -> None:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor="none",
                    edgecolor=str(report_style["border_color"]),
                    linewidth=float(report_style["border_linewidth"]),
                    transform=transform,
                    zorder=0.15,
                )
            )

        def draw_subrows(
            x: float,
            y: float,
            width: float,
            height: float,
            rows: list[tuple[str, str]],
            *,
            label_ratio: float = 0.30,
            line_only_value: bool = True,
        ) -> None:
            if not rows:
                return
            row_h = height / len(rows)
            label_x = x + 0.012
            value_x = x + width * label_ratio
            for idx, (label, value) in enumerate(rows):
                row_top = y + height - idx * row_h
                row_center = row_top - 0.5 * row_h
                if idx:
                    ax.plot(
                        [x, x + width],
                        [row_top, row_top],
                        transform=transform,
                        color="#6b6b6b",
                        lw=0.35,
                    )
                ax.text(
                    label_x,
                    row_center,
                    f"{label}:",
                    ha="left",
                    va="center",
                    fontsize=float(report_style["field_label_fontsize"]),
                    color="#111111",
                    **text_kwargs,
                )
                if line_only_value and not value:
                    ax.plot(
                        [value_x, x + width - 0.012],
                        [row_center, row_center],
                        transform=transform,
                        color="#444444",
                        lw=0.45,
                    )
                if value:
                    ax.text(
                        value_x + 0.004,
                        row_center,
                        value,
                        ha="left",
                        va="center",
                        fontsize=float(report_style["field_value_fontsize"]),
                        color="#111111",
                        **text_kwargs,
                    )

        fields = self._report_field_map(report, dataset)
        x0, y0, width, height = 0.02, 0.56, 0.96, 0.42
        left_width = width * 0.28
        right_width = width - left_width
        row1_h = height * 0.34
        row2_h = height * 0.18
        row3_h = height * 0.23
        row4_h = height - row1_h - row2_h - row3_h
        top = y0 + height

        logo_box = (x0, top - row1_h, left_width, row1_h)
        titles_box = (x0 + left_width, top - row1_h, right_width, row1_h)
        archive_box = (x0, top - row1_h - row2_h, left_width, row2_h)
        identity_box = (x0 + left_width, top - row1_h - row2_h, right_width, row2_h)
        scale_box = (x0, top - row1_h - row2_h - row3_h, left_width, row3_h)
        coord_mid_width = right_width * 0.74
        coords_box = (
            x0 + left_width,
            top - row1_h - row2_h - row3_h,
            coord_mid_width,
            row3_h,
        )
        services_box = (
            x0 + left_width + coord_mid_width,
            top - row1_h - row2_h - row3_h,
            right_width - coord_mid_width,
            row3_h,
        )
        footer_box = (x0, y0, width, row4_h)

        for box in (
            logo_box,
            titles_box,
            archive_box,
            identity_box,
            scale_box,
            coords_box,
            services_box,
            footer_box,
        ):
            draw_box(*box)

        provider_text = (report.provider_name or fields.get("company") or "Company").strip()
        ax.text(
            logo_box[0] + logo_box[2] * 0.5,
            logo_box[1] + logo_box[3] * 0.5,
            provider_text,
            ha="center",
            va="center",
            fontsize=float(report_style["summary_value_fontsize"]),
            fontweight="bold",
            color="#222222",
            **text_kwargs,
        )
        ax.text(
            logo_box[0] + logo_box[2] * 0.5,
            logo_box[1] + 0.08 * logo_box[3],
            "Logo Placeholder",
            ha="center",
            va="bottom",
            fontsize=max(5.0, float(report_style["field_label_fontsize"]) - 1.0),
            color="#666666",
            **text_kwargs,
        )

        service_titles = self._report_service_titles(report, dataset)
        if service_titles:
            start_y = titles_box[1] + titles_box[3] - 0.08 * titles_box[3]
            step = min(0.12 * titles_box[3], 0.28 * titles_box[3] / max(len(service_titles), 1))
            self._draw_report_service_title_lines(
                ax,
                service_titles,
                box=titles_box,
                row_step=max(step, 0.055),
                start_y=start_y,
                fallback_fontsize=float(report_style["service_fontsize"]),
                transform=transform,
                text_rotation=text_rotation,
            )

        draw_subrows(
            *archive_box,
            [
                ("Archive No", fields.get("archive_no", "")),
                ("API No", fields.get("api_no", "")),
            ],
            label_ratio=0.40,
        )

        draw_subrows(
            *identity_box,
            [
                ("Company", fields.get("company", "")),
                ("Well", fields.get("well", "")),
                ("Field", fields.get("field", "")),
                ("County", fields.get("county", fields.get("province", ""))),
            ],
            label_ratio=0.28,
        )

        ax.text(
            scale_box[0] + 0.012,
            scale_box[1] + scale_box[3] - 0.18 * scale_box[3],
            f"Version: {fields.get('version', '')}".rstrip(),
            ha="left",
            va="top",
            fontsize=float(report_style["field_value_fontsize"]),
            color="#111111",
            **text_kwargs,
        )
        ax.text(
            scale_box[0] + 0.012,
            scale_box[1] + 0.36 * scale_box[3],
            "Scale:",
            ha="left",
            va="center",
            fontsize=float(report_style["field_label_fontsize"]),
            color="#111111",
            **text_kwargs,
        )
        if fields.get("scale"):
            ax.text(
                scale_box[0] + 0.16,
                scale_box[1] + 0.36 * scale_box[3],
                fields["scale"],
                ha="left",
                va="center",
                fontsize=float(report_style["field_value_fontsize"]),
                color="#111111",
                **text_kwargs,
            )

        lat_line, long_line, third_line = self._report_location_lines(fields.get("location", ""))
        ax.text(
            coords_box[0] + 0.012,
            coords_box[1] + coords_box[3] - 0.12 * coords_box[3],
            "Coordinates:",
            ha="left",
            va="top",
            fontsize=float(report_style["field_label_fontsize"]),
            color="#111111",
            fontweight="bold",
            **text_kwargs,
        )
        coord_lines = [line for line in (lat_line, long_line, third_line) if line]
        if not coord_lines:
            coord_lines = ["", "", ""]
        for index, line in enumerate(coord_lines[:3]):
            ax.text(
                coords_box[0] + 0.012,
                coords_box[1] + coords_box[3] - 0.38 * coords_box[3] - index * 0.22 * coords_box[3],
                line,
                ha="left",
                va="top",
                fontsize=float(report_style["field_value_fontsize"]),
                color="#111111",
                **text_kwargs,
            )

        ax.text(
            services_box[0] + 0.012,
            services_box[1] + services_box[3] - 0.12 * services_box[3],
            "Services",
            ha="left",
            va="top",
            fontsize=float(report_style["field_label_fontsize"]),
            color="#111111",
            fontweight="bold",
            **text_kwargs,
        )

        left_footer_w = width * 0.28
        mid_footer_w = width * 0.46
        left_footer_x = footer_box[0]
        mid_footer_x = left_footer_x + left_footer_w
        right_footer_x = mid_footer_x + mid_footer_w
        ax.plot(
            [mid_footer_x, mid_footer_x],
            [footer_box[1], footer_box[1] + footer_box[3]],
            transform=transform,
            color="#3d3d3d",
            lw=0.45,
        )
        ax.plot(
            [right_footer_x, right_footer_x],
            [footer_box[1], footer_box[1] + footer_box[3]],
            transform=transform,
            color="#3d3d3d",
            lw=0.45,
        )
        footer_rows = [
            ("Measured From", fields.get("measured_from", "")),
            ("Log Measured From", fields.get("log_measured_from", "")),
            ("Perforation Measured From", fields.get("perforation_measured_from", "")),
        ]
        row_h = footer_box[3] / len(footer_rows)
        for index, (label, value) in enumerate(footer_rows):
            row_top = footer_box[1] + footer_box[3] - index * row_h
            row_center = row_top - 0.5 * row_h
            if index:
                ax.plot(
                    [footer_box[0], right_footer_x],
                    [row_top, row_top],
                    transform=transform,
                    color="#6b6b6b",
                    lw=0.35,
                )
            ax.text(
                left_footer_x + 0.012,
                row_center,
                label,
                ha="left",
                va="center",
                fontsize=float(report_style["field_label_fontsize"]),
                color="#111111",
                **text_kwargs,
            )
            if value:
                ax.text(
                    mid_footer_x + 0.012,
                    row_center,
                    value,
                    ha="left",
                    va="center",
                    fontsize=float(report_style["field_value_fontsize"]),
                    color="#111111",
                    **text_kwargs,
                )
        altitude_rows = [
            ("Altitudes", ""),
            ("KB", fields.get("elevation_kb", "")),
            ("GL", fields.get("elevation_gl", "")),
            ("DF", fields.get("elevation_df", "")),
        ]
        altitude_row_h = footer_box[3] / len(altitude_rows)
        for row_index in range(1, len(altitude_rows)):
            divider_y = footer_box[1] + row_index * altitude_row_h
            ax.plot(
                [right_footer_x, footer_box[0] + footer_box[2]],
                [divider_y, divider_y],
                transform=transform,
                color="#6b6b6b",
                lw=0.35,
            )
        for row_index, (label, value) in enumerate(altitude_rows):
            row_center_y = footer_box[1] + footer_box[3] - (row_index + 0.5) * altitude_row_h
            if row_index == 0:
                ax.text(
                    right_footer_x + 0.012,
                    row_center_y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=float(report_style["field_label_fontsize"]),
                    color="#111111",
                    fontweight="bold",
                    **text_kwargs,
                )
                continue
            row_text = label if not value else f"{label}  {value}"
            ax.text(
                right_footer_x + 0.012,
                row_center_y,
                row_text,
                ha="left",
                va="center",
                fontsize=float(report_style["field_value_fontsize"]),
                color="#111111",
                **text_kwargs,
            )

    def _draw_report_summary_band(
        self,
        ax,
        report: ReportBlockSpec,
        dataset: WellDataset,
        report_style: dict[str, Any],
        *,
        compact: bool,
        transform,
        text_rotation: float,
    ) -> None:
        from matplotlib.patches import Rectangle

        band_x = 0.0
        band_y = 0.78 if not compact else 0.43
        band_w = 1.0
        band_h = 0.22 if not compact else 0.57
        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"
        ax.add_patch(
            Rectangle(
                (band_x, band_y),
                band_w,
                band_h,
                facecolor=str(report_style["summary_band_color"]),
                edgecolor=str(report_style["border_color"]),
                linewidth=float(report_style["border_linewidth"]),
                transform=transform,
                zorder=0.2,
            )
        )
        if report.provider_name:
            ax.text(
                band_x + band_w - 0.02,
                band_y + band_h - 0.06,
                report.provider_name,
                ha="right",
                va="top",
                fontsize=float(report_style["provider_fontsize"]),
                color=str(report_style["summary_text_color"]),
                fontweight="bold",
                zorder=0.3,
                **text_kwargs,
            )
        summary_rows = self._report_summary_fields(report, dataset)
        if not summary_rows:
            return
        label_size = (
            float(report_style["tail_label_fontsize"])
            if compact
            else float(report_style["summary_label_fontsize"])
        )
        value_size = (
            float(report_style["tail_value_fontsize"])
            if compact
            else float(report_style["summary_value_fontsize"])
        )
        top = band_y + band_h - (0.15 if report.provider_name else 0.08)
        step = min(0.12, max(0.055, (band_h - 0.16) / max(len(summary_rows), 1)))
        label_x = band_x + 0.02
        value_x = band_x + (0.16 if compact else 0.20)
        for index, (label, value) in enumerate(summary_rows):
            y = top - index * step
            ax.text(
                label_x,
                y,
                f"{label}:",
                ha="left",
                va="top",
                fontsize=label_size,
                color=str(report_style["summary_text_color"]),
                zorder=0.3,
                **text_kwargs,
            )
            ax.text(
                value_x,
                y,
                value,
                ha="left",
                va="top",
                fontsize=value_size,
                color=str(report_style["summary_text_color"]),
                fontweight="bold" if compact else "normal",
                zorder=0.3,
                **text_kwargs,
            )

    def _draw_report_tail_section(
        self,
        ax,
        report: ReportBlockSpec,
        dataset: WellDataset,
        report_style: dict[str, Any],
        *,
        transform,
        text_rotation: float,
    ) -> None:
        from matplotlib.patches import Rectangle

        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"

        def draw_box(x: float, y: float, width: float, height: float) -> None:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor="none",
                    edgecolor=str(report_style["border_color"]),
                    linewidth=float(report_style["border_linewidth"]),
                    transform=transform,
                    zorder=0.15,
                )
            )

        fields = self._report_field_map(report, dataset)
        x0, y0, width, height = 0.01, 0.08, 0.98, 0.84
        draw_box(x0, y0, width, height)

        logo_width = width * 0.21
        summary_width = width * 0.44
        detail_width = width - logo_width - summary_width

        logo_box = (x0, y0, logo_width, height)
        summary_box = (x0 + logo_width, y0, summary_width, height)
        detail_box = (x0 + logo_width + summary_width, y0, detail_width, height)
        services_height = detail_box[3] * 0.64
        services_box = (
            detail_box[0],
            detail_box[1] + detail_box[3] - services_height,
            detail_box[2],
            services_height,
        )
        scale_box = (
            detail_box[0],
            detail_box[1],
            detail_box[2],
            detail_box[3] - services_height,
        )

        for box in (logo_box, summary_box, detail_box, services_box, scale_box):
            draw_box(*box)

        provider_text = (report.provider_name or fields.get("company") or "Company").strip()
        ax.text(
            logo_box[0] + logo_box[2] * 0.5,
            logo_box[1] + logo_box[3] * 0.62,
            provider_text,
            ha="center",
            va="center",
            fontsize=float(report_style["summary_value_fontsize"]),
            fontweight="bold",
            color="#222222",
            **text_kwargs,
        )
        ax.text(
            logo_box[0] + logo_box[2] * 0.5,
            logo_box[1] + logo_box[3] * 0.27,
            "Logo Placeholder",
            ha="center",
            va="center",
            fontsize=max(5.0, float(report_style["field_label_fontsize"]) - 1.0),
            color="#666666",
            **text_kwargs,
        )

        summary_rows = self._report_summary_fields(report, dataset)
        row_h = summary_box[3] / max(len(summary_rows), 1)
        for idx, (label, value) in enumerate(summary_rows):
            row_top = summary_box[1] + summary_box[3] - idx * row_h
            row_center = row_top - 0.5 * row_h
            if idx:
                ax.plot(
                    [summary_box[0], summary_box[0] + summary_box[2]],
                    [row_top, row_top],
                    transform=transform,
                    color="#6b6b6b",
                    lw=0.35,
                )
            ax.text(
                summary_box[0] + 0.012,
                row_center,
                f"{label}:",
                ha="left",
                va="center",
                fontsize=float(report_style["field_label_fontsize"]),
                color="#111111",
                **text_kwargs,
            )
            ax.text(
                summary_box[0] + 0.22,
                row_center,
                value,
                ha="left",
                va="center",
                fontsize=float(report_style["field_value_fontsize"]),
                color="#111111",
                **text_kwargs,
            )

        ax.text(
            services_box[0] + 0.012,
            services_box[1] + services_box[3] - 0.14 * services_box[3],
            "Services",
            ha="left",
            va="top",
            fontsize=float(report_style["field_label_fontsize"]),
            color="#111111",
            fontweight="bold",
            **text_kwargs,
        )
        service_titles = self._report_service_titles(report, dataset)
        if service_titles:
            start_y = services_box[1] + services_box[3] - 0.38 * services_box[3]
            step = min(0.23 * services_box[3], 0.58 * services_box[3] / max(len(service_titles), 1))
            self._draw_report_service_title_lines(
                ax,
                service_titles,
                box=services_box,
                row_step=max(step, 0.055),
                start_y=start_y,
                fallback_fontsize=float(
                    report_style.get("tail_service_fontsize", report_style["service_fontsize"])
                ),
                transform=transform,
                text_rotation=text_rotation,
            )

        ax.text(
            scale_box[0] + 0.012,
            scale_box[1] + scale_box[3] - 0.16 * scale_box[3],
            "Scale",
            ha="left",
            va="top",
            fontsize=float(report_style["field_label_fontsize"]),
            color="#111111",
            fontweight="bold",
            **text_kwargs,
        )
        ax.text(
            scale_box[0] + 0.012,
            scale_box[1] + 0.32 * scale_box[3],
            fields.get("scale", ""),
            ha="left",
            va="center",
            fontsize=float(report_style["field_value_fontsize"]),
            color="#111111",
            **text_kwargs,
        )

    def _draw_report_general_fields(
        self,
        ax,
        report: ReportBlockSpec,
        dataset: WellDataset,
        report_style: dict[str, Any],
        *,
        transform,
        text_rotation: float,
    ) -> None:
        from matplotlib.patches import Rectangle

        rows = self._report_general_table_fields(report, dataset)
        if not rows:
            return
        x0, y0, width, height = 0.02, 0.61, 0.96, 0.13
        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                facecolor="none",
                edgecolor=str(report_style["border_color"]),
                linewidth=float(report_style["border_linewidth"]),
                transform=transform,
                zorder=0.15,
            )
        )
        columns = 2 if len(rows) > 6 else 1
        rows_per_col = int(np.ceil(len(rows) / columns))
        row_step = height / max(rows_per_col, 1)
        for row_index in range(rows_per_col):
            if row_index == 0:
                continue
            y = y0 + height - row_index * row_step
            ax.plot([x0, x0 + width], [y, y], transform=transform, color="#4a4a4a", lw=0.35)
        if columns == 2:
            divider_x = x0 + width * 0.5
            ax.plot(
                [divider_x, divider_x],
                [y0, y0 + height],
                transform=transform,
                color="#4a4a4a",
                lw=0.45,
            )
        for index, (label, value) in enumerate(rows):
            col = index // rows_per_col
            row = index % rows_per_col
            col_x = x0 + col * (width / columns)
            y = y0 + height - (row + 0.5) * row_step
            ax.text(
                col_x + 0.012,
                y,
                f"{label}:",
                ha="left",
                va="center",
                fontsize=float(report_style["field_label_fontsize"]),
                color="#222222",
                **text_kwargs,
            )
            ax.text(
                col_x + (width / columns) * 0.42,
                y,
                value,
                ha="left",
                va="center",
                fontsize=float(report_style["field_value_fontsize"]),
                color="#111111",
                **text_kwargs,
            )

    def _draw_report_service_titles(
        self,
        ax,
        report: ReportBlockSpec,
        dataset: WellDataset,
        report_style: dict[str, Any],
        *,
        compact: bool,
        transform,
        text_rotation: float,
    ) -> None:
        titles = self._report_service_titles(report, dataset)
        if not titles:
            return
        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"
        if compact:
            start_y = 0.33
            step = min(0.12, 0.26 / max(len(titles), 1))
            self._draw_report_service_title_lines(
                ax,
                titles,
                box=(0.0, 0.0, 1.0, 1.0),
                row_step=max(step, 0.055),
                start_y=start_y,
                fallback_fontsize=float(report_style["tail_service_fontsize"]),
                transform=transform,
                text_rotation=text_rotation,
            )
            return
        start_y = 0.56
        step = min(0.04, 0.10 / max(len(titles), 1))
        self._draw_report_service_title_lines(
            ax,
            titles,
            box=(0.10, 0.0, 0.88, 1.0),
            row_step=max(step, 0.04),
            start_y=start_y,
            fallback_fontsize=float(report_style["service_fontsize"]),
            transform=transform,
            text_rotation=text_rotation,
        )

    def _draw_report_detail_table(
        self,
        ax,
        detail: ReportDetailSpec,
        dataset: WellDataset,
        report_style: dict[str, Any],
        *,
        transform,
        text_rotation: float,
    ) -> None:
        from matplotlib.patches import Rectangle

        x0, y0, width, height = 0.02, 0.02, 0.96, 0.50
        text_kwargs: dict[str, Any] = {"transform": transform}
        if text_rotation:
            text_kwargs["rotation"] = text_rotation
            text_kwargs["rotation_mode"] = "anchor"
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                facecolor="none",
                edgecolor=str(report_style["border_color"]),
                linewidth=float(report_style["border_linewidth"]),
                transform=transform,
                zorder=0.15,
            )
        )
        title_height = 0.04
        ax.text(
            x0 + 0.01,
            y0 + height - 0.015,
            detail.title or "",
            ha="left",
            va="top",
            fontsize=float(report_style["detail_header_fontsize"]),
            fontweight="bold",
            color="#111111",
            **text_kwargs,
        )
        header_y = y0 + height - title_height - 0.03
        ax.plot(
            [x0, x0 + width],
            [header_y, header_y],
            transform=transform,
            color="#3d3d3d",
            lw=0.5,
        )
        label_w = 0.32 * width
        value_w = width - label_w
        col_count = (
            len(detail.column_titles)
            if detail.column_titles
            else len(detail.rows[0].columns)
        )
        for col in range(col_count + 1):
            x = x0 + label_w + (value_w / col_count) * col
            ax.plot([x, x], [y0, y0 + height], transform=transform, color="#3d3d3d", lw=0.45)
        ax.plot(
            [x0 + label_w, x0 + label_w],
            [y0, y0 + height],
            transform=transform,
            color="#3d3d3d",
            lw=0.55,
        )
        row_count = len(detail.rows)
        row_height = (header_y - y0) / max(row_count + (1 if detail.column_titles else 0), 1)
        if detail.column_titles:
            for col_index, title in enumerate(detail.column_titles):
                center_x = x0 + label_w + (col_index + 0.5) * (value_w / col_count)
                ax.text(
                    center_x,
                    header_y - 0.5 * row_height,
                    title,
                    ha="center",
                    va="center",
                    fontsize=float(report_style["detail_label_fontsize"]),
                    fontweight="bold",
                    color="#111111",
                    **text_kwargs,
                )
            start_row_index = 1
        else:
            start_row_index = 0
        for line_index in range(start_row_index, row_count + start_row_index + 1):
            y = header_y - line_index * row_height
            ax.plot([x0, x0 + width], [y, y], transform=transform, color="#6b6b6b", lw=0.35)

        def draw_cells(
            area_x: float,
            area_y: float,
            area_width: float,
            area_height: float,
            cells: tuple[ReportDetailCellSpec, ...],
            *,
            is_label: bool,
        ) -> None:
            from matplotlib.patches import Rectangle

            cell_width = area_width / len(cells)
            for cell_index, cell in enumerate(cells):
                cell_x = area_x + cell_index * cell_width
                previous_cell = cells[cell_index - 1] if cell_index else None
                if cell.background_color:
                    ax.add_patch(
                        Rectangle(
                            (cell_x, area_y),
                            cell_width,
                            area_height,
                            facecolor=cell.background_color,
                            edgecolor="none",
                            transform=transform,
                            zorder=0.12,
                        )
                    )
                if (
                    cell_index
                    and previous_cell is not None
                    and previous_cell.divider_right_visible
                    and cell.divider_left_visible
                ):
                    ax.plot(
                        [cell_x, cell_x],
                        [area_y, area_y + area_height],
                        transform=transform,
                        color="#6b6b6b",
                        lw=0.35,
                    )
                text = self._resolve_report_value(cell.value, dataset)
                if not text:
                    continue
                text_kwargs_local: dict[str, Any] = dict(text_kwargs)
                if cell.font_weight is not None:
                    text_kwargs_local["fontweight"] = cell.font_weight
                ax.text(
                    cell_x + (0.008 if is_label else 0.5 * cell_width),
                    area_y + 0.5 * area_height,
                    text,
                    ha="left" if is_label else "center",
                    va="center",
                    fontsize=(
                        float(report_style["detail_label_fontsize"])
                        if is_label
                        else float(report_style["detail_value_fontsize"])
                    ),
                    color=cell.text_color or "#111111",
                    **text_kwargs_local,
                )

        for row_index, row in enumerate(detail.rows):
            row_top = header_y - (row_index + start_row_index) * row_height
            row_bottom = row_top - row_height
            draw_cells(
                x0,
                row_bottom,
                label_w,
                row_height,
                row.label_cells,
                is_label=True,
            )
            for col_index, column in enumerate(row.columns):
                column_x = x0 + label_w + col_index * (value_w / col_count)
                draw_cells(
                    column_x,
                    row_bottom,
                    value_w / col_count,
                    row_height,
                    column.cells,
                    is_label=False,
                )

    def _draw_report_page(
        self,
        fig,
        report: ReportBlockSpec,
        dataset: WellDataset,
        *,
        compact: bool,
        frame: tuple[float, float, float, float] | None = None,
    ) -> None:
        ax = fig.add_axes([0, 0, 1, 1] if frame is None else list(frame))
        ax.set_axis_off()
        report_style = self._style_section("report")
        rotated = not compact
        transform = self._report_page_transform(ax, rotated=rotated)
        text_rotation = -90.0 if rotated else 0.0
        self._draw_report_outer_frame(ax, report_style)
        if compact:
            self._draw_report_tail_section(
                ax,
                report,
                dataset,
                report_style,
                transform=transform,
                text_rotation=text_rotation,
            )
            return
        self._draw_report_cover_section(
            ax,
            report,
            dataset,
            report_style,
            transform=transform,
            text_rotation=text_rotation,
        )
        if report.detail is not None:
            self._draw_report_detail_table(
                ax,
                report.detail,
                dataset,
                report_style,
                transform=transform,
                text_rotation=text_rotation,
            )

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
        if self._is_reference_track(track):
            return None
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

    def _header_char_budget(
        self,
        ax,
        *,
        available_width_ratio: float,
        font_size_pt: float,
        char_width_ratio: float,
        min_chars: int,
    ) -> int:
        available_px = max(ax.bbox.width * available_width_ratio, 1.0)
        dpi = float(getattr(ax.figure, "dpi", 72.0) or 72.0)
        approx_char_px = max(font_size_pt * (dpi / 72.0) * char_width_ratio, 1.0)
        return max(min_chars, int(available_px / approx_char_px))

    def _text_line_budget(
        self,
        ax,
        *,
        available_height_ratio: float,
        font_size_pt: float,
        min_lines: int = 1,
    ) -> int:
        available_px = max(ax.bbox.height * available_height_ratio, 1.0)
        dpi = float(getattr(ax.figure, "dpi", 72.0) or 72.0)
        approx_line_px = max(font_size_pt * (dpi / 72.0) * 1.25, 1.0)
        return max(min_lines, int(available_px / approx_line_px))

    def _wrap_box_text(
        self,
        ax,
        *,
        text: str,
        available_width_ratio: float,
        available_height_ratio: float,
        font_size_pt: float,
        wrap_enabled: bool,
    ) -> str:
        if not text:
            return ""
        if not wrap_enabled:
            return text

        max_chars = self._header_char_budget(
            ax,
            available_width_ratio=max(available_width_ratio, 0.01),
            font_size_pt=font_size_pt,
            char_width_ratio=0.62,
            min_chars=1,
        )
        max_lines = self._text_line_budget(
            ax,
            available_height_ratio=max(available_height_ratio, 0.01),
            font_size_pt=font_size_pt,
            min_lines=1,
        )
        wrapper = textwrap.TextWrapper(
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        force_wrapper = textwrap.TextWrapper(
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines: list[str] = []
        paragraphs = text.splitlines() or [text]
        truncated = False
        for paragraph_index, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                wrapped_lines = wrapper.wrap(paragraph)
                if not wrapped_lines:
                    wrapped_lines = force_wrapper.wrap(paragraph)
            else:
                wrapped_lines = [""]
            for line in wrapped_lines:
                if len(lines) >= max_lines:
                    truncated = True
                    break
                lines.append(line)
            if truncated:
                break
            if paragraph_index != len(paragraphs) - 1:
                if len(lines) >= max_lines:
                    truncated = True
                    break
                lines.append("")
        if not lines:
            return text
        if truncated:
            last_index = min(len(lines), max_lines) - 1
            last_line = lines[last_index].rstrip()
            if max_chars <= 3:
                lines[last_index] = last_line[:max_chars]
            else:
                if len(last_line) > max_chars - 3:
                    last_line = last_line[: max_chars - 3].rstrip()
                if not last_line.endswith("..."):
                    last_line = f"{last_line}..."
                lines[last_index] = last_line
        return "\n".join(lines[:max_lines])

    def _wrap_annotation_label_text(
        self,
        ax,
        *,
        text: str,
        available_width_ratio: float,
        font_size_pt: float,
        max_lines: int = 2,
    ) -> str:
        if not text:
            return ""
        max_chars = self._header_char_budget(
            ax,
            available_width_ratio=max(available_width_ratio, 0.01),
            font_size_pt=font_size_pt,
            char_width_ratio=0.62,
            min_chars=1,
        )
        wrapper = textwrap.TextWrapper(
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        force_wrapper = textwrap.TextWrapper(
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines = wrapper.wrap(text) or force_wrapper.wrap(text)
        if not lines:
            return text
        if len(lines) <= max_lines:
            return "\n".join(lines)
        kept = lines[:max_lines]
        last_line = kept[-1].rstrip()
        if max_chars <= 3:
            kept[-1] = last_line[:max_chars]
        else:
            if len(last_line) > max_chars - 3:
                last_line = last_line[: max_chars - 3].rstrip()
            if not last_line.endswith("..."):
                last_line = f"{last_line}..."
            kept[-1] = last_line
        return "\n".join(kept)

    def _format_curve_header_label(
        self,
        element: CurveElement,
        *,
        label: str,
        max_chars: int,
    ) -> str:
        if not label:
            return ""
        if max_chars <= 0:
            return ""
        if not element.header_display.wrap_name:
            if len(label) > max_chars:
                if max_chars <= 3:
                    return label[:max_chars]
                return f"{label[: max_chars - 3]}..."
            return label

        wrapped = textwrap.wrap(
            label,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
            max_lines=2,
            placeholder="...",
        )
        if not wrapped or any(len(line) > max_chars for line in wrapped):
            wrapped = textwrap.wrap(
                label,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False,
                max_lines=2,
                placeholder="...",
            )
        return "\n".join(wrapped) if wrapped else label

    def _curve_header_line_style(self, element: CurveElement) -> str:
        if not element.header_display.show_color:
            return "-"
        return element.style.line_style

    def _curve_header_line_width(self, element: CurveElement) -> float:
        if not element.header_display.show_color:
            return 0.7
        return max(0.7, element.style.line_width)

    def _curve_fill_header_label(self, element: CurveElement) -> str:
        assert element.fill is not None
        if element.fill.label is not None:
            return element.fill.label
        if (
            element.fill.kind == CurveFillKind.BETWEEN_CURVES
            and element.fill.other_channel is not None
        ):
            return f"{element.channel} / {element.fill.other_channel}"
        if (
            element.fill.kind == CurveFillKind.BETWEEN_INSTANCES
            and element.fill.other_element_id is not None
        ):
            target = element.fill.other_element_id
            source = element.id or element.channel
            return f"{source} / {target}"
        if element.fill.kind == CurveFillKind.TO_LOWER_LIMIT:
            return "Lower Limit Fill"
        if element.fill.kind == CurveFillKind.TO_UPPER_LIMIT:
            return "Upper Limit Fill"
        if element.fill.kind == CurveFillKind.BASELINE_SPLIT:
            return "Baseline Fill"
        return "Fill"

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
        if self._is_reference_track(track) and curves:
            self._draw_track_header_curve_pairs(
                ax,
                track,
                document,
                dataset,
                slot_top,
                slot_bottom,
            )
            return
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
                    max_chars = self._header_char_budget(
                        ax,
                        available_width_ratio=0.9,
                        font_size_pt=fontsize,
                        char_width_ratio=float(track_header_style["legend_char_width_ratio"]),
                        min_chars=int(track_header_style["legend_min_chars"]),
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
            max_chars = self._header_char_budget(
                ax,
                available_width_ratio=0.9,
                font_size_pt=fontsize,
                char_width_ratio=float(track_header_style["legend_char_width_ratio"]),
                min_chars=int(track_header_style["legend_min_chars"]),
            )
            label = self._format_curve_header_label(
                element,
                label=label,
                max_chars=max_chars,
            )

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
                linespacing=0.9,
                multialignment="center",
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
        fill_row_count = self._fill_header_row_count(document, track)
        rows = self._curve_row_bounds(slot_top, slot_bottom, row_count * 2 + fill_row_count)
        curve_rows = rows[: row_count * 2]
        fill_rows = rows[row_count * 2 :]
        for curve_index, element in enumerate(curves):
            name_top, name_bottom = curve_rows[curve_index * 2]
            scale_top, scale_bottom = curve_rows[curve_index * 2 + 1]
            row_color = self._curve_header_color(element)

            name_fontsize = self._slot_font_size(
                ax,
                name_top,
                name_bottom,
                min_pt=float(track_header_style["legend_row_min_pt"]),
                max_pt=float(track_header_style["legend_row_max_pt"]),
            )
            label = self._curve_header_label(element)
            max_chars = self._header_char_budget(
                ax,
                available_width_ratio=0.9,
                font_size_pt=name_fontsize,
                char_width_ratio=float(track_header_style["legend_char_width_ratio"]),
                min_chars=int(track_header_style["legend_min_chars"]),
            )
            label = self._format_curve_header_label(
                element,
                label=label,
                max_chars=max_chars,
            )
            ax.text(
                0.5,
                0.5 * (name_top + name_bottom),
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=name_fontsize,
                color=row_color,
                linespacing=0.9,
                multialignment="center",
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

        if not fill_rows:
            return

        from matplotlib.patches import Rectangle

        separator_color = str(track_header_style["separator_color"])
        separator_linewidth = float(track_header_style["separator_linewidth"])
        fill_hatch = str(track_header_style.get("fill_hatch", ""))
        fill_row_font_min = float(track_header_style.get("fill_row_min_pt", 3.3))
        fill_row_font_max = float(track_header_style.get("fill_row_max_pt", 5.1))
        fill_text_color = str(track_header_style.get("fill_text_color", "#222222"))
        label_box_facecolor = str(track_header_style.get("fill_label_box_facecolor", "#ffffff"))
        label_box_edgecolor = str(track_header_style.get("fill_label_box_edgecolor", "#666666"))
        label_box_linewidth = float(track_header_style.get("fill_label_box_linewidth", 0.35))
        label_box_pad = float(track_header_style.get("fill_label_box_pad", 0.16))
        fill_elements = self._fill_header_elements(track)
        independent_curve_scales = self._uses_independent_curve_scales(track)
        for element, (row_top, row_bottom) in zip(fill_elements, fill_rows, strict=False):
            assert element.fill is not None
            segments = self._curve_fill_header_segments(
                track,
                element,
                document,
                dataset,
                independent_curve_scales=independent_curve_scales,
            )
            for left_fraction, right_fraction, fill_color, fill_alpha in segments:
                ax.add_patch(
                    Rectangle(
                        (left_fraction, row_bottom),
                        max(0.0, right_fraction - left_fraction),
                        row_top - row_bottom,
                        transform=ax.transAxes,
                        facecolor=fill_color,
                        edgecolor="none",
                        hatch=fill_hatch,
                        alpha=fill_alpha,
                        zorder=0.2,
                    )
                )

            marker_x = self._curve_fill_header_marker_x(
                track,
                element,
                dataset,
            )
            if marker_x is not None:
                line_color, line_width, line_style = self._resolved_curve_fill_baseline_line_style(
                    element
                )
                ax.plot(
                    [marker_x, marker_x],
                    [row_bottom, row_top],
                    transform=ax.transAxes,
                    color=line_color,
                    linewidth=line_width,
                    linestyle=line_style,
                    zorder=0.35,
                )

            ax.plot(
                [0.0, 1.0],
                [row_top, row_top],
                transform=ax.transAxes,
                color=separator_color,
                linewidth=separator_linewidth,
            )
            ax.plot(
                [0.0, 1.0],
                [row_bottom, row_bottom],
                transform=ax.transAxes,
                color=separator_color,
                linewidth=separator_linewidth,
            )

            fontsize = self._slot_font_size(
                ax,
                row_top,
                row_bottom,
                min_pt=fill_row_font_min,
                max_pt=fill_row_font_max,
            )
            label = self._curve_fill_header_label(element)
            max_chars = self._header_char_budget(
                ax,
                available_width_ratio=0.76,
                font_size_pt=fontsize,
                char_width_ratio=float(track_header_style["legend_char_width_ratio"]),
                min_chars=int(track_header_style["legend_min_chars"]),
            )
            if len(label) > max_chars:
                label = f"{label[: max_chars - 3]}..."
            ax.text(
                0.5,
                0.5 * (row_top + row_bottom),
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=fill_text_color,
                bbox={
                    "boxstyle": f"square,pad={label_box_pad}",
                    "facecolor": label_box_facecolor,
                    "edgecolor": label_box_edgecolor,
                    "linewidth": label_box_linewidth,
                },
                clip_on=True,
                zorder=0.5,
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
                ax.set_xlim(0.0, 1.0)
                ax.set_xticks([])
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
            self._draw_reference_events(ax, track, window)
            self._draw_curve_callouts(
                ax,
                track,
                document,
                dataset,
                window,
                independent_curve_scales=False,
            )
            self._draw_reference_event_callouts(ax, track, document, window)
            self._draw_marker_callouts(ax, document, window)
            return

        if self._is_annotation_track(track):
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.tick_params(axis="y", length=0, labelleft=False)
            self._draw_annotation_objects(ax, track, window)
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
        if self._should_draw_array_plot_sample_axis(page_layout, track):
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
        self._draw_curve_callouts(
            ax,
            track,
            document,
            dataset,
            window,
            independent_curve_scales=independent_curve_scales,
        )

    def _uses_independent_curve_scales(self, track) -> bool:
        if self._is_reference_track(track) or self._is_annotation_track(track):
            return False
        return self._curve_count(track) > 1

    def _track_has_bottom_header_on_page(self, page_layout, track) -> bool:
        return any(frame.track is track for frame in page_layout.track_header_bottom_frames)

    def _should_draw_array_plot_sample_axis(self, page_layout, track) -> bool:
        if track.kind != TrackKind.ARRAY:
            return False
        return not self._track_has_bottom_header_on_page(page_layout, track)

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

        reference_overlay = self._resolved_reference_overlay(track, element)
        if reference_overlay is not None:
            if reference_overlay.mode in {
                ReferenceCurveOverlayMode.CURVE,
                ReferenceCurveOverlayMode.INDICATOR,
            }:
                lane_start, lane_end = self._reference_overlay_lane(reference_overlay)
                normalized = np.full(plot_values.shape, np.nan, dtype=float)
                if scale is not None:
                    normalized, normalized_mask = self._normalize_curve_values(plot_values, scale)
                    valid_mask &= normalized_mask
                else:
                    finite_mask = np.isfinite(plot_values)
                    finite = plot_values[finite_mask]
                    if finite.size >= 2 and not np.isclose(
                        float(np.nanmin(finite)),
                        float(np.nanmax(finite)),
                    ):
                        normalized[finite_mask] = (
                            plot_values[finite_mask] - float(np.nanmin(finite))
                        ) / (float(np.nanmax(finite)) - float(np.nanmin(finite)))
                        valid_mask &= finite_mask
                    elif finite.size >= 1:
                        normalized[finite_mask] = 0.5
                        valid_mask &= finite_mask
                    else:
                        valid_mask &= finite_mask
                plot_values = lane_start + normalized * (lane_end - lane_start)
                wrapped_mask &= valid_mask
                return _CurvePlotData(
                    depth,
                    values,
                    plot_values,
                    valid_mask,
                    wrapped_mask,
                    scale,
                    x_is_fractional=True,
                )
            if reference_overlay.mode == ReferenceCurveOverlayMode.TICKS:
                tick_side = reference_overlay.tick_side
                if tick_side == ReferenceCurveTickSide.LEFT:
                    anchor_x = 0.0
                elif tick_side == ReferenceCurveTickSide.RIGHT:
                    anchor_x = 1.0
                else:
                    anchor_x = 0.5
                threshold = self._reference_overlay_threshold(reference_overlay)
                active_mask = valid_mask & (np.abs(values) > threshold)
                plot_values = np.full(values.shape, anchor_x, dtype=float)
                wrapped_mask &= active_mask
                return _CurvePlotData(
                    depth,
                    values,
                    plot_values,
                    active_mask,
                    wrapped_mask,
                    scale,
                    x_is_fractional=True,
                )

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

    def _resolved_curve_fill_baseline_colors(self, element: CurveElement) -> tuple[str, str]:
        assert element.fill is not None
        assert element.fill.baseline is not None
        fallback = self._resolved_curve_fill_color(element)
        lower_color = element.fill.baseline.lower_color or fallback
        upper_color = element.fill.baseline.upper_color or fallback
        return lower_color, upper_color

    def _resolved_curve_fill_baseline_line_style(
        self,
        element: CurveElement,
    ) -> tuple[str, float, str]:
        assert element.fill is not None
        assert element.fill.baseline is not None
        line_color = element.fill.baseline.line_color or element.style.color
        return (
            line_color,
            element.fill.baseline.line_width,
            element.fill.baseline.line_style,
        )

    def _curve_fill_display_bounds(
        self,
        track,
        element: CurveElement,
        dataset: WellDataset,
    ) -> tuple[float, float]:
        scale = element.scale or track.x_scale
        if scale is not None:
            return float(scale.minimum), float(scale.maximum)
        channel = dataset.get_channel(element.channel)
        if isinstance(channel, ScalarChannel):
            finite = channel.masked_values()
            finite = finite[np.isfinite(finite)]
            if finite.size >= 1:
                return float(np.nanmin(finite)), float(np.nanmax(finite))
        return 0.0, 1.0

    def _curve_fill_plot_coordinate(
        self,
        raw_value: float,
        scale,
        *,
        independent_curve_scales: bool,
    ) -> float:
        if scale is None:
            return float(raw_value)
        if independent_curve_scales or scale.kind == ScaleKind.TANGENTIAL:
            normalized, mask = self._normalize_curve_values(
                np.asarray([raw_value], dtype=float),
                scale,
            )
            if not np.any(mask):
                return float("nan")
            return float(normalized[0])
        return float(raw_value)

    def _resolve_curve_fill_target(
        self,
        track,
        element: CurveElement,
        *,
        independent_curve_scales: bool,
    ) -> CurveElement:
        assert element.fill is not None
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
            return compatible[0]
        if element.fill.kind == CurveFillKind.BETWEEN_INSTANCES:
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
            return other
        raise TemplateValidationError(
            f"Curve fill kind {element.fill.kind!s} is not implemented yet."
        )

    def _prepare_curve_fill_data(
        self,
        track,
        element: CurveElement,
        document,
        dataset,
        *,
        independent_curve_scales: bool,
    ) -> _CurveFillRenderData:
        assert element.fill is not None
        if element.wrap:
            raise TemplateValidationError(
                f"Curve fill for {element.channel!r} does not support wrapped curves yet."
            )

        primary_data = self._curve_plot_data(
            track,
            element,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )

        if element.fill.kind in {
            CurveFillKind.BETWEEN_CURVES,
            CurveFillKind.BETWEEN_INSTANCES,
        }:
            other = self._resolve_curve_fill_target(
                track,
                element,
                independent_curve_scales=independent_curve_scales,
            )
            if other.wrap:
                raise TemplateValidationError(
                    f"Curve fill for {element.channel!r} cannot target wrapped curve "
                    f"{other.channel!r} yet."
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
        elif element.fill.kind in {CurveFillKind.TO_LOWER_LIMIT, CurveFillKind.TO_UPPER_LIMIT}:
            lower_bound, upper_bound = self._curve_fill_display_bounds(track, element, dataset)
            target_raw = (
                lower_bound
                if element.fill.kind == CurveFillKind.TO_LOWER_LIMIT
                else upper_bound
            )
            target_plot = self._curve_fill_plot_coordinate(
                target_raw,
                primary_data.scale,
                independent_curve_scales=independent_curve_scales,
            )
            secondary_values = np.full(primary_data.plot_values.shape, target_plot, dtype=float)
        elif element.fill.kind == CurveFillKind.BASELINE_SPLIT:
            assert element.fill.baseline is not None
            target_plot = self._curve_fill_plot_coordinate(
                element.fill.baseline.value,
                primary_data.scale,
                independent_curve_scales=independent_curve_scales,
            )
            secondary_values = np.full(primary_data.plot_values.shape, target_plot, dtype=float)
        else:
            raise TemplateValidationError(
                f"Curve fill kind {element.fill.kind!s} is not implemented yet."
            )

        valid_mask = (
            primary_data.valid_mask
            & np.isfinite(primary_data.plot_values)
            & np.isfinite(secondary_values)
        )
        return _CurveFillRenderData(primary_data, secondary_values, valid_mask)

    def _curve_fill_header_segments_from_masks(
        self,
        primary_count: int,
        secondary_count: int,
        *,
        primary_color: str,
        secondary_color: str,
        alpha: float,
        fallback_color: str,
        fallback_alpha: float,
    ) -> list[tuple[float, float, str, float]]:
        if primary_count > 0 and secondary_count > 0:
            total = primary_count + secondary_count
            primary_fraction = primary_count / total
            return [
                (0.0, primary_fraction, primary_color, alpha),
                (primary_fraction, 1.0, secondary_color, alpha),
            ]
        if primary_count > 0:
            return [(0.0, 1.0, primary_color, alpha)]
        if secondary_count > 0:
            return [(0.0, 1.0, secondary_color, alpha)]
        return [(0.0, 1.0, fallback_color, fallback_alpha)]

    def _curve_fill_header_marker_x(
        self,
        track,
        element: CurveElement,
        dataset,
    ) -> float | None:
        if element.fill is None or element.fill.kind != CurveFillKind.BASELINE_SPLIT:
            return None
        assert element.fill.baseline is not None
        scale = element.scale or track.x_scale
        if scale is not None:
            marker_x = self._curve_fill_plot_coordinate(
                element.fill.baseline.value,
                scale,
                independent_curve_scales=True,
            )
            if np.isfinite(marker_x):
                return float(marker_x)
            return None
        lower_bound, upper_bound = self._curve_fill_display_bounds(track, element, dataset)
        if np.isclose(lower_bound, upper_bound):
            return 0.5
        fraction = (element.fill.baseline.value - lower_bound) / (upper_bound - lower_bound)
        return float(np.clip(fraction, 0.0, 1.0))

    def _curve_fill_header_segments(
        self,
        track,
        element: CurveElement,
        document,
        dataset,
        *,
        independent_curve_scales: bool,
    ) -> list[tuple[float, float, str, float]]:
        assert element.fill is not None
        fill_data = self._prepare_curve_fill_data(
            track,
            element,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )
        primary_data = fill_data.primary
        secondary_values = fill_data.secondary_values
        valid_mask = fill_data.valid_mask
        if not np.any(valid_mask):
            return []

        fill_color = self._resolved_curve_fill_color(element)
        fill_alpha = self._resolved_curve_fill_alpha(element)
        if element.fill.kind in {CurveFillKind.TO_LOWER_LIMIT, CurveFillKind.TO_UPPER_LIMIT}:
            return [(0.0, 1.0, fill_color, fill_alpha)]

        if element.fill.kind == CurveFillKind.BASELINE_SPLIT:
            assert element.fill.baseline is not None
            lower_color, upper_color = self._resolved_curve_fill_baseline_colors(element)
            marker_x = self._curve_fill_header_marker_x(track, element, dataset)
            if marker_x is None:
                marker_x = 0.5
            marker_x = float(np.clip(marker_x, 0.0, 1.0))
            scale = element.scale or track.x_scale
            reverse = bool(scale.reverse) if scale is not None else False
            left_color = upper_color if reverse else lower_color
            right_color = lower_color if reverse else upper_color
            segments: list[tuple[float, float, str, float]] = []
            if marker_x > 0.0:
                segments.append((0.0, marker_x, left_color, fill_alpha))
            if marker_x < 1.0:
                segments.append((marker_x, 1.0, right_color, fill_alpha))
            if segments:
                return segments
            return [(0.0, 1.0, fill_color, fill_alpha)]

        if not element.fill.crossover.enabled:
            return [(0.0, 1.0, fill_color, fill_alpha)]

        left_mask = valid_mask & (primary_data.plot_values < secondary_values)
        right_mask = valid_mask & (primary_data.plot_values > secondary_values)
        left_color = element.fill.crossover.left_color or fill_color
        right_color = element.fill.crossover.right_color or fill_color
        crossover_alpha = (
            element.fill.crossover.alpha
            if element.fill.crossover.alpha is not None
            else fill_alpha
        )
        return self._curve_fill_header_segments_from_masks(
            int(np.count_nonzero(left_mask)),
            int(np.count_nonzero(right_mask)),
            primary_color=left_color,
            secondary_color=right_color,
            alpha=crossover_alpha,
            fallback_color=fill_color,
            fallback_alpha=fill_alpha,
        )

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
        if self._is_reference_track(track):
            raise TemplateValidationError(
                f"Curve fills are not supported in reference track overlays ({track.id!r}) yet."
            )
        fill_data = self._prepare_curve_fill_data(
            track,
            element,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )
        primary_data = fill_data.primary
        secondary_values = fill_data.secondary_values
        valid_mask = fill_data.valid_mask
        if not np.any(valid_mask):
            return

        alpha = self._resolved_curve_fill_alpha(element)
        if element.fill.kind == CurveFillKind.BASELINE_SPLIT:
            assert element.fill.baseline is not None
            lower_color, upper_color = self._resolved_curve_fill_baseline_colors(element)
            lower_mask = valid_mask & (primary_data.raw_values < element.fill.baseline.value)
            upper_mask = valid_mask & (primary_data.raw_values > element.fill.baseline.value)
            if np.any(lower_mask):
                ax.fill_betweenx(
                    primary_data.depth,
                    primary_data.plot_values,
                    secondary_values,
                    where=lower_mask,
                    facecolor=lower_color,
                    alpha=alpha,
                    linewidth=0.0,
                    interpolate=True,
                )
            if np.any(upper_mask):
                ax.fill_betweenx(
                    primary_data.depth,
                    primary_data.plot_values,
                    secondary_values,
                    where=upper_mask,
                    facecolor=upper_color,
                    alpha=alpha,
                    linewidth=0.0,
                    interpolate=True,
                )
            baseline_values = secondary_values[valid_mask]
            if baseline_values.size > 0:
                line_color, line_width, line_style = self._resolved_curve_fill_baseline_line_style(
                    element
                )
                ax.axvline(
                    float(baseline_values[0]),
                    color=line_color,
                    linewidth=line_width,
                    linestyle=line_style,
                    zorder=0.35,
                )
            return

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

    def _draw_reference_curve_ticks(
        self,
        ax,
        *,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        overlay: ReferenceCurveOverlaySpec,
        element: CurveElement,
    ) -> None:
        if not np.any(valid_mask):
            return
        tick_length = self._reference_overlay_tick_length_ratio(overlay)
        if overlay.tick_side in {ReferenceCurveTickSide.LEFT, ReferenceCurveTickSide.BOTH}:
            ax.hlines(
                depth[valid_mask],
                0.0,
                tick_length,
                colors=element.style.color,
                linewidth=element.style.line_width,
                linestyles=element.style.line_style,
                alpha=element.style.opacity,
            )
        if overlay.tick_side in {ReferenceCurveTickSide.RIGHT, ReferenceCurveTickSide.BOTH}:
            ax.hlines(
                depth[valid_mask],
                1.0 - tick_length,
                1.0,
                colors=element.style.color,
                linewidth=element.style.line_width,
                linestyles=element.style.line_style,
                alpha=element.style.opacity,
            )

    def _draw_reference_events(self, ax, track, window) -> None:
        from matplotlib.transforms import blended_transform_factory

        events = self._reference_event_elements(track)
        if not events:
            return
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        for event in events:
            if event.depth < window.start or event.depth > window.stop:
                continue
            for left_fraction, right_fraction in self._reference_event_segments(event):
                ax.plot(
                    [left_fraction, right_fraction],
                    [event.depth, event.depth],
                    transform=transform,
                    color=event.color,
                    linewidth=event.line_width,
                    linestyle=event.line_style,
                    clip_on=True,
                    zorder=3.6,
                )

    def _draw_reference_event_callouts(self, ax, track, document: LogDocument, window) -> None:
        from matplotlib.transforms import blended_transform_factory

        events = self._reference_event_elements(track)
        if not events:
            return
        callout_style = self._style_section("curve_callouts")
        text_transform = blended_transform_factory(ax.transAxes, ax.transData)
        default_offset = self._curve_callout_depth_step(document) * float(
            callout_style["default_depth_offset_steps"]
        )
        for event in events:
            if event.depth < window.start or event.depth > window.stop or not event.label:
                continue
            segments = self._reference_event_segments(event)
            segment_left = min(left for left, _ in segments)
            segment_right = max(right for _, right in segments)
            segment_center = 0.5 * (segment_left + segment_right)
            text_side = event.text_side
            if text_side == "auto":
                text_side = "right" if segment_center <= 0.5 else "left"
            anchor_x = segment_right if text_side == "right" else segment_left
            text_x = (
                event.text_x
                if event.text_x is not None
                else float(callout_style["right_text_x"])
                if text_side == "right"
                else float(callout_style["left_text_x"])
            )
            text_y = event.depth + (
                event.depth_offset if event.depth_offset is not None else default_offset
            )
            ax.annotate(
                event.label,
                xy=(anchor_x, event.depth),
                xycoords=text_transform,
                xytext=(float(text_x), float(text_y)),
                textcoords=text_transform,
                fontsize=(
                    event.font_size
                    if event.font_size is not None
                    else float(callout_style["font_size"])
                ),
                color=event.color,
                fontweight=event.font_weight,
                fontstyle=event.font_style,
                ha=self._curve_callout_horizontal_alignment(text_side),
                va="center",
                zorder=4.2,
                arrowprops=(
                    {
                        "arrowstyle": event.arrow_style or str(callout_style["arrow_style"]),
                        "color": event.color,
                        "lw": (
                            event.arrow_linewidth
                            if event.arrow_linewidth is not None
                            else float(callout_style["arrow_linewidth"])
                        ),
                        "shrinkA": 0,
                        "shrinkB": 0,
                        "relpos": self._curve_callout_arrow_relpos(text_side),
                    }
                    if event.arrow
                    else None
                ),
                annotation_clip=True,
            )

    def _annotation_visible_interval(
        self,
        top: float,
        base: float,
        window,
    ) -> tuple[float, float] | None:
        if base < window.start or top > window.stop:
            return None
        return max(float(top), float(window.start)), min(float(base), float(window.stop))

    def _annotation_box_anchor(
        self,
        *,
        lane_start: float,
        lane_end: float,
        top: float,
        base: float,
        padding: float,
        horizontal_alignment: str,
        vertical_alignment: str,
    ) -> tuple[float, float]:
        width = lane_end - lane_start
        height = base - top
        x_padding = width * padding
        y_padding = height * padding
        if horizontal_alignment == "left":
            x = lane_start + x_padding
        elif horizontal_alignment == "right":
            x = lane_end - x_padding
        else:
            x = 0.5 * (lane_start + lane_end)
        if vertical_alignment == "top":
            y = top + y_padding
        elif vertical_alignment == "bottom":
            y = base - y_padding
        else:
            y = 0.5 * (top + base)
        return x, y

    def _annotation_marker_symbol(self, shape: str) -> str:
        return {
            "circle": "o",
            "square": "s",
            "diamond": "D",
            "triangle_up": "^",
            "triangle_down": "v",
            "triangle_left": "<",
            "triangle_right": ">",
            "x": "x",
            "plus": "+",
            "bar_horizontal": "_",
            "bar_vertical": "|",
        }[shape]

    def _draw_annotation_interval(self, ax, annotation: AnnotationIntervalSpec, window) -> None:
        visible = self._annotation_visible_interval(annotation.top, annotation.base, window)
        if visible is None:
            return
        top, base = visible
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (annotation.lane_start, top),
            annotation.lane_end - annotation.lane_start,
            base - top,
            facecolor=annotation.fill_color,
            alpha=annotation.fill_alpha,
            edgecolor=annotation.border_color,
            linewidth=annotation.border_linewidth,
            linestyle=annotation.border_style,
            zorder=1.2,
            clip_on=True,
        )
        ax.add_patch(rect)
        if not annotation.text:
            return
        x, y = self._annotation_box_anchor(
            lane_start=annotation.lane_start,
            lane_end=annotation.lane_end,
            top=top,
            base=base,
            padding=annotation.padding,
            horizontal_alignment=annotation.horizontal_alignment,
            vertical_alignment=annotation.vertical_alignment,
        )
        width_ratio = max(
            annotation.lane_end - annotation.lane_start - 2 * annotation.padding,
            0.01,
        )
        window_span = max(float(window.stop - window.start), 1e-9)
        height_ratio = max((base - top) / window_span - 2 * annotation.padding, 0.01)
        text_value = annotation.text
        if annotation.text_orientation == "horizontal":
            text_value = self._wrap_box_text(
                ax,
                text=text_value,
                available_width_ratio=width_ratio,
                available_height_ratio=height_ratio,
                font_size_pt=annotation.font_size,
                wrap_enabled=annotation.text_wrap,
            )
        ax.text(
            x,
            y,
            text_value,
            transform=ax.transData,
            ha=annotation.horizontal_alignment,
            va=annotation.vertical_alignment,
            fontsize=annotation.font_size,
            fontweight=annotation.font_weight,
            fontstyle=annotation.font_style,
            color=annotation.text_color,
            rotation=90 if annotation.text_orientation == "vertical" else 0,
            rotation_mode="anchor",
            multialignment=annotation.horizontal_alignment,
            linespacing=0.92,
            clip_on=True,
            zorder=1.6,
        )

    def _draw_annotation_text(self, ax, annotation: AnnotationTextSpec, window) -> None:
        from matplotlib.patches import Rectangle

        if annotation.depth is not None:
            if annotation.depth < window.start or annotation.depth > window.stop:
                return
            x, y = self._annotation_box_anchor(
                lane_start=annotation.lane_start,
                lane_end=annotation.lane_end,
                top=annotation.depth,
                base=annotation.depth,
                padding=annotation.padding,
                horizontal_alignment=annotation.horizontal_alignment,
                vertical_alignment="center",
            )
            width_ratio = max(
                annotation.lane_end - annotation.lane_start - 2 * annotation.padding,
                0.01,
            )
            height_ratio = 0.08
            bbox = (
                {
                    "facecolor": annotation.background_color or "none",
                    "edgecolor": annotation.border_color or "none",
                    "linewidth": annotation.border_linewidth or 0.6,
                    "boxstyle": f"square,pad={annotation.padding}",
                }
                if annotation.background_color is not None or annotation.border_color is not None
                else None
            )
            vertical_alignment = "center"
        else:
            assert annotation.top is not None and annotation.base is not None
            visible = self._annotation_visible_interval(annotation.top, annotation.base, window)
            if visible is None:
                return
            top, base = visible
            if annotation.background_color is not None or annotation.border_color is not None:
                rect = Rectangle(
                    (annotation.lane_start, top),
                    annotation.lane_end - annotation.lane_start,
                    base - top,
                    facecolor=annotation.background_color or "none",
                    edgecolor=annotation.border_color or "none",
                    linewidth=annotation.border_linewidth or 0.6,
                    linestyle="-",
                    zorder=1.25,
                    clip_on=True,
                )
                ax.add_patch(rect)
            x, y = self._annotation_box_anchor(
                lane_start=annotation.lane_start,
                lane_end=annotation.lane_end,
                top=top,
                base=base,
                padding=annotation.padding,
                horizontal_alignment=annotation.horizontal_alignment,
                vertical_alignment=annotation.vertical_alignment,
            )
            width_ratio = max(
                annotation.lane_end - annotation.lane_start - 2 * annotation.padding,
                0.01,
            )
            window_span = max(float(window.stop - window.start), 1e-9)
            height_ratio = max((base - top) / window_span - 2 * annotation.padding, 0.01)
            bbox = None
            vertical_alignment = annotation.vertical_alignment

        text_value = annotation.text
        if annotation.text_orientation == "horizontal":
            text_value = self._wrap_box_text(
                ax,
                text=text_value,
                available_width_ratio=width_ratio,
                available_height_ratio=height_ratio,
                font_size_pt=annotation.font_size,
                wrap_enabled=annotation.wrap,
            )
        ax.text(
            x,
            y,
            text_value,
            transform=ax.transData,
            ha=annotation.horizontal_alignment,
            va=vertical_alignment,
            fontsize=annotation.font_size,
            fontweight=annotation.font_weight,
            fontstyle=annotation.font_style,
            color=annotation.color,
            rotation=90 if annotation.text_orientation == "vertical" else 0,
            rotation_mode="anchor",
            multialignment=annotation.horizontal_alignment,
            linespacing=0.92,
            bbox=bbox,
            clip_on=True,
            zorder=1.7,
        )

    def _draw_annotation_marker_shape(self, ax, annotation: AnnotationMarkerSpec, window) -> None:
        from matplotlib.transforms import blended_transform_factory

        if annotation.depth < window.start or annotation.depth > window.stop:
            return
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        symbol = self._annotation_marker_symbol(annotation.shape)
        edge_color = annotation.edge_color or annotation.color
        if annotation.shape in {"x", "plus", "bar_horizontal", "bar_vertical"}:
            ax.scatter(
                [annotation.x],
                [annotation.depth],
                transform=transform,
                marker=symbol,
                s=annotation.size,
                c=[edge_color],
                linewidths=annotation.line_width,
                zorder=1.85,
                clip_on=True,
            )
        else:
            ax.scatter(
                [annotation.x],
                [annotation.depth],
                transform=transform,
                marker=symbol,
                s=annotation.size,
                facecolors=annotation.fill_color or annotation.color,
                edgecolors=edge_color,
                linewidths=annotation.line_width,
                zorder=1.85,
                clip_on=True,
            )
    def _annotation_marker_label_record(
        self,
        annotation: AnnotationMarkerSpec,
        callout_style: dict[str, Any],
    ) -> _AnnotationLabelRecord | None:
        if not annotation.label or annotation.label_mode == AnnotationLabelMode.NONE:
            return None
        text_side = annotation.text_side
        if text_side == "auto":
            text_side = "right" if annotation.x <= 0.5 else "left"
        text_x = (
            float(annotation.text_x)
            if annotation.text_x is not None
            else float(callout_style["right_text_x"])
            if text_side == "right"
            else float(callout_style["left_text_x"])
        )
        text_y = annotation.depth + (
            float(annotation.depth_offset) if annotation.depth_offset is not None else 0.0
        )
        label_side: str | None
        preferred_x: float
        if annotation.label_mode == AnnotationLabelMode.DEDICATED_LANE:
            lane_start = float(annotation.label_lane_start)
            lane_end = float(annotation.label_lane_end)
            preferred_x = float(
                np.clip(
                    text_x,
                    lane_start,
                    lane_end,
                )
            )
            lane_center = 0.5 * (lane_start + lane_end)
            if lane_center > annotation.x:
                label_side = "right"
                if annotation.text_x is None:
                    preferred_x = lane_start
            elif lane_center < annotation.x:
                label_side = "left"
                if annotation.text_x is None:
                    preferred_x = lane_end
            else:
                label_side = None
                if annotation.text_x is None:
                    preferred_x = lane_center
        else:
            label_side = text_side
            preferred_x = float(text_x)
        return _AnnotationLabelRecord(
            label=annotation.label,
            anchor_x=float(annotation.x),
            anchor_y=float(annotation.depth),
            preferred_x=preferred_x,
            preferred_y=float(text_y),
            color=annotation.color,
            font_size=(
                annotation.font_size
                if annotation.font_size is not None
                else float(callout_style["font_size"])
            ),
            font_weight=annotation.font_weight,
            font_style=annotation.font_style,
            priority=annotation.priority,
            arrow=annotation.arrow,
            arrow_style=annotation.arrow_style or str(callout_style["arrow_style"]),
            arrow_linewidth=(
                annotation.arrow_linewidth
                if annotation.arrow_linewidth is not None
                else float(callout_style["arrow_linewidth"])
            ),
            rotation=0.0,
            label_mode=annotation.label_mode,
            label_lane_start=annotation.label_lane_start,
            label_lane_end=annotation.label_lane_end,
            side=label_side,
            allow_side_flip=(
                annotation.label_mode == AnnotationLabelMode.FREE
                and annotation.text_side == "auto"
                and annotation.text_x is None
            ),
        )

    def _draw_annotation_arrow_line(self, ax, annotation: AnnotationArrowSpec, window) -> None:
        from matplotlib.transforms import blended_transform_factory

        interval_top = min(annotation.start_depth, annotation.end_depth)
        interval_base = max(annotation.start_depth, annotation.end_depth)
        if interval_base < window.start or interval_top > window.stop:
            return
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        ax.annotate(
            "",
            xy=(annotation.end_x, annotation.end_depth),
            xycoords=transform,
            xytext=(annotation.start_x, annotation.start_depth),
            textcoords=transform,
            zorder=1.82,
            arrowprops={
                "arrowstyle": annotation.arrow_style,
                "color": annotation.color,
                "lw": annotation.line_width,
                "linestyle": annotation.line_style,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=True,
        )
    def _annotation_arrow_label_record(
        self,
        annotation: AnnotationArrowSpec,
    ) -> _AnnotationLabelRecord | None:
        if not annotation.label or annotation.label_mode == AnnotationLabelMode.NONE:
            return None
        label_y = (
            annotation.label_depth
            if annotation.label_depth is not None
            else 0.5 * (annotation.start_depth + annotation.end_depth)
        )
        label_x = (
            annotation.label_x
            if annotation.label_x is not None
            else 0.5 * (annotation.start_x + annotation.end_x)
        )
        label_side: str | None
        preferred_x: float
        if annotation.label_mode == AnnotationLabelMode.DEDICATED_LANE:
            lane_start = float(annotation.label_lane_start)
            lane_end = float(annotation.label_lane_end)
            preferred_x = float(np.clip(label_x, lane_start, lane_end))
            lane_center = 0.5 * (lane_start + lane_end)
            if lane_center > annotation.end_x:
                label_side = "right"
                if annotation.label_x is None:
                    preferred_x = lane_start
            elif lane_center < annotation.end_x:
                label_side = "left"
                if annotation.label_x is None:
                    preferred_x = lane_end
            else:
                label_side = None
                if annotation.label_x is None:
                    preferred_x = lane_center
        else:
            label_side = None
            preferred_x = float(label_x)
        return _AnnotationLabelRecord(
            label=annotation.label,
            anchor_x=float(annotation.end_x),
            anchor_y=float(annotation.end_depth),
            preferred_x=preferred_x,
            preferred_y=float(label_y),
            color=annotation.color,
            font_size=annotation.font_size,
            font_weight=annotation.font_weight,
            font_style=annotation.font_style,
            priority=annotation.priority,
            arrow=False,
            arrow_style=annotation.arrow_style,
            arrow_linewidth=annotation.line_width,
            rotation=annotation.text_rotation,
            label_mode=annotation.label_mode,
            label_lane_start=annotation.label_lane_start,
            label_lane_end=annotation.label_lane_end,
            side=label_side,
        )

    def _draw_annotation_glyph(self, ax, annotation: AnnotationGlyphSpec, window) -> None:
        from matplotlib.patches import Rectangle

        if annotation.depth is not None:
            if annotation.depth < window.start or annotation.depth > window.stop:
                return
            x, y = self._annotation_box_anchor(
                lane_start=annotation.lane_start,
                lane_end=annotation.lane_end,
                top=annotation.depth,
                base=annotation.depth,
                padding=annotation.padding,
                horizontal_alignment=annotation.horizontal_alignment,
                vertical_alignment="center",
            )
            bbox = (
                {
                    "facecolor": annotation.background_color or "none",
                    "edgecolor": annotation.border_color or "none",
                    "linewidth": annotation.border_linewidth or 0.6,
                    "boxstyle": f"square,pad={annotation.padding}",
                }
                if annotation.background_color is not None or annotation.border_color is not None
                else None
            )
            vertical_alignment = "center"
        else:
            assert annotation.top is not None and annotation.base is not None
            visible = self._annotation_visible_interval(annotation.top, annotation.base, window)
            if visible is None:
                return
            top, base = visible
            if annotation.background_color is not None or annotation.border_color is not None:
                rect = Rectangle(
                    (annotation.lane_start, top),
                    annotation.lane_end - annotation.lane_start,
                    base - top,
                    facecolor=annotation.background_color or "none",
                    edgecolor=annotation.border_color or "none",
                    linewidth=annotation.border_linewidth or 0.6,
                    linestyle="-",
                    zorder=1.25,
                    clip_on=True,
                )
                ax.add_patch(rect)
            x, y = self._annotation_box_anchor(
                lane_start=annotation.lane_start,
                lane_end=annotation.lane_end,
                top=top,
                base=base,
                padding=annotation.padding,
                horizontal_alignment=annotation.horizontal_alignment,
                vertical_alignment=annotation.vertical_alignment,
            )
            bbox = None
            vertical_alignment = annotation.vertical_alignment
        ax.text(
            x,
            y,
            annotation.glyph,
            transform=ax.transData,
            ha=annotation.horizontal_alignment,
            va=vertical_alignment,
            fontsize=annotation.font_size,
            fontweight=annotation.font_weight,
            fontstyle=annotation.font_style,
            color=annotation.color,
            rotation=annotation.rotation,
            rotation_mode="anchor",
            multialignment=annotation.horizontal_alignment,
            bbox=bbox,
            clip_on=True,
            zorder=1.75,
        )

    def _annotation_obstacle_bboxes(self, ax, renderer) -> list[Any]:
        bboxes: list[Any] = []
        for artist in list(ax.patches) + list(ax.texts):
            try:
                bbox = artist.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            bboxes.append(bbox)
        return bboxes

    def _annotation_label_candidate_x_positions(
        self,
        record: _AnnotationLabelRecord,
        callout_style: dict[str, Any],
    ) -> list[tuple[str | None, float]]:
        if (
            record.label_mode == AnnotationLabelMode.DEDICATED_LANE
            and record.label_lane_start is not None
            and record.label_lane_end is not None
        ):
            lane_count = max(int(callout_style["lane_count"]), 1)
            lane_step = float(callout_style["lane_step_x"])
            center = float(
                np.clip(
                    record.preferred_x,
                    record.label_lane_start,
                    record.label_lane_end,
                )
            )
            candidates: list[tuple[str | None, float]] = []
            offsets = [0.0]
            for index in range(1, lane_count):
                if record.side == "right":
                    offsets.append(index * lane_step)
                elif record.side == "left":
                    offsets.append(-index * lane_step)
                else:
                    offsets.extend((-index * lane_step, index * lane_step))
            for offset in offsets:
                value = float(
                    np.clip(
                        center + offset,
                        record.label_lane_start,
                        record.label_lane_end,
                    )
                )
                if any(np.isclose(value, placed_x, atol=1e-6) for _, placed_x in candidates):
                    continue
                candidates.append((record.side, value))
            return candidates
        lane_count = max(int(callout_style["lane_count"]), 1)
        lane_step = float(callout_style["lane_step_x"])
        candidates: list[tuple[str | None, float]] = []
        if record.side is not None:
            sides = [record.side]
            if record.allow_side_flip:
                sides.append("left" if record.side == "right" else "right")
            for side in sides:
                base = (
                    record.preferred_x
                    if side == record.side
                    else float(callout_style["right_text_x"])
                    if side == "right"
                    else float(callout_style["left_text_x"])
                )
                for index in range(lane_count):
                    offset = lane_step * index
                    value = base - offset if side == "right" else base + offset
                    value = float(np.clip(value, 0.0, 1.0))
                    candidate = (side, value)
                    if any(
                        placed_side == side and np.isclose(value, placed_x, atol=1e-6)
                        for placed_side, placed_x in candidates
                    ):
                        continue
                    candidates.append(candidate)
            return candidates
        offsets = [0.0]
        for index in range(1, lane_count):
            offsets.extend((-index * lane_step, index * lane_step))
        for offset in offsets:
            value = float(np.clip(record.preferred_x + offset, 0.0, 1.0))
            candidate = (None, value)
            if any(np.isclose(value, placed_x, atol=1e-6) for _, placed_x in candidates):
                continue
            candidates.append(candidate)
        return candidates

    def _annotation_label_candidate_y_positions(
        self,
        record: _AnnotationLabelRecord,
        *,
        lower: float,
        upper: float,
        min_gap: float,
    ) -> list[float]:
        offsets = [0.0]
        search_steps = 6 if record.label_mode == AnnotationLabelMode.DEDICATED_LANE else 3
        for index in range(1, search_steps + 1):
            offsets.extend((-index * min_gap, index * min_gap))
        candidates: list[float] = []
        for offset in offsets:
            value = float(np.clip(record.preferred_y + offset, lower, upper))
            if any(np.isclose(value, current, atol=1e-6) for current in candidates):
                continue
            candidates.append(value)
        return candidates

    def _annotation_label_horizontal_alignment(self, side: str | None) -> str:
        if side is None:
            return "center"
        return self._curve_callout_horizontal_alignment(side)

    def _annotation_label_penalty(
        self,
        *,
        ax,
        record: _AnnotationLabelRecord,
        side: str | None,
        text_x: float,
        text_y: float,
        text_transform,
    ) -> float:
        anchor_display = text_transform.transform((record.anchor_x, record.anchor_y))
        text_display = text_transform.transform((text_x, text_y))
        leader_length = float(
            np.hypot(text_display[0] - anchor_display[0], text_display[1] - anchor_display[1])
        )
        side_flip_penalty = 0.0
        if record.side is not None and side is not None and side != record.side:
            side_flip_penalty = 50.0
        depth_shift = abs(text_y - record.preferred_y)
        lateral_shift = abs(text_x - record.preferred_x)
        return leader_length * 0.01 + depth_shift * 0.5 + lateral_shift * 25.0 + side_flip_penalty

    def _place_annotation_label_records(
        self,
        ax,
        records: list[_AnnotationLabelRecord],
        window,
    ) -> list[_AnnotationLabelRecord]:
        if not records:
            return []

        from matplotlib.transforms import blended_transform_factory

        callout_style = self._style_section("curve_callouts")
        min_gap = max(
            float(window.stop - window.start) * 0.012,
            float(callout_style["min_vertical_gap_steps"]),
        )
        lower = float(window.start) + 0.5 * min_gap
        upper = float(window.stop) - 0.5 * min_gap
        if lower > upper:
            lower = float(window.start)
            upper = float(window.stop)
        renderer = self._curve_callout_renderer(ax)
        text_transform = blended_transform_factory(ax.transAxes, ax.transData)
        axes_bbox = ax.get_window_extent(renderer=renderer)
        edge_padding_px = float(callout_style["edge_padding_px"])
        obstacle_bboxes = self._annotation_obstacle_bboxes(ax, renderer)
        placed_bboxes: list[Any] = []
        ordered = sorted(
            records,
            key=lambda record: (
                -record.priority,
                -len(record.label),
                record.preferred_y,
            ),
        )
        for record in ordered:
            label_text = record.label
            if (
                record.label_mode == AnnotationLabelMode.DEDICATED_LANE
                and record.label_lane_start is not None
                and record.label_lane_end is not None
            ):
                available_width_ratio = max(record.label_lane_end - record.label_lane_start, 0.01)
                padding_ratio = min(
                    edge_padding_px / max(axes_bbox.width, 1.0),
                    available_width_ratio * 0.25,
                )
                label_text = self._wrap_annotation_label_text(
                    ax,
                    text=record.label,
                    available_width_ratio=max(available_width_ratio - 2.0 * padding_ratio, 0.01),
                    font_size_pt=record.font_size,
                    max_lines=2,
                )
            record.display_label = label_text
            best: tuple[float, str | None, float, float] | None = None
            for side, candidate_x in self._annotation_label_candidate_x_positions(
                record,
                callout_style,
            ):
                alignment = self._annotation_label_horizontal_alignment(side)
                for candidate_y in self._annotation_label_candidate_y_positions(
                    record,
                    lower=lower,
                    upper=upper,
                    min_gap=min_gap,
                ):
                    bbox = self._measure_curve_callout_bbox(
                        ax,
                        renderer=renderer,
                        label=record.display_label or record.label,
                        text_x=candidate_x,
                        text_y=candidate_y,
                        transform=text_transform,
                        fontsize=record.font_size,
                        color=record.color,
                        fontweight=record.font_weight,
                        fontstyle=record.font_style,
                        horizontal_alignment=alignment,
                    )
                    adjusted_x = self._adjust_curve_callout_x_to_fit(
                        text_x=candidate_x,
                        bbox=bbox,
                        axes_bbox=axes_bbox,
                        padding_px=edge_padding_px,
                    )
                    if not np.isclose(adjusted_x, candidate_x):
                        candidate_x = adjusted_x
                        bbox = self._measure_curve_callout_bbox(
                            ax,
                            renderer=renderer,
                            label=record.display_label or record.label,
                            text_x=candidate_x,
                            text_y=candidate_y,
                            transform=text_transform,
                            fontsize=record.font_size,
                            color=record.color,
                            fontweight=record.font_weight,
                            fontstyle=record.font_style,
                            horizontal_alignment=alignment,
                        )
                    candidate_y = self._adjust_curve_callout_y_to_fit(
                        text_x=candidate_x,
                        text_y=candidate_y,
                        bbox=bbox,
                        axes_bbox=axes_bbox,
                        padding_px=edge_padding_px,
                        transform=text_transform,
                    )
                    if not np.isclose(candidate_y, record.preferred_y):
                        bbox = self._measure_curve_callout_bbox(
                            ax,
                            renderer=renderer,
                            label=record.display_label or record.label,
                            text_x=candidate_x,
                            text_y=candidate_y,
                            transform=text_transform,
                            fontsize=record.font_size,
                            color=record.color,
                            fontweight=record.font_weight,
                            fontstyle=record.font_style,
                            horizontal_alignment=alignment,
                        )
                    if (
                        bbox.x0 < axes_bbox.x0 + edge_padding_px
                        or bbox.x1 > axes_bbox.x1 - edge_padding_px
                        or bbox.y0 < axes_bbox.y0 + edge_padding_px
                        or bbox.y1 > axes_bbox.y1 - edge_padding_px
                    ):
                        continue
                    if any(bbox.overlaps(other) for other in obstacle_bboxes):
                        continue
                    if any(bbox.overlaps(other) for other in placed_bboxes):
                        continue
                    penalty = self._annotation_label_penalty(
                        ax=ax,
                        record=record,
                        side=side,
                        text_x=candidate_x,
                        text_y=candidate_y,
                        text_transform=text_transform,
                    )
                    if best is None or penalty < best[0]:
                        best = (penalty, side, candidate_x, candidate_y)
            if best is None:
                continue
            _, placed_side, placed_x, placed_y = best
            record.placed_side = placed_side
            record.text_x = placed_x
            record.text_y = placed_y
            alignment = self._annotation_label_horizontal_alignment(placed_side)
            placed_bboxes.append(
                self._measure_curve_callout_bbox(
                    ax,
                    renderer=renderer,
                    label=record.display_label or record.label,
                    text_x=placed_x,
                    text_y=placed_y,
                    transform=text_transform,
                    fontsize=record.font_size,
                    color=record.color,
                    fontweight=record.font_weight,
                    fontstyle=record.font_style,
                    horizontal_alignment=alignment,
                )
            )
        return ordered

    def _draw_annotation_label_records(self, ax, records: list[_AnnotationLabelRecord]) -> None:
        if not records:
            return
        from matplotlib.transforms import blended_transform_factory

        text_transform = blended_transform_factory(ax.transAxes, ax.transData)
        for record in records:
            if record.text_x is None or record.text_y is None:
                continue
            side = record.placed_side
            ax.annotate(
                record.display_label or record.label,
                xy=(record.anchor_x, record.anchor_y),
                xycoords=text_transform,
                xytext=(record.text_x, record.text_y),
                textcoords=text_transform,
                fontsize=record.font_size,
                color=record.color,
                fontweight=record.font_weight,
                fontstyle=record.font_style,
                rotation=record.rotation,
                ha=self._annotation_label_horizontal_alignment(side),
                va="center",
                zorder=1.95,
                arrowprops=(
                    {
                        "arrowstyle": record.arrow_style,
                        "color": record.color,
                        "lw": record.arrow_linewidth,
                        "shrinkA": 0,
                        "shrinkB": 0,
                        "relpos": (0.5, 0.5)
                        if side is None
                        else self._curve_callout_arrow_relpos(side),
                    }
                    if record.arrow
                    else None
                ),
                annotation_clip=True,
            )

    def _draw_annotation_objects(self, ax, track, window) -> None:
        label_records: list[_AnnotationLabelRecord] = []
        for annotation in track.annotations:
            if isinstance(annotation, AnnotationIntervalSpec):
                self._draw_annotation_interval(ax, annotation, window)
            elif isinstance(annotation, AnnotationTextSpec):
                self._draw_annotation_text(ax, annotation, window)
            elif isinstance(annotation, AnnotationMarkerSpec):
                self._draw_annotation_marker_shape(ax, annotation, window)
                record = self._annotation_marker_label_record(
                    annotation,
                    self._style_section("curve_callouts"),
                )
                if record is not None:
                    label_records.append(record)
            elif isinstance(annotation, AnnotationArrowSpec):
                self._draw_annotation_arrow_line(ax, annotation, window)
                record = self._annotation_arrow_label_record(annotation)
                if record is not None:
                    if window.start <= record.preferred_y <= window.stop:
                        label_records.append(record)
            elif isinstance(annotation, AnnotationGlyphSpec):
                self._draw_annotation_glyph(ax, annotation, window)
        placed_labels = self._place_annotation_label_records(ax, label_records, window)
        self._draw_annotation_label_records(ax, placed_labels)

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
        reference_overlay = self._resolved_reference_overlay(track, element)

        if element.render_mode == "line":
            self._draw_curve_fill(
                ax,
                track,
                element,
                document,
                dataset,
                independent_curve_scales=independent_curve_scales,
            )

        if (
            reference_overlay is not None
            and reference_overlay.mode == ReferenceCurveOverlayMode.TICKS
        ):
            self._draw_reference_curve_ticks(
                ax,
                depth=depth,
                valid_mask=plot_data.valid_mask,
                overlay=reference_overlay,
                element=element,
            )
            return

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

    def _curve_callout_depth_step(self, document: LogDocument) -> float:
        if document.depth_axis.minor_step > 0:
            return float(document.depth_axis.minor_step)
        if document.depth_axis.major_step > 0:
            return float(document.depth_axis.major_step) / 5.0
        return 1.0

    def _interpolate_curve_x_at_depth(
        self,
        plot_data: _CurvePlotData,
        *,
        depth_value: float,
    ) -> float | None:
        mask = plot_data.valid_mask & np.isfinite(plot_data.plot_values)
        if not np.any(mask):
            return None
        source_depth = plot_data.depth[mask]
        source_values = plot_data.plot_values[mask]
        order = np.argsort(source_depth, kind="mergesort")
        source_depth = source_depth[order]
        source_values = source_values[order]
        unique_depth, unique_indices = np.unique(source_depth, return_index=True)
        if unique_depth.size == 0:
            return None
        if depth_value < unique_depth[0] or depth_value > unique_depth[-1]:
            return None
        unique_values = source_values[unique_indices]
        interpolated = float(
            np.interp(depth_value, unique_depth, unique_values, left=np.nan, right=np.nan)
        )
        if not np.isfinite(interpolated):
            return None
        return interpolated

    def _curve_callout_fraction(
        self,
        plot_value: float,
        plot_data: _CurvePlotData,
        *,
        independent_curve_scales: bool,
    ) -> float:
        if plot_data.x_is_fractional:
            return float(np.clip(plot_value, 0.0, 1.0))
        if independent_curve_scales or (
            plot_data.scale is not None and plot_data.scale.kind == ScaleKind.TANGENTIAL
        ):
            return float(np.clip(plot_value, 0.0, 1.0))
        if plot_data.scale is not None:
            normalized, mask = self._normalize_curve_values(
                np.asarray([plot_value], dtype=float),
                plot_data.scale,
            )
            if np.any(mask):
                return float(np.clip(normalized[0], 0.0, 1.0))
        finite = plot_data.plot_values[plot_data.valid_mask & np.isfinite(plot_data.plot_values)]
        if finite.size < 2 or np.isclose(float(np.nanmin(finite)), float(np.nanmax(finite))):
            return 0.5
        fraction = (plot_value - float(np.nanmin(finite))) / (
            float(np.nanmax(finite)) - float(np.nanmin(finite))
        )
        return float(np.clip(fraction, 0.0, 1.0))

    def _curve_callout_label(self, element: CurveElement, callout) -> str:
        if callout.label is not None:
            return callout.label
        return element.label or element.channel

    def _curve_callout_window_bounds(self, window) -> tuple[float, float]:
        start = float(window.start)
        stop = float(window.stop)
        return (start, stop) if start <= stop else (stop, start)

    def _curve_callout_default_top_distance(
        self,
        document: LogDocument,
        callout_style: dict[str, Any],
    ) -> float:
        return self._curve_callout_depth_step(document) * float(callout_style["top_distance_steps"])

    def _curve_callout_default_bottom_distance(
        self,
        document: LogDocument,
        callout_style: dict[str, Any],
    ) -> float:
        return self._curve_callout_depth_step(document) * float(
            callout_style["bottom_distance_steps"]
        )

    def _expanded_curve_callout_anchors(
        self,
        document: LogDocument,
        callout,
        callout_style: dict[str, Any],
        window,
        *,
        section_start: float,
        section_stop: float,
    ) -> list[float]:
        visible_lower, visible_upper = self._curve_callout_window_bounds(window)
        section_lower, section_upper = (
            (section_start, section_stop)
            if section_start <= section_stop
            else (section_stop, section_start)
        )
        base_depth = float(callout.depth)
        if callout.every is None:
            if base_depth < visible_lower or base_depth > visible_upper:
                return []
            return [base_depth]

        every = float(callout.every)
        if callout.placement == "top":
            start_depth = section_lower + (
                callout.distance_from_top
                if callout.distance_from_top is not None
                else self._curve_callout_default_top_distance(document, callout_style)
            )
            return [
                float(depth_value)
                for depth_value in np.arange(start_depth, section_upper + 0.5 * every, every)
                if visible_lower <= depth_value <= visible_upper
            ]
        if callout.placement == "bottom":
            start_depth = section_upper - (
                callout.distance_from_bottom
                if callout.distance_from_bottom is not None
                else self._curve_callout_default_bottom_distance(document, callout_style)
            )
            return [
                float(depth_value)
                for depth_value in np.arange(start_depth, section_lower - 0.5 * every, -every)
                if visible_lower <= depth_value <= visible_upper
            ]
        if callout.placement == "top_and_bottom":
            top_start = section_lower + (
                callout.distance_from_top
                if callout.distance_from_top is not None
                else self._curve_callout_default_top_distance(document, callout_style)
            )
            bottom_start = section_upper - (
                callout.distance_from_bottom
                if callout.distance_from_bottom is not None
                else self._curve_callout_default_bottom_distance(document, callout_style)
            )
            anchor_values = {
                round(float(depth_value), 9)
                for depth_value in np.arange(top_start, section_upper + 0.5 * every, every)
                if visible_lower <= depth_value <= visible_upper
            }
            anchor_values.update(
                round(float(depth_value), 9)
                for depth_value in np.arange(bottom_start, section_lower - 0.5 * every, -every)
                if visible_lower <= depth_value <= visible_upper
            )
            return [float(depth_value) for depth_value in sorted(anchor_values)]

        first_index = math.floor((section_lower - base_depth) / every)
        anchors: list[float] = []
        while True:
            depth_value = base_depth + first_index * every
            if depth_value > section_upper:
                break
            if visible_lower <= depth_value <= visible_upper:
                anchors.append(float(depth_value))
            first_index += 1
        return anchors

    def _curve_callout_target_y(
        self,
        *,
        callout,
        anchor_depth: float,
        default_offset: float,
    ) -> float:
        return float(
            anchor_depth
            + (
                callout.depth_offset
                if callout.depth_offset is not None
                else 0.0
                if callout.every is not None and callout.placement != "inline"
                else default_offset
            )
        )

    def _curve_callout_records(
        self,
        track,
        document: LogDocument,
        dataset: WellDataset,
        window,
        *,
        independent_curve_scales: bool,
    ) -> list[_CurveCalloutRenderRecord]:
        callout_style = self._style_section("curve_callouts")
        default_offset = self._curve_callout_depth_step(document) * float(
            callout_style["default_depth_offset_steps"]
        )
        section_start, section_stop = document.resolve_depth_range(dataset, self.registry)
        records: list[_CurveCalloutRenderRecord] = []
        for element in self._curve_elements(track):
            if not element.callouts:
                continue
            plot_data = self._curve_plot_data(
                track,
                element,
                document,
                dataset,
                independent_curve_scales=independent_curve_scales,
            )
            for callout in element.callouts:
                for anchor_depth in self._expanded_curve_callout_anchors(
                    document,
                    callout,
                    callout_style,
                    window,
                    section_start=float(section_start),
                    section_stop=float(section_stop),
                ):
                    anchor_x = self._interpolate_curve_x_at_depth(
                        plot_data,
                        depth_value=anchor_depth,
                    )
                    if anchor_x is None:
                        continue
                    fraction = self._curve_callout_fraction(
                        anchor_x,
                        plot_data,
                        independent_curve_scales=independent_curve_scales,
                    )
                    side = callout.side
                    if side == "auto":
                        side = "right" if fraction <= 0.5 else "left"
                    text_x = (
                        callout.text_x
                        if callout.text_x is not None
                        else float(callout_style["right_text_x"])
                        if side == "right"
                        else float(callout_style["left_text_x"])
                    )
                    records.append(
                        _CurveCalloutRenderRecord(
                            label=self._curve_callout_label(element, callout),
                            side=side,
                            allow_side_flip=callout.side == "auto" and callout.text_x is None,
                            curve_key=id(element),
                            anchor_x=float(anchor_x),
                            anchor_y=float(anchor_depth),
                            text_x=float(text_x),
                            desired_text_y=self._curve_callout_target_y(
                                callout=callout,
                                anchor_depth=float(anchor_depth),
                                default_offset=default_offset,
                            ),
                            color=callout.color or element.style.color,
                            font_size=(
                                callout.font_size
                                if callout.font_size is not None
                                else float(callout_style["font_size"])
                            ),
                            font_weight=callout.font_weight,
                            font_style=callout.font_style,
                            arrow=callout.arrow,
                            arrow_style=callout.arrow_style or str(callout_style["arrow_style"]),
                            arrow_linewidth=(
                                callout.arrow_linewidth
                                if callout.arrow_linewidth is not None
                                else float(callout_style["arrow_linewidth"])
                            ),
                        )
                    )
        return records

    def _curve_callout_horizontal_alignment(self, side: str) -> str:
        return "left" if side == "right" else "right"

    def _curve_callout_arrow_relpos(self, side: str) -> tuple[float, float]:
        return (0.0, 0.5) if side == "right" else (1.0, 0.5)

    def _curve_callout_text_kwargs(
        self,
        record: _CurveCalloutRenderRecord,
        *,
        side: str,
    ) -> dict[str, Any]:
        return {
            "fontsize": record.font_size,
            "color": record.color,
            "fontweight": record.font_weight,
            "fontstyle": record.font_style,
            "ha": self._curve_callout_horizontal_alignment(side),
            "va": "center",
            "zorder": 4.2,
        }

    def _curve_callout_candidate_sides(
        self,
        record: _CurveCalloutRenderRecord,
    ) -> list[str]:
        sides = [record.side]
        if record.allow_side_flip:
            alternate = "left" if record.side == "right" else "right"
            sides.append(alternate)
        return sides

    def _curve_callout_candidate_x_positions(
        self,
        record: _CurveCalloutRenderRecord,
        side: str,
        callout_style: dict[str, Any],
    ) -> list[float]:
        if side == record.side:
            base = record.text_x
        else:
            base_key = "right_text_x" if side == "right" else "left_text_x"
            base = float(callout_style[base_key])
        lane_count = max(int(callout_style["lane_count"]), 1)
        lane_step = float(callout_style["lane_step_x"])
        candidates: list[float] = []
        for index in range(lane_count):
            offset = lane_step * index
            value = base - offset if side == "right" else base + offset
            value = float(np.clip(value, 0.0, 1.0))
            if any(np.isclose(value, current, atol=1e-6) for current in candidates):
                continue
            candidates.append(value)
        return candidates

    def _curve_callout_candidate_y_positions(
        self,
        record: _CurveCalloutRenderRecord,
        *,
        lower: float,
        upper: float,
        min_gap: float,
    ) -> list[float]:
        offsets = [0.0]
        for index in range(1, 4):
            offsets.extend((-index * min_gap, index * min_gap))
        candidates: list[float] = []
        for offset in offsets:
            value = float(np.clip(record.desired_text_y + offset, lower, upper))
            if any(np.isclose(value, current, atol=1e-6) for current in candidates):
                continue
            candidates.append(value)
        return candidates

    def _curve_callout_renderer(self, ax):
        canvas = ax.figure.canvas
        get_renderer = getattr(canvas, "get_renderer", None)
        if get_renderer is None:
            canvas.draw()
            get_renderer = canvas.get_renderer
        renderer = get_renderer()
        if renderer is None:
            canvas.draw()
            renderer = get_renderer()
        return renderer

    def _measure_curve_callout_bbox(
        self,
        ax,
        *,
        renderer,
        label: str,
        text_x: float,
        text_y: float,
        transform,
        fontsize: float,
        color: str,
        fontweight: str,
        fontstyle: str,
        horizontal_alignment: str,
    ):
        text = ax.text(
            text_x,
            text_y,
            label,
            transform=transform,
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            fontstyle=fontstyle,
            ha=horizontal_alignment,
            va="center",
            alpha=0.0,
        )
        try:
            return text.get_window_extent(renderer=renderer)
        finally:
            text.remove()

    def _adjust_curve_callout_x_to_fit(
        self,
        *,
        text_x: float,
        bbox,
        axes_bbox,
        padding_px: float,
    ) -> float:
        shift_px = 0.0
        if bbox.x0 < axes_bbox.x0 + padding_px:
            shift_px = (axes_bbox.x0 + padding_px) - bbox.x0
        elif bbox.x1 > axes_bbox.x1 - padding_px:
            shift_px = (axes_bbox.x1 - padding_px) - bbox.x1
        if np.isclose(shift_px, 0.0):
            return text_x
        width_px = max(float(axes_bbox.width), 1.0)
        return float(np.clip(text_x + shift_px / width_px, 0.0, 1.0))

    def _adjust_curve_callout_y_to_fit(
        self,
        *,
        text_x: float,
        text_y: float,
        bbox,
        axes_bbox,
        padding_px: float,
        transform,
    ) -> float:
        shift_px = 0.0
        if bbox.y0 < axes_bbox.y0 + padding_px:
            shift_px = (axes_bbox.y0 + padding_px) - bbox.y0
        elif bbox.y1 > axes_bbox.y1 - padding_px:
            shift_px = (axes_bbox.y1 - padding_px) - bbox.y1
        if np.isclose(shift_px, 0.0):
            return text_y
        display_x, display_y = transform.transform((text_x, text_y))
        _, adjusted_y = transform.inverted().transform((display_x, display_y + shift_px))
        return float(adjusted_y)

    def _curve_display_point_sets(
        self,
        ax,
        track,
        document: LogDocument,
        dataset: WellDataset,
        *,
        independent_curve_scales: bool,
    ) -> dict[int, np.ndarray]:
        point_sets: dict[int, np.ndarray] = {}
        for element in self._curve_elements(track):
            plot_data = self._curve_plot_data(
                track,
                element,
                document,
                dataset,
                independent_curve_scales=independent_curve_scales,
            )
            mask = plot_data.valid_mask & np.isfinite(plot_data.plot_values)
            if not np.any(mask):
                continue
            points = np.column_stack((plot_data.plot_values[mask], plot_data.depth[mask]))
            point_sets[id(element)] = ax.transData.transform(points)
        return point_sets

    def _count_curve_points_in_bbox(
        self,
        points: np.ndarray,
        bbox,
        *,
        padding_px: float,
    ) -> int:
        if points.size == 0:
            return 0
        x0 = bbox.x0 - padding_px
        x1 = bbox.x1 + padding_px
        y0 = bbox.y0 - padding_px
        y1 = bbox.y1 + padding_px
        inside = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        return int(np.count_nonzero(inside))

    def _curve_callout_penalty(
        self,
        *,
        ax,
        record: _CurveCalloutRenderRecord,
        side: str,
        text_x: float,
        text_y: float,
        bbox,
        text_transform,
        point_sets: dict[int, np.ndarray],
        curve_buffer_px: float,
    ) -> float:
        own_points = point_sets.get(record.curve_key, np.empty((0, 2)))
        own_overlap = self._count_curve_points_in_bbox(
            own_points,
            bbox,
            padding_px=curve_buffer_px,
        )
        other_overlap = sum(
            self._count_curve_points_in_bbox(points, bbox, padding_px=curve_buffer_px)
            for key, points in point_sets.items()
            if key != record.curve_key
        )
        anchor_display = ax.transData.transform((record.anchor_x, record.anchor_y))
        text_display = text_transform.transform((text_x, text_y))
        leader_length = float(
            np.hypot(text_display[0] - anchor_display[0], text_display[1] - anchor_display[1])
        )
        side_flip_penalty = 0.0 if side == record.side else 50.0
        depth_shift = abs(text_y - record.desired_text_y)
        return (
            own_overlap * 25.0
            + other_overlap * 6.0
            + leader_length * 0.01
            + depth_shift * 0.5
            + side_flip_penalty
        )

    def _place_curve_callouts(
        self,
        ax,
        track,
        document: LogDocument,
        dataset: WellDataset,
        window,
        *,
        independent_curve_scales: bool,
    ) -> list[_CurveCalloutRenderRecord]:
        records = self._curve_callout_records(
            track,
            document,
            dataset,
            window,
            independent_curve_scales=independent_curve_scales,
        )
        if not records:
            return []

        from matplotlib.transforms import blended_transform_factory

        callout_style = self._style_section("curve_callouts")
        min_gap = self._curve_callout_depth_step(document) * float(
            callout_style["min_vertical_gap_steps"]
        )
        lower = float(window.start) + 0.5 * min_gap
        upper = float(window.stop) - 0.5 * min_gap
        if lower > upper:
            lower = float(window.start)
            upper = float(window.stop)
        renderer = self._curve_callout_renderer(ax)
        text_transform = blended_transform_factory(ax.transAxes, ax.transData)
        axes_bbox = ax.get_window_extent(renderer=renderer)
        edge_padding_px = float(callout_style["edge_padding_px"])
        curve_buffer_px = float(callout_style["curve_buffer_px"])
        point_sets = self._curve_display_point_sets(
            ax,
            track,
            document,
            dataset,
            independent_curve_scales=independent_curve_scales,
        )
        placed_bboxes = []
        ordered = sorted(
            records,
            key=lambda record: (
                record.allow_side_flip,
                -len(record.label),
                record.desired_text_y,
            ),
        )
        for record in ordered:
            best: tuple[float, str, float, float] | None = None
            for side in self._curve_callout_candidate_sides(record):
                horizontal_alignment = self._curve_callout_horizontal_alignment(side)
                for candidate_x in self._curve_callout_candidate_x_positions(
                    record,
                    side,
                    callout_style,
                ):
                    for candidate_y in self._curve_callout_candidate_y_positions(
                        record,
                        lower=lower,
                        upper=upper,
                        min_gap=min_gap,
                    ):
                        bbox = self._measure_curve_callout_bbox(
                            ax,
                            renderer=renderer,
                            label=record.label,
                            text_x=candidate_x,
                            text_y=candidate_y,
                            transform=text_transform,
                            fontsize=record.font_size,
                            color=record.color,
                            fontweight=record.font_weight,
                            fontstyle=record.font_style,
                            horizontal_alignment=horizontal_alignment,
                        )
                        adjusted_x = self._adjust_curve_callout_x_to_fit(
                            text_x=candidate_x,
                            bbox=bbox,
                            axes_bbox=axes_bbox,
                            padding_px=edge_padding_px,
                        )
                        if not np.isclose(adjusted_x, candidate_x):
                            adjusted_candidate_x = adjusted_x
                            bbox = self._measure_curve_callout_bbox(
                                ax,
                                renderer=renderer,
                                label=record.label,
                                text_x=adjusted_candidate_x,
                                text_y=candidate_y,
                                transform=text_transform,
                                fontsize=record.font_size,
                                color=record.color,
                                fontweight=record.font_weight,
                                fontstyle=record.font_style,
                                horizontal_alignment=horizontal_alignment,
                            )
                        else:
                            adjusted_candidate_x = candidate_x
                        candidate_y = self._adjust_curve_callout_y_to_fit(
                            text_x=adjusted_candidate_x,
                            text_y=candidate_y,
                            bbox=bbox,
                            axes_bbox=axes_bbox,
                            padding_px=edge_padding_px,
                            transform=text_transform,
                        )
                        if not np.isclose(candidate_y, record.desired_text_y):
                            bbox = self._measure_curve_callout_bbox(
                                ax,
                                renderer=renderer,
                                label=record.label,
                                text_x=adjusted_candidate_x,
                                text_y=candidate_y,
                                transform=text_transform,
                                fontsize=record.font_size,
                                color=record.color,
                                fontweight=record.font_weight,
                                fontstyle=record.font_style,
                                horizontal_alignment=horizontal_alignment,
                            )
                        if (
                            bbox.x0 < axes_bbox.x0 + edge_padding_px
                            or bbox.x1 > axes_bbox.x1 - edge_padding_px
                            or bbox.y0 < axes_bbox.y0 + edge_padding_px
                            or bbox.y1 > axes_bbox.y1 - edge_padding_px
                        ):
                            continue
                        if any(bbox.overlaps(other) for other in placed_bboxes):
                            continue
                        penalty = self._curve_callout_penalty(
                            ax=ax,
                            record=record,
                            side=side,
                            text_x=adjusted_candidate_x,
                            text_y=candidate_y,
                            bbox=bbox,
                            text_transform=text_transform,
                            point_sets=point_sets,
                            curve_buffer_px=curve_buffer_px,
                        )
                        if best is None or penalty < best[0]:
                            best = (penalty, side, adjusted_candidate_x, candidate_y)
            if best is None:
                continue
            _, placed_side, placed_x, placed_y = best
            record.placed_side = placed_side
            record.text_x = placed_x
            record.text_y = placed_y
            horizontal_alignment = self._curve_callout_horizontal_alignment(placed_side)
            placed_bbox = self._measure_curve_callout_bbox(
                ax,
                renderer=renderer,
                label=record.label,
                text_x=placed_x,
                text_y=placed_y,
                transform=text_transform,
                fontsize=record.font_size,
                color=record.color,
                fontweight=record.font_weight,
                fontstyle=record.font_style,
                horizontal_alignment=horizontal_alignment,
            )
            placed_bboxes.append(placed_bbox)
        return ordered

    def _draw_curve_callouts(
        self,
        ax,
        track,
        document: LogDocument,
        dataset: WellDataset,
        window,
        *,
        independent_curve_scales: bool,
    ) -> None:
        records = self._place_curve_callouts(
            ax,
            track,
            document,
            dataset,
            window,
            independent_curve_scales=independent_curve_scales,
        )
        if not records:
            return

        from matplotlib.transforms import blended_transform_factory

        text_transform = blended_transform_factory(ax.transAxes, ax.transData)
        for record in records:
            if record.text_y is None:
                continue
            placed_side = record.placed_side or record.side
            text_kwargs = self._curve_callout_text_kwargs(record, side=placed_side)
            arrowprops = None
            if record.arrow:
                arrowprops = {
                    "arrowstyle": record.arrow_style,
                    "color": record.color,
                    "lw": record.arrow_linewidth,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "relpos": self._curve_callout_arrow_relpos(placed_side),
                }
            ax.annotate(
                record.label,
                xy=(record.anchor_x, record.anchor_y),
                xycoords=ax.transData,
                xytext=(record.text_x, record.text_y),
                textcoords=text_transform,
                **text_kwargs,
                arrowprops=arrowprops,
                annotation_clip=True,
            )

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
