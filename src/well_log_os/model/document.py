from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..errors import TemplateValidationError
from ..units import DEFAULT_UNITS, SimpleUnitRegistry
from .dataset import WellDataset


class TrackKind(StrEnum):
    REFERENCE = "reference"
    NORMAL = "normal"
    ARRAY = "array"
    ANNOTATION = "annotation"
    # Backward-compatible aliases.
    DEPTH = "reference"
    CURVE = "normal"
    IMAGE = "array"


class ScaleKind(StrEnum):
    LINEAR = "linear"
    LOG = "log"
    TANGENTIAL = "tangential"


class GridScaleKind(StrEnum):
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    TANGENTIAL = "tangential"


class GridDisplayMode(StrEnum):
    BELOW = "below"
    ABOVE = "above"
    NONE = "none"


class GridSpacingMode(StrEnum):
    COUNT = "count"
    SCALE = "scale"


class TrackHeaderObjectKind(StrEnum):
    TITLE = "title"
    SCALE = "scale"
    LEGEND = "legend"
    DIVISIONS = "divisions"


class ReferenceAxisKind(StrEnum):
    DEPTH = "depth"
    TIME = "time"


class NumberFormatKind(StrEnum):
    AUTOMATIC = "automatic"
    FIXED = "fixed"
    SCIENTIFIC = "scientific"
    CONCISE = "concise"


class RasterProfileKind(StrEnum):
    GENERIC = "generic"
    VDL = "vdl"
    WAVEFORM = "waveform"


class RasterNormalizationKind(StrEnum):
    AUTO = "auto"
    NONE = "none"
    TRACE_MAXABS = "trace_maxabs"
    GLOBAL_MAXABS = "global_maxabs"


class RasterColorbarPosition(StrEnum):
    RIGHT = "right"
    HEADER = "header"


class CurveFillKind(StrEnum):
    BETWEEN_CURVES = "between_curves"
    BETWEEN_INSTANCES = "between_instances"
    TO_LOWER_LIMIT = "to_lower_limit"
    TO_UPPER_LIMIT = "to_upper_limit"
    BASELINE_SPLIT = "baseline_split"


class ReferenceCurveOverlayMode(StrEnum):
    CURVE = "curve"
    INDICATOR = "indicator"
    TICKS = "ticks"


class ReferenceCurveTickSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass(slots=True)
class StyleSpec:
    color: str = "black"
    line_width: float = 0.8
    line_style: str = "-"
    opacity: float = 1.0
    fill_color: str | None = None
    fill_alpha: float = 0.2
    colormap: str = "viridis"


@dataclass(slots=True)
class ScaleSpec:
    kind: ScaleKind = ScaleKind.LINEAR
    minimum: float = 0.0
    maximum: float = 1.0
    reverse: bool = False

    def __post_init__(self) -> None:
        if self.minimum == self.maximum:
            raise ValueError("Scale minimum and maximum must differ.")
        if self.kind == ScaleKind.LOG and (self.minimum <= 0 or self.maximum <= 0):
            raise ValueError("Log scales require positive bounds.")


@dataclass(slots=True)
class GridSpec:
    major: bool = True
    minor: bool = True
    major_alpha: float = 0.35
    minor_alpha: float = 0.15
    horizontal_display: GridDisplayMode = GridDisplayMode.BELOW
    horizontal_major_visible: bool = True
    horizontal_minor_visible: bool = True
    horizontal_major_color: str | None = None
    horizontal_minor_color: str | None = None
    horizontal_major_thickness: float | None = None
    horizontal_minor_thickness: float | None = None
    horizontal_major_alpha: float | None = None
    horizontal_minor_alpha: float | None = None
    vertical_display: GridDisplayMode = GridDisplayMode.BELOW
    vertical_main_visible: bool = True
    vertical_main_line_count: int = 4
    vertical_main_thickness: float | None = None
    vertical_main_color: str | None = None
    vertical_main_alpha: float = 0.35
    vertical_main_scale: GridScaleKind = GridScaleKind.LINEAR
    vertical_main_spacing_mode: GridSpacingMode = GridSpacingMode.COUNT
    vertical_secondary_visible: bool = True
    vertical_secondary_line_count: int = 4
    vertical_secondary_thickness: float | None = None
    vertical_secondary_color: str | None = None
    vertical_secondary_alpha: float = 0.15
    vertical_secondary_scale: GridScaleKind = GridScaleKind.LINEAR
    vertical_secondary_spacing_mode: GridSpacingMode = GridSpacingMode.COUNT

    def __post_init__(self) -> None:
        for name, alpha in (
            ("major_alpha", self.major_alpha),
            ("minor_alpha", self.minor_alpha),
            ("horizontal_major_alpha", self.horizontal_major_alpha),
            ("horizontal_minor_alpha", self.horizontal_minor_alpha),
            ("vertical_main_alpha", self.vertical_main_alpha),
            ("vertical_secondary_alpha", self.vertical_secondary_alpha),
        ):
            if alpha is None:
                continue
            if alpha < 0 or alpha > 1:
                raise ValueError(f"Grid {name} must be between 0 and 1.")

        for name, thickness in (
            ("horizontal_major_thickness", self.horizontal_major_thickness),
            ("horizontal_minor_thickness", self.horizontal_minor_thickness),
        ):
            if thickness is None:
                continue
            if thickness <= 0:
                raise ValueError(f"Grid {name} must be positive when provided.")

        if self.vertical_main_line_count <= 0:
            raise ValueError("Grid vertical_main_line_count must be positive.")
        if self.vertical_secondary_line_count <= 0:
            raise ValueError("Grid vertical_secondary_line_count must be positive.")
        if self.vertical_main_thickness is not None and self.vertical_main_thickness <= 0:
            raise ValueError("Grid vertical_main_thickness must be positive when provided.")
        if self.vertical_secondary_thickness is not None and self.vertical_secondary_thickness <= 0:
            raise ValueError("Grid vertical_secondary_thickness must be positive when provided.")


@dataclass(slots=True)
class ReferenceTrackSpec:
    axis: ReferenceAxisKind = ReferenceAxisKind.DEPTH
    define_layout: bool = True
    unit: str | None = None
    scale_ratio: int | None = None
    major_step: float | None = None
    minor_step: float | None = None
    secondary_grid_display: bool = True
    secondary_grid_line_count: int = 4
    display_unit_in_header: bool = True
    display_scale_in_header: bool = True
    display_annotations_in_header: bool = True
    number_format: NumberFormatKind = NumberFormatKind.AUTOMATIC
    precision: int = 2
    values_orientation: str = "horizontal"
    events: tuple[ReferenceEventSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.scale_ratio is not None and self.scale_ratio <= 0:
            raise ValueError("Reference track scale_ratio must be positive when provided.")
        if self.major_step is not None and self.major_step <= 0:
            raise ValueError("Reference track major_step must be positive when provided.")
        if self.minor_step is not None and self.minor_step <= 0:
            raise ValueError("Reference track minor_step must be positive when provided.")
        if self.secondary_grid_line_count <= 0:
            raise ValueError("Reference track secondary_grid_line_count must be positive.")
        if self.precision < 0:
            raise ValueError("Reference track precision must be non-negative.")
        orientation = self.values_orientation.strip().lower()
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("Reference track values_orientation must be horizontal or vertical.")
        self.values_orientation = orientation


@dataclass(slots=True, frozen=True)
class TrackHeaderObjectSpec:
    kind: TrackHeaderObjectKind
    enabled: bool = True
    reserve_space: bool = True
    line_units: int = 1

    def __post_init__(self) -> None:
        if self.line_units <= 0:
            raise ValueError("Track header object line_units must be positive.")


def _default_track_header_objects() -> tuple[TrackHeaderObjectSpec, ...]:
    return (
        TrackHeaderObjectSpec(kind=TrackHeaderObjectKind.TITLE, line_units=1),
        TrackHeaderObjectSpec(kind=TrackHeaderObjectKind.SCALE, line_units=1),
        TrackHeaderObjectSpec(kind=TrackHeaderObjectKind.LEGEND, line_units=2),
        TrackHeaderObjectSpec(
            kind=TrackHeaderObjectKind.DIVISIONS,
            enabled=False,
            reserve_space=False,
            line_units=1,
        ),
    )


@dataclass(slots=True)
class TrackHeaderSpec:
    objects: tuple[TrackHeaderObjectSpec, ...] = field(
        default_factory=_default_track_header_objects
    )

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("Track header must contain at least one object.")
        kinds = [item.kind for item in self.objects]
        if len(set(kinds)) != len(kinds):
            raise ValueError("Track header object kinds must be unique per track.")

    def reserved_objects(self) -> tuple[TrackHeaderObjectSpec, ...]:
        return tuple(item for item in self.objects if item.reserve_space)


@dataclass(slots=True)
class CurveValueLabelsSpec:
    step: float = 5.0
    number_format: NumberFormatKind = NumberFormatKind.AUTOMATIC
    precision: int = 2
    color: str | None = None
    font_size: float = 5.5
    font_family: str | None = None
    font_weight: str = "normal"
    font_style: str = "normal"
    horizontal_alignment: str = "center"
    vertical_alignment: str = "center"

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("Curve value-label step must be positive.")
        if self.precision < 0:
            raise ValueError("Curve value-label precision must be non-negative.")
        if self.font_size <= 0:
            raise ValueError("Curve value-label font_size must be positive.")
        if self.horizontal_alignment not in {"left", "center", "right"}:
            raise ValueError(
                "Curve value-label horizontal_alignment must be left, center, or right."
            )
        if self.vertical_alignment not in {"top", "center", "bottom"}:
            raise ValueError("Curve value-label vertical_alignment must be top, center, or bottom.")


@dataclass(slots=True)
class CurveHeaderDisplaySpec:
    show_name: bool = True
    show_unit: bool = True
    show_limits: bool = True
    show_color: bool = True
    wrap_name: bool = False


@dataclass(slots=True)
class ReferenceCurveOverlaySpec:
    mode: ReferenceCurveOverlayMode = ReferenceCurveOverlayMode.CURVE
    lane_start: float | None = None
    lane_end: float | None = None
    tick_side: ReferenceCurveTickSide = ReferenceCurveTickSide.BOTH
    tick_length_ratio: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if (self.lane_start is None) != (self.lane_end is None):
            raise ValueError(
                "Reference curve overlay lane_start and lane_end must be set together."
            )
        if self.lane_start is not None and self.lane_end is not None:
            if not 0.0 <= self.lane_start < self.lane_end <= 1.0:
                raise ValueError(
                    "Reference curve overlay lane_start/lane_end must satisfy "
                    "0 <= lane_start < lane_end <= 1."
                )
        if self.tick_length_ratio is not None and self.tick_length_ratio <= 0:
            raise ValueError(
                "Reference curve overlay tick_length_ratio must be positive when provided."
            )


@dataclass(slots=True)
class ReferenceEventSpec:
    depth: float
    label: str = ""
    color: str = "#222222"
    line_style: str = "-"
    line_width: float = 0.7
    tick_side: ReferenceCurveTickSide = ReferenceCurveTickSide.RIGHT
    tick_length_ratio: float | None = None
    lane_start: float | None = None
    lane_end: float | None = None
    text_side: str = "auto"
    text_x: float | None = None
    depth_offset: float | None = None
    font_size: float | None = None
    font_weight: str = "bold"
    font_style: str = "normal"
    arrow: bool = True
    arrow_style: str | None = None
    arrow_linewidth: float | None = None

    def __post_init__(self) -> None:
        if self.label and not str(self.label).strip():
            raise ValueError("Reference event label must be non-empty when provided.")
        if not str(self.color).strip():
            raise ValueError("Reference event color must be non-empty.")
        if not str(self.line_style).strip():
            raise ValueError("Reference event line_style must be non-empty.")
        if self.line_width <= 0:
            raise ValueError("Reference event line_width must be positive.")
        if self.tick_length_ratio is not None and self.tick_length_ratio <= 0:
            raise ValueError("Reference event tick_length_ratio must be positive when provided.")
        if (self.lane_start is None) != (self.lane_end is None):
            raise ValueError("Reference event lane_start and lane_end must be set together.")
        if self.lane_start is not None and self.lane_end is not None:
            if not 0.0 <= self.lane_start < self.lane_end <= 1.0:
                raise ValueError(
                    "Reference event lane_start/lane_end must satisfy "
                    "0 <= lane_start < lane_end <= 1."
                )
        side = self.text_side.strip().lower()
        if side not in {"auto", "left", "right"}:
            raise ValueError("Reference event text_side must be auto, left, or right.")
        self.text_side = side
        if self.text_x is not None and not 0.0 <= self.text_x <= 1.0:
            raise ValueError("Reference event text_x must be between 0 and 1.")
        if self.font_size is not None and self.font_size <= 0:
            raise ValueError("Reference event font_size must be positive when provided.")
        if not str(self.font_weight).strip():
            raise ValueError("Reference event font_weight must be non-empty.")
        if not str(self.font_style).strip():
            raise ValueError("Reference event font_style must be non-empty.")
        if self.arrow_style is not None and not str(self.arrow_style).strip():
            raise ValueError("Reference event arrow_style must be non-empty when provided.")
        if self.arrow_linewidth is not None and self.arrow_linewidth <= 0:
            raise ValueError("Reference event arrow_linewidth must be positive when provided.")


@dataclass(slots=True)
class CurveCalloutSpec:
    depth: float
    label: str | None = None
    side: str = "auto"
    placement: str = "inline"
    text_x: float | None = None
    depth_offset: float | None = None
    distance_from_top: float | None = None
    distance_from_bottom: float | None = None
    every: float | None = None
    color: str | None = None
    font_size: float | None = None
    font_weight: str = "bold"
    font_style: str = "normal"
    arrow: bool = True
    arrow_style: str | None = None
    arrow_linewidth: float | None = None

    def __post_init__(self) -> None:
        if self.label is not None and not str(self.label).strip():
            raise ValueError("Curve callout label must be non-empty when provided.")
        side = self.side.strip().lower()
        if side not in {"auto", "left", "right"}:
            raise ValueError("Curve callout side must be auto, left, or right.")
        self.side = side
        placement = self.placement.strip().lower()
        if placement not in {"inline", "top", "bottom", "top_and_bottom"}:
            raise ValueError(
                "Curve callout placement must be inline, top, bottom, or top_and_bottom."
            )
        self.placement = placement
        if self.text_x is not None and (self.text_x < 0 or self.text_x > 1):
            raise ValueError("Curve callout text_x must be between 0 and 1.")
        if self.distance_from_top is not None and self.distance_from_top < 0:
            raise ValueError("Curve callout distance_from_top must be non-negative.")
        if self.distance_from_bottom is not None and self.distance_from_bottom < 0:
            raise ValueError("Curve callout distance_from_bottom must be non-negative.")
        if self.every is not None and self.every <= 0:
            raise ValueError("Curve callout every must be positive when provided.")
        if self.color is not None and not str(self.color).strip():
            raise ValueError("Curve callout color must be non-empty when provided.")
        if self.font_size is not None and self.font_size <= 0:
            raise ValueError("Curve callout font_size must be positive when provided.")
        if not str(self.font_weight).strip():
            raise ValueError("Curve callout font_weight must be non-empty.")
        if not str(self.font_style).strip():
            raise ValueError("Curve callout font_style must be non-empty.")
        if self.arrow_style is not None and not str(self.arrow_style).strip():
            raise ValueError("Curve callout arrow_style must be non-empty when provided.")
        if self.arrow_linewidth is not None and self.arrow_linewidth <= 0:
            raise ValueError("Curve callout arrow_linewidth must be positive when provided.")


@dataclass(slots=True)
class CurveFillCrossoverSpec:
    enabled: bool = False
    left_color: str | None = None
    right_color: str | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.left_color is not None and not str(self.left_color).strip():
            raise ValueError("Curve fill crossover left_color must be non-empty when provided.")
        if self.right_color is not None and not str(self.right_color).strip():
            raise ValueError("Curve fill crossover right_color must be non-empty when provided.")
        if self.alpha is not None and (self.alpha < 0 or self.alpha > 1):
            raise ValueError("Curve fill crossover alpha must be between 0 and 1.")


@dataclass(slots=True)
class CurveFillBaselineSpec:
    value: float
    lower_color: str | None = None
    upper_color: str | None = None
    line_color: str | None = None
    line_width: float = 0.6
    line_style: str = "--"

    def __post_init__(self) -> None:
        if self.lower_color is not None and not str(self.lower_color).strip():
            raise ValueError("Curve fill baseline lower_color must be non-empty when provided.")
        if self.upper_color is not None and not str(self.upper_color).strip():
            raise ValueError("Curve fill baseline upper_color must be non-empty when provided.")
        if self.line_color is not None and not str(self.line_color).strip():
            raise ValueError("Curve fill baseline line_color must be non-empty when provided.")
        if self.line_width <= 0:
            raise ValueError("Curve fill baseline line_width must be positive.")
        if not str(self.line_style).strip():
            raise ValueError("Curve fill baseline line_style must be non-empty.")


@dataclass(slots=True)
class CurveFillSpec:
    kind: CurveFillKind
    other_channel: str | None = None
    other_element_id: str | None = None
    baseline: CurveFillBaselineSpec | None = None
    label: str | None = None
    color: str | None = None
    alpha: float | None = None
    crossover: CurveFillCrossoverSpec = field(default_factory=CurveFillCrossoverSpec)

    def __post_init__(self) -> None:
        if self.other_channel is not None:
            self.other_channel = str(self.other_channel).strip()
            if not self.other_channel:
                raise ValueError("Curve fill other_channel must be non-empty when provided.")
        if self.other_element_id is not None:
            self.other_element_id = str(self.other_element_id).strip()
            if not self.other_element_id:
                raise ValueError("Curve fill other_element_id must be non-empty when provided.")
        if self.kind == CurveFillKind.BETWEEN_CURVES:
            if self.other_channel is None:
                raise ValueError("Curve fill between_curves requires other_channel.")
            if self.other_element_id is not None:
                raise ValueError("Curve fill between_curves does not accept other_element_id.")
            if self.baseline is not None:
                raise ValueError("Curve fill between_curves does not accept baseline.")
        if self.kind == CurveFillKind.BETWEEN_INSTANCES:
            if self.other_element_id is None:
                raise ValueError("Curve fill between_instances requires other_element_id.")
            if self.other_channel is not None:
                raise ValueError("Curve fill between_instances does not accept other_channel.")
            if self.baseline is not None:
                raise ValueError("Curve fill between_instances does not accept baseline.")
        if self.kind in {CurveFillKind.TO_LOWER_LIMIT, CurveFillKind.TO_UPPER_LIMIT}:
            if self.other_channel is not None or self.other_element_id is not None:
                raise ValueError("Curve limit fills do not accept other targets.")
            if self.baseline is not None:
                raise ValueError("Curve limit fills do not accept baseline.")
        if self.kind == CurveFillKind.BASELINE_SPLIT:
            if self.baseline is None:
                raise ValueError("Curve fill baseline_split requires baseline.")
            if self.other_channel is not None or self.other_element_id is not None:
                raise ValueError("Curve fill baseline_split does not accept other targets.")
        if self.label is not None and not str(self.label).strip():
            raise ValueError("Curve fill label must be non-empty when provided.")
        if self.color is not None and not str(self.color).strip():
            raise ValueError("Curve fill color must be non-empty when provided.")
        if self.alpha is not None and (self.alpha < 0 or self.alpha > 1):
            raise ValueError("Curve fill alpha must be between 0 and 1.")
        if (
            self.kind
            not in {CurveFillKind.BETWEEN_CURVES, CurveFillKind.BETWEEN_INSTANCES}
            and self.crossover.enabled
        ):
            raise ValueError("Curve fill crossover is only supported for between-curve fills.")


@dataclass(slots=True)
class CurveElement:
    channel: str
    id: str | None = None
    label: str | None = None
    style: StyleSpec = field(default_factory=StyleSpec)
    scale: ScaleSpec | None = None
    reference_overlay: ReferenceCurveOverlaySpec | None = None
    wrap: bool = False
    wrap_color: str | None = None
    render_mode: str = "line"
    value_labels: CurveValueLabelsSpec = field(default_factory=CurveValueLabelsSpec)
    header_display: CurveHeaderDisplaySpec = field(default_factory=CurveHeaderDisplaySpec)
    callouts: tuple[CurveCalloutSpec, ...] = ()
    fill: CurveFillSpec | None = None

    def __post_init__(self) -> None:
        if self.id is not None and not str(self.id).strip():
            raise ValueError("Curve id must be non-empty when provided.")
        if not isinstance(self.wrap, bool):
            raise ValueError("Curve wrap must be boolean.")
        if self.wrap_color is not None and not str(self.wrap_color).strip():
            raise ValueError("Curve wrap_color must be non-empty when provided.")
        mode = self.render_mode.strip().lower()
        if mode not in {"line", "value_labels"}:
            raise ValueError("Curve render_mode must be line or value_labels.")
        self.render_mode = mode


@dataclass(slots=True)
class RasterWaveformSpec:
    enabled: bool = False
    stride: int = 1
    amplitude_scale: float = 0.35
    color: str = "#5b3f8c"
    line_width: float = 0.3
    fill: bool = True
    positive_fill_color: str = "#000000"
    negative_fill_color: str = "#ffffff"
    invert_fill_polarity: bool = False
    max_traces: int | None = None

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("Raster waveform stride must be positive.")
        if self.amplitude_scale <= 0:
            raise ValueError("Raster waveform amplitude_scale must be positive.")
        if self.line_width <= 0:
            raise ValueError("Raster waveform line_width must be positive.")
        if self.max_traces is not None and self.max_traces <= 0:
            raise ValueError("Raster waveform max_traces must be positive.")
        if not str(self.color).strip():
            raise ValueError("Raster waveform color must be non-empty.")
        if not str(self.positive_fill_color).strip():
            raise ValueError("Raster waveform positive_fill_color must be non-empty.")
        if not str(self.negative_fill_color).strip():
            raise ValueError("Raster waveform negative_fill_color must be non-empty.")


@dataclass(slots=True)
class RasterElement:
    channel: str
    label: str | None = None
    style: StyleSpec = field(default_factory=StyleSpec)
    profile: RasterProfileKind = RasterProfileKind.GENERIC
    normalization: RasterNormalizationKind = RasterNormalizationKind.AUTO
    waveform_normalization: RasterNormalizationKind = RasterNormalizationKind.AUTO
    clip_percentiles: tuple[float, float] | None = None
    interpolation: str = "nearest"
    show_raster: bool = True
    raster_alpha: float = 1.0
    color_limits: tuple[float, float] | None = None
    colorbar_enabled: bool = False
    colorbar_label: str | None = None
    colorbar_position: RasterColorbarPosition = RasterColorbarPosition.RIGHT
    sample_axis_enabled: bool = False
    sample_axis_label: str | None = None
    sample_axis_unit: str | None = None
    sample_axis_source_origin: float | None = None
    sample_axis_source_step: float | None = None
    sample_axis_min: float | None = None
    sample_axis_max: float | None = None
    sample_axis_tick_count: int = 5
    waveform: RasterWaveformSpec = field(default_factory=RasterWaveformSpec)

    def __post_init__(self) -> None:
        if self.label is not None and not str(self.label).strip():
            raise ValueError("Raster label must be non-empty when provided.")
        if self.clip_percentiles is not None:
            low, high = self.clip_percentiles
            if low < 0 or low > 100 or high < 0 or high > 100:
                raise ValueError("Raster clip_percentiles must be between 0 and 100.")
            if low >= high:
                raise ValueError("Raster clip_percentiles must be increasing.")
        if self.raster_alpha < 0 or self.raster_alpha > 1:
            raise ValueError("Raster raster_alpha must be between 0 and 1.")
        if self.colorbar_label is not None and not str(self.colorbar_label).strip():
            raise ValueError("Raster colorbar_label must be non-empty when provided.")
        if self.sample_axis_label is not None and not str(self.sample_axis_label).strip():
            raise ValueError("Raster sample_axis_label must be non-empty when provided.")
        if self.sample_axis_unit is not None and not str(self.sample_axis_unit).strip():
            raise ValueError("Raster sample_axis_unit must be non-empty when provided.")
        has_source_origin = self.sample_axis_source_origin is not None
        has_source_step = self.sample_axis_source_step is not None
        if has_source_origin != has_source_step:
            raise ValueError(
                "Raster sample_axis_source_origin and sample_axis_source_step must be set together."
            )
        if has_source_step and np.isclose(float(self.sample_axis_source_step), 0.0):
            raise ValueError("Raster sample_axis_source_step must be non-zero.")
        has_min = self.sample_axis_min is not None
        has_max = self.sample_axis_max is not None
        if has_min != has_max:
            raise ValueError("Raster sample_axis_min and sample_axis_max must be set together.")
        if has_min and has_max and float(self.sample_axis_min) == float(self.sample_axis_max):
            raise ValueError("Raster sample_axis_min and sample_axis_max must differ.")
        if self.sample_axis_tick_count < 2:
            raise ValueError("Raster sample_axis_tick_count must be at least 2.")


TrackElement = CurveElement | RasterElement


@dataclass(slots=True)
class HeaderField:
    label: str
    source_key: str
    default: str = ""


@dataclass(slots=True)
class HeaderSpec:
    title: str | None = None
    subtitle: str | None = None
    fields: tuple[HeaderField, ...] = ()


@dataclass(slots=True)
class FooterSpec:
    lines: tuple[str, ...] = ()


@dataclass(slots=True)
class MarkerSpec:
    depth: float
    label: str
    color: str = "#666666"
    line_style: str = "--"


@dataclass(slots=True)
class ZoneSpec:
    top: float
    base: float
    label: str
    fill_color: str = "#d9d9d9"
    alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.base <= self.top:
            raise ValueError("Zone base must be greater than top.")


_PAGE_SIZES_MM = {
    "A4": (210.0, 297.0),
    "LETTER": (215.9, 279.4),
}


@dataclass(slots=True)
class PageSpec:
    width_mm: float
    height_mm: float
    continuous: bool = False
    margin_left_mm: float = 0.0
    margin_right_mm: float = 10.0
    margin_top_mm: float = 10.0
    margin_bottom_mm: float = 10.0
    header_height_mm: float = 18.0
    track_header_height_mm: float = 8.0
    footer_height_mm: float = 10.0
    track_gap_mm: float = 0.0

    @classmethod
    def from_name(cls, name: str, orientation: str = "portrait", **kwargs: Any) -> PageSpec:
        size_name = name.strip().upper()
        if size_name not in _PAGE_SIZES_MM:
            raise TemplateValidationError(f"Unsupported page size {name!r}.")
        width_mm, height_mm = _PAGE_SIZES_MM[size_name]
        if orientation.strip().lower() == "landscape":
            width_mm, height_mm = height_mm, width_mm
        return cls(width_mm=width_mm, height_mm=height_mm, **kwargs)

    @property
    def usable_width_mm(self) -> float:
        return self.width_mm - self.margin_left_mm - self.margin_right_mm

    @property
    def plot_top_mm(self) -> float:
        return self.margin_top_mm + self.header_height_mm + self.track_header_height_mm

    @property
    def plot_height_mm(self) -> float:
        return (
            self.height_mm
            - self.margin_top_mm
            - self.margin_bottom_mm
            - self.header_height_mm
            - self.track_header_height_mm
            - self.footer_height_mm
        )


@dataclass(slots=True)
class DepthAxisSpec:
    unit: str = "m"
    scale_ratio: int = 200
    major_step: float = 10.0
    minor_step: float = 2.0

    def __post_init__(self) -> None:
        if self.scale_ratio <= 0:
            raise ValueError("Depth scale ratio must be positive.")
        if self.major_step <= 0 or self.minor_step <= 0:
            raise ValueError("Depth tick steps must be positive.")


@dataclass(slots=True)
class TrackSpec:
    id: str
    title: str
    kind: TrackKind
    width_mm: float
    elements: tuple[TrackElement, ...] = ()
    x_scale: ScaleSpec | None = None
    header: TrackHeaderSpec = field(default_factory=TrackHeaderSpec)
    grid: GridSpec = field(default_factory=GridSpec)
    reference: ReferenceTrackSpec | None = None

    def __post_init__(self) -> None:
        if self.width_mm <= 0:
            raise ValueError(f"Track {self.id} width must be positive.")
        element_ids = [
            element.id
            for element in self.elements
            if isinstance(element, CurveElement) and element.id is not None
        ]
        if len(set(element_ids)) != len(element_ids):
            raise ValueError(f"Track {self.id} contains duplicate curve ids.")
        if self.kind == TrackKind.REFERENCE:
            if self.reference is None:
                self.reference = ReferenceTrackSpec()
            invalid = [element for element in self.elements if isinstance(element, RasterElement)]
            if invalid:
                raise ValueError(
                    f"Reference track {self.id} cannot contain raster elements. "
                    "Use an array track instead."
                )
        if self.kind == TrackKind.NORMAL:
            invalid = [element for element in self.elements if isinstance(element, RasterElement)]
            if invalid:
                raise ValueError(
                    f"Normal track {self.id} cannot contain raster elements. "
                    "Use an array track instead."
                )
            invalid_reference_overlays = [
                element
                for element in self.elements
                if isinstance(element, CurveElement) and element.reference_overlay is not None
            ]
            if invalid_reference_overlays:
                raise ValueError(
                    f"Normal track {self.id} cannot use reference curve overlays."
                )
        if self.kind == TrackKind.ARRAY:
            invalid_reference_overlays = [
                element
                for element in self.elements
                if isinstance(element, CurveElement) and element.reference_overlay is not None
            ]
            if invalid_reference_overlays:
                raise ValueError(
                    f"Array track {self.id} cannot use reference curve overlays."
                )
        if self.kind == TrackKind.ANNOTATION and self.elements:
            raise ValueError(
                f"Annotation track {self.id} currently does not accept curve/raster elements."
            )


@dataclass(slots=True)
class LogDocument:
    name: str
    page: PageSpec
    depth_axis: DepthAxisSpec
    tracks: tuple[TrackSpec, ...]
    depth_range: tuple[float, float] | None = None
    header: HeaderSpec = field(default_factory=HeaderSpec)
    footer: FooterSpec = field(default_factory=FooterSpec)
    markers: tuple[MarkerSpec, ...] = ()
    zones: tuple[ZoneSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracks:
            raise ValueError("A log document must contain at least one track.")

    def resolve_depth_range(
        self,
        dataset: WellDataset,
        registry: SimpleUnitRegistry = DEFAULT_UNITS,
    ) -> tuple[float, float]:
        if self.depth_range is not None:
            top, base = self.depth_range
        else:
            top, base = dataset.depth_range(self.depth_axis.unit, registry)
        return min(top, base), max(top, base)
