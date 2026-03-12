from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..errors import DependencyUnavailableError
from ..layout import LayoutEngine
from ..model import (
    CurveElement,
    LogDocument,
    RasterChannel,
    RasterElement,
    RasterNormalizationKind,
    RasterProfileKind,
    ScalarChannel,
    ScaleKind,
    TrackKind,
    WellDataset,
)
from ..units import DEFAULT_UNITS, SimpleUnitRegistry
from .base import Renderer, RenderResult


class PlotlyRenderer(Renderer):
    def __init__(self, registry: SimpleUnitRegistry = DEFAULT_UNITS) -> None:
        self.registry = registry
        self.layout_engine = LayoutEngine(registry)

    def render(
        self,
        document: LogDocument,
        dataset: WellDataset,
        *,
        output_path: str | Path | None = None,
    ) -> RenderResult:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Plotly is required for interactive rendering. Install well-log-os[interactive]."
            ) from exc

        layouts = self.layout_engine.layout(document, dataset)
        page_layout = layouts[0]
        widths = [frame.track.width_mm for frame in page_layout.track_frames]
        figure = make_subplots(
            rows=1,
            cols=len(page_layout.track_frames),
            shared_yaxes=True,
            horizontal_spacing=0.02,
            column_widths=[width / sum(widths) for width in widths],
            subplot_titles=[frame.track.title for frame in page_layout.track_frames],
        )

        for column, track_frame in enumerate(page_layout.track_frames, start=1):
            track = track_frame.track
            if track.kind == TrackKind.REFERENCE:
                continue
            for element in track.elements:
                if isinstance(element, CurveElement):
                    channel = dataset.get_channel(element.channel)
                    if not isinstance(channel, ScalarChannel):
                        raise TypeError(
                            f"Curve element {element.channel} requires a scalar channel."
                        )
                    curve_scale = element.scale or track.x_scale
                    x_values = channel.masked_values()
                    y_values = channel.depth_in(document.depth_axis.unit, self.registry)
                    valid_mask = np.isfinite(x_values)
                    wrapped_mask = np.zeros(x_values.shape, dtype=bool)

                    if curve_scale is not None and element.wrap:
                        x_values, valid_mask, wrapped_mask = self._transform_wrap_values(
                            x_values,
                            curve_scale,
                        )
                    elif curve_scale is not None and curve_scale.kind == ScaleKind.LOG:
                        valid_mask &= x_values > 0

                    if curve_scale is not None and curve_scale.kind == ScaleKind.TANGENTIAL:
                        x_values = self._transform_tangential_values(x_values, curve_scale)

                    self._add_curve_trace(
                        figure,
                        row=1,
                        col=column,
                        x_values=x_values,
                        y_values=y_values,
                        mask=valid_mask & ~wrapped_mask,
                        color=element.style.color,
                        width=element.style.line_width,
                        name=element.label or element.channel,
                        showlegend=column == 1,
                    )
                    if np.any(wrapped_mask):
                        self._add_curve_trace(
                            figure,
                            row=1,
                            col=column,
                            x_values=x_values,
                            y_values=y_values,
                            mask=valid_mask & wrapped_mask,
                            color=element.wrap_color or element.style.color,
                            width=element.style.line_width,
                            name=element.label or element.channel,
                            showlegend=False,
                        )
                    if element.scale is not None:
                        self._update_xaxis(figure, element.scale, row=1, col=column)
                elif isinstance(element, RasterElement):
                    channel = dataset.get_channel(element.channel)
                    if not isinstance(channel, RasterChannel):
                        raise TypeError(
                            f"Raster element {element.channel} requires a raster channel."
                        )
                    axis_min, axis_max, axis_unit = self._raster_axis_limits(
                        track,
                        element,
                        channel,
                    )
                    sample_axis = np.linspace(axis_min, axis_max, channel.values.shape[1])
                    raster_values = self._normalize_raster_values(channel.values, element)
                    heatmap_kwargs = {
                        "z": raster_values.T,
                        "x": sample_axis,
                        "y": channel.depth_in(document.depth_axis.unit, self.registry),
                        "colorscale": element.style.colormap,
                        "showscale": element.colorbar_enabled,
                        "name": element.label or element.channel,
                    }
                    limits = self._resolve_raster_color_limits(raster_values, element)
                    if limits is not None:
                        heatmap_kwargs["zmin"] = float(limits[0])
                        heatmap_kwargs["zmax"] = float(limits[1])
                    if element.colorbar_enabled:
                        colorbar_title = element.colorbar_label or channel.value_unit or ""
                        if colorbar_title:
                            heatmap_kwargs["colorbar"] = {"title": {"text": colorbar_title}}
                    figure.add_trace(
                        go.Heatmap(**heatmap_kwargs),
                        row=1,
                        col=column,
                    )
                    if element.sample_axis_enabled:
                        axis_title = element.sample_axis_label
                        if not axis_title:
                            axis_title = channel.sample_label
                            if axis_unit:
                                axis_title = f"{axis_title} ({axis_unit})"
                        figure.update_xaxes(title_text=axis_title, row=1, col=column)
            if track.x_scale is not None:
                self._update_xaxis(figure, track.x_scale, row=1, col=column)

        figure.update_yaxes(autorange="reversed")
        figure.update_layout(title=document.name, template="plotly_white")

        output = Path(output_path) if output_path is not None else None
        if output is not None:
            if output.suffix.lower() == ".html":
                figure.write_html(str(output))
            elif output.suffix.lower() == ".json":
                output.write_text(figure.to_json(), encoding="utf-8")
            else:
                raise ValueError("Plotly output_path must end with .html or .json.")
        return RenderResult(
            backend="plotly",
            page_count=len(layouts),
            artifact=figure,
            output_path=output,
        )

    def _update_xaxis(self, figure, scale, *, row: int, col: int) -> None:
        if scale.kind == ScaleKind.LOG:
            lower = math.log10(scale.minimum)
            upper = math.log10(scale.maximum)
            axis_range = [upper, lower] if scale.reverse else [lower, upper]
            figure.update_xaxes(type="log", range=axis_range, row=row, col=col)
            return
        if scale.kind == ScaleKind.TANGENTIAL:
            axis_range = [1.0, 0.0] if scale.reverse else [0.0, 1.0]
            figure.update_xaxes(type="linear", range=axis_range, row=row, col=col)
            return

        axis_range = (
            [scale.maximum, scale.minimum] if scale.reverse else [scale.minimum, scale.maximum]
        )
        figure.update_xaxes(type="linear", range=axis_range, row=row, col=col)

    def _transform_tangential_values(self, values, scale):
        spread = 1.2
        denominator = np.tan(0.5 * spread)
        unit = (values - scale.minimum) / (scale.maximum - scale.minimum)
        transformed = 0.5 + np.tan((unit - 0.5) * spread) / (2.0 * denominator)
        return np.clip(transformed, 0.0, 1.0)

    def _transform_wrap_values(self, values, scale):
        transformed = np.array(values, dtype=float, copy=True)
        valid_mask = np.isfinite(transformed)
        wrapped_mask = np.zeros(transformed.shape, dtype=bool)

        lower = min(scale.minimum, scale.maximum)
        upper = max(scale.minimum, scale.maximum)
        if np.isclose(lower, upper):
            return transformed, valid_mask & False, wrapped_mask

        if scale.kind == ScaleKind.LOG:
            valid_mask &= transformed > 0
            if not np.any(valid_mask):
                return transformed, valid_mask, wrapped_mask
            if lower <= 0 or upper <= 0:
                return transformed, valid_mask & False, wrapped_mask
            outside = valid_mask & ((transformed < lower) | (transformed > upper))
            wrapped_mask = outside
            if np.any(outside):
                low = float(np.log(lower))
                high = float(np.log(upper))
                period = high - low
                if np.isclose(period, 0.0):
                    return transformed, valid_mask & False, wrapped_mask & False
                outside_log = np.log(transformed[outside])
                wrapped_log = np.mod(outside_log - low, period) + low
                transformed[outside] = np.exp(wrapped_log)
            return transformed, valid_mask, wrapped_mask

        outside = valid_mask & ((transformed < lower) | (transformed > upper))
        wrapped_mask = outside
        if np.any(outside):
            period = upper - lower
            transformed[outside] = np.mod(transformed[outside] - lower, period) + lower
        return transformed, valid_mask, wrapped_mask

    def _add_curve_trace(
        self,
        figure,
        *,
        row: int,
        col: int,
        x_values,
        y_values,
        mask,
        color: str,
        width: float,
        name: str,
        showlegend: bool,
    ) -> None:
        if not np.any(mask):
            return
        import plotly.graph_objects as go

        masked_x = np.where(mask, x_values, np.nan)
        masked_y = np.where(mask, y_values, np.nan)
        figure.add_trace(
            go.Scattergl(
                x=masked_x,
                y=masked_y,
                mode="lines",
                line={"color": color, "width": width},
                name=name,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    def _raster_axis_limits(
        self,
        track,
        element: RasterElement,
        channel: RasterChannel,
    ) -> tuple[float, float, str | None]:
        if element.sample_axis_min is not None and element.sample_axis_max is not None:
            axis_min = float(element.sample_axis_min)
            axis_max = float(element.sample_axis_max)
        elif track.x_scale is not None:
            axis_min = float(track.x_scale.minimum)
            axis_max = float(track.x_scale.maximum)
        else:
            axis_min = float(channel.sample_axis[0])
            axis_max = float(channel.sample_axis[-1])
        unit_text = element.sample_axis_unit or channel.sample_unit or None
        return axis_min, axis_max, unit_text

    def _resolve_raster_normalization(self, element: RasterElement) -> RasterNormalizationKind:
        if element.normalization != RasterNormalizationKind.AUTO:
            return element.normalization
        if element.profile == RasterProfileKind.VDL:
            return RasterNormalizationKind.TRACE_MAXABS
        return RasterNormalizationKind.NONE

    def _normalize_raster_values(self, values: np.ndarray, element: RasterElement) -> np.ndarray:
        normalized = np.asarray(values, dtype=float)
        mode = self._resolve_raster_normalization(element)
        if mode == RasterNormalizationKind.NONE:
            return normalized
        if mode == RasterNormalizationKind.GLOBAL_MAXABS:
            denominator = float(np.nanmax(np.abs(normalized)))
            if np.isfinite(denominator) and not np.isclose(denominator, 0.0):
                return normalized / denominator
            return normalized
        denominators = np.nanmax(np.abs(normalized), axis=1, keepdims=True)
        valid = np.isfinite(denominators) & ~np.isclose(denominators, 0.0)
        safe_denominators = np.where(valid, denominators, 1.0)
        return normalized / safe_denominators

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
        if element.clip_percentiles is not None:
            low, high = element.clip_percentiles
            lower, upper = np.nanpercentile(finite, [low, high])
            if np.isclose(lower, upper):
                lower = float(np.nanmin(finite))
                upper = float(np.nanmax(finite))
            return float(lower), float(upper)
        if element.profile == RasterProfileKind.VDL:
            normalization = self._resolve_raster_normalization(element)
            if normalization in {
                RasterNormalizationKind.TRACE_MAXABS,
                RasterNormalizationKind.GLOBAL_MAXABS,
            }:
                return -1.0, 1.0
            lower, upper = np.nanpercentile(finite, [2.0, 98.0])
            if np.isclose(lower, upper):
                lower = float(np.nanmin(finite))
                upper = float(np.nanmax(finite))
            return float(lower), float(upper)
        return None
