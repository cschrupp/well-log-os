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
                    if curve_scale is not None and curve_scale.kind == ScaleKind.TANGENTIAL:
                        x_values = self._transform_tangential_values(x_values, curve_scale)
                    figure.add_trace(
                        go.Scattergl(
                            x=x_values,
                            y=channel.depth_in(document.depth_axis.unit, self.registry),
                            mode="lines",
                            line={
                                "color": element.style.color,
                                "width": element.style.line_width,
                            },
                            name=element.label or element.channel,
                            showlegend=column == 1,
                        ),
                        row=1,
                        col=column,
                    )
                    if element.scale is not None:
                        self._update_xaxis(figure, element.scale, row=1, col=column)
                elif isinstance(element, RasterElement):
                    channel = dataset.get_channel(element.channel)
                    if not isinstance(channel, RasterChannel):
                        raise TypeError(
                            f"Raster element {element.channel} requires a raster channel."
                        )
                    figure.add_trace(
                        go.Heatmap(
                            z=channel.values.T,
                            x=channel.sample_axis,
                            y=channel.depth_in(document.depth_axis.unit, self.registry),
                            colorscale=element.style.colormap,
                            showscale=False,
                            name=element.channel,
                        ),
                        row=1,
                        col=column,
                    )
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
