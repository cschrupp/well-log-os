from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .errors import LayoutError
from .model import LogDocument, PageSpec, TrackSpec, WellDataset
from .units import DEFAULT_UNITS, SimpleUnitRegistry


@dataclass(slots=True, frozen=True)
class DepthWindow:
    page_number: int
    start: float
    stop: float
    unit: str


@dataclass(slots=True, frozen=True)
class Frame:
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float


@dataclass(slots=True, frozen=True)
class TrackFrame:
    track: TrackSpec
    frame: Frame


@dataclass(slots=True, frozen=True)
class PageLayout:
    page_number: int
    page: PageSpec
    depth_window: DepthWindow
    header_frame: Frame
    track_header_top_frames: tuple[TrackFrame, ...]
    track_header_bottom_frames: tuple[TrackFrame, ...]
    footer_frame: Frame
    track_frames: tuple[TrackFrame, ...]

    @property
    def track_header_frames(self) -> tuple[TrackFrame, ...]:
        # Backward compatibility: top header frames keep the original name.
        return self.track_header_top_frames


class LayoutEngine:
    def __init__(self, registry: SimpleUnitRegistry = DEFAULT_UNITS) -> None:
        self.registry = registry

    def depth_units_per_mm(self, document: LogDocument) -> float:
        return self.registry.convert(
            float(document.depth_axis.scale_ratio),
            "mm",
            document.depth_axis.unit,
        )

    def _plot_geometry(
        self,
        page: PageSpec,
        *,
        reserve_top_track_header: bool,
        reserve_bottom_track_header: bool,
    ) -> tuple[float, float]:
        top_track_header = page.track_header_height_mm if reserve_top_track_header else 0.0
        bottom_track_header = page.track_header_height_mm if reserve_bottom_track_header else 0.0
        y_origin = page.margin_top_mm + page.header_height_mm + top_track_header
        plot_height = (
            page.height_mm
            - page.margin_top_mm
            - page.margin_bottom_mm
            - page.header_height_mm
            - page.footer_height_mm
            - top_track_header
            - bottom_track_header
        )
        if plot_height <= 0:
            raise LayoutError("Computed plot height must be positive.")
        return y_origin, plot_height

    def _depth_span_for_page(
        self,
        document: LogDocument,
        page: PageSpec,
        *,
        reserve_top_track_header: bool,
        reserve_bottom_track_header: bool,
    ) -> float:
        _, plot_height = self._plot_geometry(
            page,
            reserve_top_track_header=reserve_top_track_header,
            reserve_bottom_track_header=reserve_bottom_track_header,
        )
        return self.depth_units_per_mm(document) * plot_height

    def depth_span_per_page(self, document: LogDocument) -> float:
        # Default to the first-page geometry (top track header only).
        return self._depth_span_for_page(
            document,
            document.page,
            reserve_top_track_header=True,
            reserve_bottom_track_header=False,
        )

    def paginate(self, document: LogDocument, dataset: WellDataset) -> tuple[DepthWindow, ...]:
        start, stop = document.resolve_depth_range(dataset, self.registry)
        total_span = stop - start
        page = document.page

        span_single = self._depth_span_for_page(
            document,
            page,
            reserve_top_track_header=True,
            reserve_bottom_track_header=True,
        )
        span_first = self._depth_span_for_page(
            document,
            page,
            reserve_top_track_header=True,
            reserve_bottom_track_header=False,
        )
        span_middle = self._depth_span_for_page(
            document,
            page,
            reserve_top_track_header=False,
            reserve_bottom_track_header=False,
        )
        span_last = self._depth_span_for_page(
            document,
            page,
            reserve_top_track_header=False,
            reserve_bottom_track_header=True,
        )
        if min(span_single, span_first, span_middle, span_last) <= 0:
            raise LayoutError("Depth span per page must be positive.")

        if total_span <= span_single:
            return (
                DepthWindow(
                    page_number=1,
                    start=start,
                    stop=start + span_single,
                    unit=document.depth_axis.unit,
                ),
            )

        windows = []
        cursor = start
        page_number = 1

        # First page reserves top track headers.
        first_stop = min(cursor + span_first, stop)
        windows.append(
            DepthWindow(
                page_number=page_number,
                start=cursor,
                stop=first_stop,
                unit=document.depth_axis.unit,
            )
        )
        cursor = first_stop
        page_number += 1

        remaining = stop - cursor
        if remaining > 0:
            middle_pages = 0
            if remaining > span_last:
                middle_pages = int(math.ceil((remaining - span_last) / span_middle))

            for _ in range(middle_pages):
                middle_stop = min(cursor + span_middle, stop)
                windows.append(
                    DepthWindow(
                        page_number=page_number,
                        start=cursor,
                        stop=middle_stop,
                        unit=document.depth_axis.unit,
                    )
                )
                cursor = middle_stop
                page_number += 1

            windows.append(
                DepthWindow(
                    page_number=page_number,
                    start=cursor,
                    stop=stop,
                    unit=document.depth_axis.unit,
                )
            )
        normalized: list[DepthWindow] = []
        for index, window in enumerate(windows):
            if len(windows) == 1:
                span = span_single
            elif index == 0:
                span = span_first
            elif index == len(windows) - 1:
                span = span_last
            else:
                span = span_middle
            normalized.append(
                DepthWindow(
                    page_number=window.page_number,
                    start=window.start,
                    stop=window.start + span,
                    unit=window.unit,
                )
            )
        return tuple(normalized)

    def track_frames(self, document: LogDocument) -> tuple[TrackFrame, ...]:
        return self._track_frames_for_page(
            document,
            document.page,
            reserve_top_track_header=True,
            reserve_bottom_track_header=False,
        )

    def _track_frames_for_page(
        self,
        document: LogDocument,
        page: PageSpec,
        *,
        reserve_top_track_header: bool,
        reserve_bottom_track_header: bool,
    ) -> tuple[TrackFrame, ...]:
        y_origin, plot_height = self._plot_geometry(
            page,
            reserve_top_track_header=reserve_top_track_header,
            reserve_bottom_track_header=reserve_bottom_track_header,
        )
        total_width = sum(track.width_mm for track in document.tracks)
        total_gaps = max(len(document.tracks) - 1, 0) * page.track_gap_mm
        required_width = total_width + total_gaps
        if required_width > page.usable_width_mm + 1e-9:
            raise LayoutError(
                "Tracks require "
                f"{required_width:.1f} mm but only {page.usable_width_mm:.1f} mm "
                "are available."
            )

        x_cursor = page.margin_left_mm
        frames = []
        for track in document.tracks:
            frame = Frame(
                x_mm=x_cursor,
                y_mm=y_origin,
                width_mm=track.width_mm,
                height_mm=plot_height,
            )
            frames.append(TrackFrame(track=track, frame=frame))
            x_cursor += track.width_mm + page.track_gap_mm
        return tuple(frames)

    def _page_with_plot_height(
        self,
        page: PageSpec,
        plot_height_mm: float,
        *,
        reserve_top_track_header: bool,
        reserve_bottom_track_header: bool,
    ) -> PageSpec:
        if plot_height_mm <= 0:
            raise LayoutError("Continuous mode requires a positive depth span.")
        top_track_header = page.track_header_height_mm if reserve_top_track_header else 0.0
        bottom_track_header = page.track_header_height_mm if reserve_bottom_track_header else 0.0
        height_mm = (
            page.margin_top_mm
            + page.header_height_mm
            + top_track_header
            + plot_height_mm
            + bottom_track_header
            + page.footer_height_mm
            + page.margin_bottom_mm
        )
        return replace(page, height_mm=height_mm)

    def _track_header_top_frames(
        self, page: PageSpec, track_frames: tuple[TrackFrame, ...]
    ) -> tuple[TrackFrame, ...]:
        if page.track_header_height_mm <= 0:
            return ()
        return tuple(
            TrackFrame(
                track=track_frame.track,
                frame=Frame(
                    x_mm=track_frame.frame.x_mm,
                    y_mm=page.margin_top_mm + page.header_height_mm,
                    width_mm=track_frame.frame.width_mm,
                    height_mm=page.track_header_height_mm,
                ),
            )
            for track_frame in track_frames
        )

    def _track_header_bottom_frames(
        self, page: PageSpec, track_frames: tuple[TrackFrame, ...]
    ) -> tuple[TrackFrame, ...]:
        if page.track_header_height_mm <= 0:
            return ()
        y_mm = (
            page.height_mm
            - page.margin_bottom_mm
            - page.footer_height_mm
            - page.track_header_height_mm
        )
        return tuple(
            TrackFrame(
                track=track_frame.track,
                frame=Frame(
                    x_mm=track_frame.frame.x_mm,
                    y_mm=y_mm,
                    width_mm=track_frame.frame.width_mm,
                    height_mm=page.track_header_height_mm,
                ),
            )
            for track_frame in track_frames
        )

    def layout(self, document: LogDocument, dataset: WellDataset) -> tuple[PageLayout, ...]:
        start, stop = document.resolve_depth_range(dataset, self.registry)
        page = document.page

        if page.continuous:
            units_per_mm = self.depth_units_per_mm(document)
            depth_span = stop - start
            if units_per_mm <= 0:
                raise LayoutError("Depth units per mm must be positive.")
            page = self._page_with_plot_height(
                page,
                depth_span / units_per_mm,
                reserve_top_track_header=True,
                reserve_bottom_track_header=True,
            )
            windows = (
                DepthWindow(
                    page_number=1,
                    start=start,
                    stop=stop,
                    unit=document.depth_axis.unit,
                ),
            )
        else:
            windows = self.paginate(document, dataset)

        layouts: list[PageLayout] = []
        for index, window in enumerate(windows):
            reserve_top_track_header = index == 0 and page.track_header_height_mm > 0
            reserve_bottom_track_header = (
                index == len(windows) - 1 and page.track_header_height_mm > 0
            )
            track_frames = self._track_frames_for_page(
                document,
                page,
                reserve_top_track_header=reserve_top_track_header,
                reserve_bottom_track_header=reserve_bottom_track_header,
            )
            track_header_top_frames = (
                self._track_header_top_frames(page, track_frames)
                if reserve_top_track_header
                else ()
            )
            track_header_bottom_frames = (
                self._track_header_bottom_frames(page, track_frames)
                if reserve_bottom_track_header
                else ()
            )
            header_frame = Frame(
                x_mm=page.margin_left_mm,
                y_mm=page.margin_top_mm,
                width_mm=page.usable_width_mm,
                height_mm=page.header_height_mm,
            )
            footer_frame = Frame(
                x_mm=page.margin_left_mm,
                y_mm=page.height_mm - page.margin_bottom_mm - page.footer_height_mm,
                width_mm=page.usable_width_mm,
                height_mm=page.footer_height_mm,
            )
            layouts.append(
                PageLayout(
                    page_number=window.page_number,
                    page=page,
                    depth_window=window,
                    header_frame=header_frame,
                    track_header_top_frames=track_header_top_frames,
                    track_header_bottom_frames=track_header_bottom_frames,
                    footer_frame=footer_frame,
                    track_frames=track_frames,
                )
            )
        return tuple(layouts)
