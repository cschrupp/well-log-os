from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from well_log_os import (
    CurveElement,
    CurveHeaderDisplaySpec,
    GridScaleKind,
    RasterChannel,
    ScalarChannel,
    ScaleKind,
    ScaleSpec,
    StyleSpec,
    WellDataset,
    document_from_mapping,
)
from well_log_os.errors import TemplateValidationError
from well_log_os.layout import DepthWindow, LayoutEngine
from well_log_os.renderers.matplotlib import (
    DEFAULT_MPL_STYLE_PATH,
    MatplotlibRenderer,
    _load_default_mpl_style,
)


class MatplotlibStyleDefaultsTests(unittest.TestCase):
    def test_defaults_yaml_exists_and_loads(self) -> None:
        self.assertTrue(DEFAULT_MPL_STYLE_PATH.is_file())
        style = _load_default_mpl_style(DEFAULT_MPL_STYLE_PATH)
        self.assertEqual(style["track"]["frame_color"], "#2f2f2f")
        self.assertEqual(style["markers"]["callout_arrow_style"], "-|>")
        self.assertEqual(style["curve_callouts"]["right_text_x"], 0.82)
        self.assertEqual(style["curve_callouts"]["lane_count"], 3)
        self.assertEqual(style["curve_callouts"]["top_distance_steps"], 1.5)

    def test_renderer_uses_yaml_defaults(self) -> None:
        renderer = MatplotlibRenderer()
        self.assertEqual(renderer.style["track"]["frame_linewidth"], 0.8)
        self.assertEqual(renderer.style["grid"]["depth_major_alpha"], 0.9)
        self.assertEqual(renderer.style["track"]["reference_grid_mode"], "edge_ticks")
        self.assertEqual(renderer.style["track"]["reference_label_align"], "center")
        self.assertEqual(renderer.style["track"]["reference_label_x"], 0.5)
        self.assertEqual(renderer.style["track_header"]["paired_scale_text_offset_ratio"], 0.08)
        self.assertEqual(renderer.style["track_header"]["fill_hatch"], "")
        self.assertEqual(renderer.style["track_header"]["division_tick_count"], 5)
        self.assertEqual(renderer.style["track_header"]["title_align"], "left")
        self.assertEqual(renderer.style["track_header"]["title_x"], 0.03)
        self.assertTrue(renderer.style["section_title"]["enabled"])
        self.assertEqual(renderer.style["section_title"]["height_mm"], 6.0)
        self.assertEqual(renderer.style["raster"]["colorbar_width_ratio"], 0.06)
        self.assertEqual(renderer.style["raster"]["sample_axis_tick_labelsize"], 5.0)
        self.assertEqual(renderer.style["raster"]["header_colorbar_bar_height_ratio"], 0.26)
        self.assertEqual(renderer.style["curve_callouts"]["left_text_x"], 0.1)
        self.assertEqual(renderer.style["curve_callouts"]["edge_padding_px"], 8.0)
        self.assertEqual(renderer.style["curve_callouts"]["bottom_distance_steps"], 1.5)

    def test_renderer_style_override_deep_merges(self) -> None:
        renderer = MatplotlibRenderer(
            style={
                "track": {"frame_color": "#111111"},
                "track_header": {
                    "separator_linewidth": 0.5,
                    "title_align": "center",
                    "title_x": 0.5,
                },
            }
        )
        self.assertEqual(renderer.style["track"]["frame_color"], "#111111")
        self.assertEqual(renderer.style["track"]["frame_linewidth"], 0.8)
        self.assertEqual(renderer.style["track_header"]["separator_linewidth"], 0.5)
        self.assertEqual(renderer.style["track_header"]["background_color"], "#e8e8e8")
        self.assertEqual(renderer.style["track_header"]["title_align"], "center")
        self.assertEqual(renderer.style["track_header"]["title_x"], 0.5)

    def test_renderer_auto_adjusts_header_height_for_multicurve_legend(self) -> None:
        document = document_from_mapping(
            {
                "name": "multicurve",
                "page": {"size": "A4", "track_header_height_mm": 8.0},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "combo",
                        "title": "Combo",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {"kind": "curve", "channel": "A"},
                            {"kind": "curve", "channel": "B"},
                            {"kind": "curve", "channel": "C"},
                            {"kind": "curve", "channel": "D"},
                            {"kind": "curve", "channel": "E"},
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        adjusted = renderer._auto_adjust_track_header_height(document)
        self.assertAlmostEqual(adjusted.page.track_header_height_mm, 22.0)

    def test_renderer_reserves_section_title_band_in_track_header_height(self) -> None:
        document = document_from_mapping(
            {
                "name": "section title",
                "page": {"size": "A4", "track_header_height_mm": 8.0},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "combo",
                        "title": "Combo",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [{"kind": "curve", "channel": "A"}],
                    }
                ],
                "metadata": {
                    "layout_sections": {
                        "active_section": {
                            "id": "main",
                            "title": "Main Section",
                            "subtitle": "Optional Subtitle",
                        }
                    }
                },
            }
        )
        renderer = MatplotlibRenderer()
        adjusted = renderer._auto_adjust_track_header_height(document)
        self.assertAlmostEqual(adjusted.page.track_header_height_mm, 14.0)

    def test_curve_row_bounds_partition_slot(self) -> None:
        renderer = MatplotlibRenderer()
        rows = renderer._curve_row_bounds(0.9, 0.3, 3)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0][0], 0.9)
        self.assertAlmostEqual(rows[-1][1], 0.3)
        self.assertAlmostEqual(rows[0][1], rows[1][0])
        self.assertAlmostEqual(rows[1][1], rows[2][0])

    def test_curve_header_pair_slot_joins_legend_and_scale_slots(self) -> None:
        document = document_from_mapping(
            {
                "name": "pair slot",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "combo",
                        "title": "Combo",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {"kind": "curve", "channel": "A"},
                            {"kind": "curve", "channel": "B"},
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        track = document.tracks[0]
        slots = renderer._track_header_slots(track)
        pair = renderer._curve_header_pair_slot(track, slots)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair[0], 1)
        self.assertEqual(pair[1], 2)
        self.assertGreater(pair[2], pair[3])

    def test_curve_header_pairs_draw_full_scale_triplets_without_fill_rows(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve pair scale triplets",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "combo",
                        "title": "Combo",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "label": "Gamma Ray",
                                "scale": {"kind": "linear", "min": 0, "max": 150},
                                "style": {"color": "#17bf22"},
                            },
                            {
                                "kind": "curve",
                                "channel": "TT",
                                "label": "Transit Time",
                                "scale": {
                                    "kind": "linear",
                                    "min": 200,
                                    "max": 400,
                                    "reverse": True,
                                },
                                "style": {"color": "#1238ff"},
                            },
                        ],
                    }
                ],
            }
        )
        dataset = WellDataset(name="curve pair scale triplets")
        depth = np.array([1000.0, 1001.0], dtype=float)
        dataset.add_channel(ScalarChannel("GR", depth, "m", "gAPI", values=np.array([10.0, 20.0])))
        dataset.add_channel(ScalarChannel("TT", depth, "m", "us", values=np.array([250.0, 260.0])))

        renderer = MatplotlibRenderer()
        fig = plt.figure(figsize=(4, 2), dpi=100)
        try:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            renderer._draw_track_header_curve_pairs(
                ax,
                document.tracks[0],
                document,
                dataset,
                0.95,
                0.10,
            )
            text_values = [text.get_text() for text in ax.texts]
            self.assertIn("150", text_values)
            self.assertIn("gAPI", text_values)
            self.assertIn("400", text_values)
            self.assertIn("200", text_values)
            self.assertIn("us", text_values)
        finally:
            plt.close(fig)

    def test_curve_header_row_count_uses_document_wide_capacity(self) -> None:
        document = document_from_mapping(
            {
                "name": "uniform header rows",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "combo",
                        "title": "Combo",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {"kind": "curve", "channel": "A"},
                            {"kind": "curve", "channel": "B"},
                            {"kind": "curve", "channel": "C"},
                            {"kind": "curve", "channel": "D"},
                        ],
                    },
                    {
                        "id": "single",
                        "title": "Single",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [{"kind": "curve", "channel": "E"}],
                    },
                ],
            }
        )
        renderer = MatplotlibRenderer()
        self.assertEqual(renderer._document_curve_row_capacity(document), 4)
        self.assertEqual(renderer._header_property_group_capacity(document), 4)
        self.assertEqual(renderer._curve_header_row_count(document, document.tracks[0]), 4)
        self.assertEqual(renderer._curve_header_row_count(document, document.tracks[1]), 4)

    def test_fill_header_row_count_uses_document_wide_capacity(self) -> None:
        document = document_from_mapping(
            {
                "name": "uniform fill rows",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "filled",
                        "title": "Filled",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "B",
                                    "label": "Gas Effect",
                                },
                            },
                            {"kind": "curve", "channel": "B"},
                        ],
                    },
                    {
                        "id": "plain",
                        "title": "Plain",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [{"kind": "curve", "channel": "C"}],
                    },
                ],
            }
        )
        renderer = MatplotlibRenderer()
        self.assertEqual(renderer._document_fill_row_capacity(document), 1)
        self.assertEqual(renderer._fill_header_row_count(document, document.tracks[0]), 1)
        self.assertEqual(renderer._fill_header_row_count(document, document.tracks[1]), 1)

    def test_reference_overlay_curve_plot_data_uses_lane_fractions(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference overlay lane fractions",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "reference_overlay": {
                                    "mode": "indicator",
                                    "lane_start": 0.7,
                                    "lane_end": 0.9,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        dataset = WellDataset(name="reference overlay lane fractions")
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset.add_channel(
            ScalarChannel("A", depth, "m", "u", values=np.array([0.0, 50.0, 100.0], dtype=float))
        )
        plot_data = renderer._curve_plot_data(
            document.tracks[0],
            document.tracks[0].elements[0],
            document,
            dataset,
            independent_curve_scales=False,
        )
        self.assertTrue(plot_data.x_is_fractional)
        np.testing.assert_allclose(plot_data.plot_values, np.array([0.7, 0.8, 0.9]))

    def test_reference_overlay_ticks_draw_edge_markers(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference overlay ticks",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "reference_overlay": {
                                    "mode": "ticks",
                                    "tick_side": "right",
                                    "tick_length_ratio": 0.15,
                                    "threshold": 0.5,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        dataset = WellDataset(name="reference overlay ticks")
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset.add_channel(
            ScalarChannel("A", depth, "m", "u", values=np.array([0.0, 1.0, 0.0], dtype=float))
        )
        fig = plt.figure(figsize=(2, 4), dpi=100)
        try:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[0],
                document,
                dataset,
            )
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_reference_overlay_curve_callouts_reuse_fractional_track_space(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference overlay callouts",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "reference_overlay": {
                                    "mode": "curve",
                                    "lane_start": 0.08,
                                    "lane_end": 0.24,
                                },
                                "callouts": [
                                    {
                                        "depth": 1001.0,
                                        "label": "A1",
                                        "side": "right",
                                        "text_x": 0.30,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        dataset = WellDataset(name="reference overlay callouts")
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset.add_channel(
            ScalarChannel("A", depth, "m", "u", values=np.array([20.0, 50.0, 80.0], dtype=float))
        )
        fig = plt.figure(figsize=(2, 4), dpi=100)
        try:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(1002.0, 1000.0)
            renderer._draw_curve_callouts(
                ax,
                document.tracks[0],
                document,
                dataset,
                DepthWindow(page_number=1, start=1000.0, stop=1002.0, unit="m"),
                independent_curve_scales=False,
            )
            labels = [text for text in ax.texts if text.get_text() == "A1"]
            self.assertEqual(len(labels), 1)
            label = labels[0]
            self.assertGreater(label.xy[0], 0.08)
            self.assertLess(label.xy[0], 0.24)
        finally:
            plt.close(fig)

    def test_reference_track_header_keeps_scale_row_and_draws_overlay_properties(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference header properties",
                "page": {"size": "A4", "track_header_height_mm": 12.0},
                "depth": {"unit": "ft", "scale": "1:240"},
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "reference": {"define_layout": True, "unit": "ft", "scale_ratio": 240},
                        "track_header": {
                            "objects": [
                                {
                                    "kind": "title",
                                    "enabled": False,
                                    "reserve_space": False,
                                },
                                {"kind": "scale", "enabled": True, "line_units": 1},
                                {"kind": "legend", "enabled": True, "line_units": 4},
                            ]
                        },
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "TT",
                                "label": "Transit Time",
                                "style": {"color": "#1238ff"},
                                "scale": {
                                    "kind": "linear",
                                    "min": 200,
                                    "max": 400,
                                    "reverse": True,
                                },
                                "reference_overlay": {"mode": "curve"},
                            }
                        ],
                    }
                ],
            }
        )
        dataset = WellDataset(name="reference header properties")
        depth = np.array([1000.0, 1001.0], dtype=float)
        dataset.add_channel(ScalarChannel("TT", depth, "ft", "us", values=np.array([250.0, 260.0])))
        renderer = MatplotlibRenderer()
        fig = plt.figure(figsize=(2.2, 2.0), dpi=100)
        try:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            renderer._draw_track_header(ax, document.tracks[0], document, dataset)
            text_values = [text.get_text() for text in ax.texts]
            self.assertIn("ft 1:240", text_values)
            self.assertIn("Transit Time", text_values)
            self.assertIn("400", text_values)
            self.assertIn("us", text_values)
            self.assertIn("200", text_values)
        finally:
            plt.close(fig)

    def test_reference_events_draw_track_segments_and_callouts(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference events",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200", "major_step": 1, "minor_step": 0.2},
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "reference": {
                            "events": [
                                {
                                    "depth": 1001.0,
                                    "label": "Casing Foot",
                                    "tick_side": "right",
                                    "tick_length_ratio": 0.16,
                                    "text_side": "left",
                                    "text_x": 0.72,
                                }
                            ]
                        },
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        fig = plt.figure(figsize=(2, 4), dpi=100)
        try:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(1002.0, 1000.0)
            renderer._draw_reference_events(
                ax,
                document.tracks[0],
                DepthWindow(page_number=1, start=1000.0, stop=1002.0, unit="m"),
            )
            renderer._draw_reference_event_callouts(
                ax,
                document.tracks[0],
                document,
                DepthWindow(page_number=1, start=1000.0, stop=1002.0, unit="m"),
            )
            self.assertGreaterEqual(len(ax.lines), 1)
            labels = [text for text in ax.texts if text.get_text() == "Casing Foot"]
            self.assertEqual(len(labels), 1)
        finally:
            plt.close(fig)

    def test_raster_header_triplet_uses_combined_scale_legend_span(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0])
        samples = np.array([200.0, 700.0, 1200.0])
        values = np.array(
            [
                [0.1, 0.3, 0.2],
                [0.0, -0.2, -0.1],
                [0.2, 0.1, -0.2],
            ],
            dtype=float,
        )
        dataset = WellDataset(name="sample")
        dataset.add_channel(
            RasterChannel(
                "VDL",
                depth=depth,
                depth_unit="m",
                values=values,
                sample_axis=samples,
                sample_unit="us",
            )
        )
        document = document_from_mapping(
            {
                "name": "raster header triplet",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "track_header": {
                            "objects": [
                                {
                                    "kind": "title",
                                    "enabled": False,
                                    "reserve_space": False,
                                },
                                {"kind": "scale", "enabled": True, "line_units": 1},
                                {"kind": "legend", "enabled": True, "line_units": 2},
                            ]
                        },
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "colorbar": {"enabled": True, "position": "header"},
                                "sample_axis": {"min": 200, "max": 1200, "unit": "us"},
                            }
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        track = document.tracks[0]
        slots = renderer._track_header_slots(track)
        triplet = renderer._raster_header_triplet_slot(track, slots, dataset)
        self.assertIsNotNone(triplet)
        assert triplet is not None
        self.assertEqual(triplet[0], 0)
        self.assertEqual(triplet[1], 1)
        self.assertGreater(triplet[2], triplet[3])

    def test_raster_header_uses_curve_property_group_capacity(self) -> None:
        document = document_from_mapping(
            {
                "name": "uniform raster header groups",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "combo",
                        "title": "Combo",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {"kind": "curve", "channel": "A"},
                            {"kind": "curve", "channel": "B"},
                            {"kind": "curve", "channel": "C"},
                            {"kind": "curve", "channel": "D"},
                        ],
                    },
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "track_header": {
                            "objects": [
                                {
                                    "kind": "title",
                                    "enabled": False,
                                    "reserve_space": False,
                                },
                                {"kind": "scale", "enabled": True, "line_units": 1},
                                {"kind": "legend", "enabled": True, "line_units": 2},
                            ]
                        },
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "colorbar": {"enabled": True, "position": "header"},
                            }
                        ],
                    },
                ],
            }
        )
        renderer = MatplotlibRenderer()
        self.assertEqual(renderer._header_property_group_capacity(document), 4)
        rows = renderer._curve_row_bounds(
            0.9,
            0.3,
            renderer._header_property_group_capacity(document) * 3,
        )
        self.assertEqual(len(rows), 12)

    def test_vdl_header_colorbar_uses_min_amplitude_max_labels(self) -> None:
        renderer = MatplotlibRenderer()
        element = document_from_mapping(
            {
                "name": "vdl header labels",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "colorbar": {"enabled": True, "label": "Amplitude"},
                            }
                        ],
                    }
                ],
            }
        ).tracks[0].elements[0]
        channel = RasterChannel(
            "VDL",
            depth=np.array([0.0, 1.0]),
            depth_unit="m",
            values=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            sample_axis=np.array([200.0, 1200.0]),
            sample_unit="us",
        )
        left, center, right = renderer._raster_header_colorbar_text_triplet(
            element,
            channel,
            limits=(-1.0, 1.0),
        )
        self.assertEqual(left, "Min")
        self.assertEqual(center, "Amplitude")
        self.assertEqual(right, "Max")

    def test_curve_header_display_controls_scale_text_and_color(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0])
        dataset = WellDataset(name="sample")
        dataset.add_channel(
            ScalarChannel(
                "TENS",
                depth,
                "m",
                "lbf",
                values=np.array([5000.0, 7000.0, 9000.0]),
            )
        )
        element = CurveElement(
            channel="TENS",
            style=StyleSpec(color="#123456"),
            scale=ScaleSpec(kind=ScaleKind.LINEAR, minimum=5000.0, maximum=15000.0, reverse=False),
            header_display=CurveHeaderDisplaySpec(
                show_name=False,
                show_unit=False,
                show_limits=True,
                show_color=False,
            ),
        )
        renderer = MatplotlibRenderer()
        left, unit, right = renderer._curve_scale_text_triplet(
            track=None, element=element, dataset=dataset
        )
        self.assertEqual(left, "5000")
        self.assertEqual(unit, "")
        self.assertEqual(right, "15000")
        self.assertEqual(renderer._curve_header_label(element), "")
        self.assertEqual(renderer._curve_header_color(element), "#111111")

    def test_curve_header_name_wrap_is_optional(self) -> None:
        renderer = MatplotlibRenderer()
        wrapped = renderer._format_curve_header_label(
            CurveElement(
                channel="TT",
                label="Transit Time Overlay (TT)",
                header_display=CurveHeaderDisplaySpec(wrap_name=True),
            ),
            label="Transit Time Overlay (TT)",
            max_chars=12,
        )
        truncated = renderer._format_curve_header_label(
            CurveElement(
                channel="TT",
                label="Transit Time Overlay (TT)",
                header_display=CurveHeaderDisplaySpec(wrap_name=False),
            ),
            label="Transit Time Overlay (TT)",
            max_chars=12,
        )
        self.assertIn("\n", wrapped)
        self.assertNotIn("\n", truncated)
        self.assertTrue(truncated.endswith("..."))

    def test_header_char_budget_converts_points_to_pixels(self) -> None:
        renderer = MatplotlibRenderer(dpi=300)
        fig, ax = plt.subplots(figsize=(1.0, 1.0), dpi=300)
        try:
            budget = renderer._header_char_budget(
                ax,
                available_width_ratio=0.9,
                font_size_pt=5.0,
                char_width_ratio=0.75,
                min_chars=4,
            )
        finally:
            plt.close(fig)
        self.assertLessEqual(budget, 14)
        self.assertGreaterEqual(budget, 4)

    def test_grid_segment_positions_support_scale_modes(self) -> None:
        renderer = MatplotlibRenderer()
        linear = renderer._grid_segment_positions(4, GridScaleKind.LINEAR)
        logarithmic = renderer._grid_segment_positions(4, GridScaleKind.LOGARITHMIC)
        tangential = renderer._grid_segment_positions(4, GridScaleKind.TANGENTIAL)

        self.assertTrue(np.all(np.diff(linear) > 0))
        self.assertTrue(np.all(np.diff(logarithmic) > 0))
        self.assertTrue(np.all(np.diff(tangential) > 0))
        self.assertAlmostEqual(float(linear[0]), 0.0)
        self.assertAlmostEqual(float(linear[-1]), 1.0)
        self.assertAlmostEqual(float(logarithmic[0]), 0.0)
        self.assertAlmostEqual(float(logarithmic[-1]), 1.0)
        self.assertAlmostEqual(float(tangential[0]), 0.0)
        self.assertAlmostEqual(float(tangential[-1]), 1.0)
        self.assertFalse(np.allclose(linear, logarithmic))
        self.assertFalse(np.allclose(linear, tangential))

    def test_normalize_curve_values_supports_tangential_scale(self) -> None:
        renderer = MatplotlibRenderer()
        values = np.array([0.0, 25.0, 50.0, 100.0, 150.0], dtype=float)
        normalized, mask = renderer._normalize_curve_values(
            values,
            ScaleSpec(kind=ScaleKind.TANGENTIAL, minimum=0.0, maximum=150.0),
        )
        self.assertTrue(np.all(mask))
        self.assertGreaterEqual(float(np.nanmin(normalized[mask])), 0.0)
        self.assertLessEqual(float(np.nanmax(normalized[mask])), 1.0)
        self.assertTrue(np.all(np.diff(normalized[mask]) > 0))

    def test_log_wrap_transforms_values_into_log_interval(self) -> None:
        renderer = MatplotlibRenderer()
        scale = ScaleSpec(kind=ScaleKind.LOG, minimum=2.0, maximum=200.0)
        values = np.array([1.0, 2.0, 20.0, 200.0, 2000.0], dtype=float)
        wrapped, valid_mask, wrapped_mask = renderer._wrap_curve_values(values, scale)
        self.assertTrue(np.all(valid_mask))
        self.assertTrue(np.all(np.isfinite(wrapped[valid_mask])))
        self.assertGreaterEqual(float(np.nanmin(wrapped[valid_mask])), 2.0)
        self.assertLessEqual(float(np.nanmax(wrapped[valid_mask])), 200.0)
        self.assertTrue(wrapped_mask[0])
        self.assertFalse(wrapped_mask[3])
        self.assertAlmostEqual(float(wrapped[0]), 100.0, places=4)
        self.assertAlmostEqual(float(wrapped[4]), 20.0, places=4)

    def test_curve_callouts_draw_text_annotations(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve callouts",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200", "major_step": 10, "minor_step": 2},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "linear", "min": 0, "max": 150},
                                "callouts": [
                                    {"depth": 1005, "label": "GR Sand"},
                                    {"depth": 1010, "label": "GR Shale", "side": "left"},
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1005.0, 1010.0, 1015.0], dtype=float)
        dataset = WellDataset(name="curve callouts")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.array([20, 35, 110, 90]))
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            window = type("Window", (), {"start": 1000.0, "stop": 1015.0})()
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[0],
                document,
                dataset,
                independent_curve_scales=False,
            )
            renderer._draw_curve_callouts(
                ax,
                document.tracks[0],
                document,
                dataset,
                window,
                independent_curve_scales=False,
            )
            labels = {text.get_text() for text in ax.texts}
            self.assertIn("GR Sand", labels)
            self.assertIn("GR Shale", labels)
        finally:
            plt.close(fig)

    def test_curve_callout_auto_side_uses_curve_position(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve callout auto side",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200", "major_step": 10, "minor_step": 2},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "linear", "min": 0, "max": 150},
                                "callouts": [
                                    {"depth": 1005, "label": "Low GR"},
                                    {"depth": 1010, "label": "High GR"},
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1005.0, 1010.0, 1015.0], dtype=float)
        dataset = WellDataset(name="curve callout auto side")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.array([15, 25, 125, 130]))
        )

        renderer = MatplotlibRenderer()
        window = type("Window", (), {"start": 1000.0, "stop": 1015.0})()
        records = renderer._curve_callout_records(
            document.tracks[0],
            document,
            dataset,
            window,
            independent_curve_scales=False,
        )
        by_label = {record.label: record for record in records}
        self.assertEqual(by_label["Low GR"].side, "right")
        self.assertEqual(by_label["High GR"].side, "left")

    def test_curve_callout_arrow_relpos_uses_side_center(self) -> None:
        renderer = MatplotlibRenderer()
        self.assertEqual(renderer._curve_callout_arrow_relpos("right"), (0.0, 0.5))
        self.assertEqual(renderer._curve_callout_arrow_relpos("left"), (1.0, 0.5))

    def test_curve_callout_top_and_bottom_expands_repeated_records(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve callout repeated bands",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200", "major_step": 10, "minor_step": 2},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "linear", "min": 0, "max": 150},
                                "callouts": [
                                    {
                                        "depth": 1000,
                                        "label": "GR",
                                        "placement": "top_and_bottom",
                                        "distance_from_top": 1,
                                        "distance_from_bottom": 2,
                                        "every": 5,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1005.0, 1010.0, 1015.0], dtype=float)
        dataset = WellDataset(name="curve callout repeated bands")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.array([15, 25, 125, 130]))
        )

        renderer = MatplotlibRenderer()
        window = type("Window", (), {"start": 1000.0, "stop": 1015.0})()
        records = renderer._curve_callout_records(
            document.tracks[0],
            document,
            dataset,
            window,
            independent_curve_scales=False,
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(
            [record.anchor_y for record in records],
            [1001.0, 1003.0, 1006.0, 1008.0, 1011.0, 1013.0],
        )
        self.assertEqual(
            [record.desired_text_y for record in records],
            [1001.0, 1003.0, 1006.0, 1008.0, 1011.0, 1013.0],
        )

    def test_curve_callout_repetition_uses_section_depth_range_not_page_window(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve callout section repetition",
                "page": {"size": "A4"},
                "depth_range": [1000.0, 1100.0],
                "depth": {"unit": "m", "scale": "1:200", "major_step": 10, "minor_step": 2},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "linear", "min": 0, "max": 150},
                                "callouts": [
                                    {
                                        "depth": 0,
                                        "label": "GR",
                                        "placement": "top",
                                        "distance_from_top": 5,
                                        "every": 30,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1020.0, 1040.0, 1060.0, 1080.0, 1100.0], dtype=float)
        dataset = WellDataset(name="curve callout section repetition")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.array([15, 25, 125, 130, 95, 80]))
        )

        renderer = MatplotlibRenderer()
        window = type("Window", (), {"start": 1020.0, "stop": 1040.0})()
        records = renderer._curve_callout_records(
            document.tracks[0],
            document,
            dataset,
            window,
            independent_curve_scales=False,
        )
        self.assertEqual([record.anchor_y for record in records], [1035.0])
        self.assertEqual([record.desired_text_y for record in records], [1035.0])

    def test_curve_callout_text_stays_within_track_edges(self) -> None:
        long_label = "Curve Label Near Edge"
        document = document_from_mapping(
            {
                "name": "curve callout edge avoidance",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200", "major_step": 10, "minor_step": 2},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "linear", "min": 0, "max": 150},
                                "callouts": [
                                    {
                                        "depth": 1006,
                                        "label": long_label,
                                        "side": "left",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1005.0, 1010.0, 1015.0], dtype=float)
        dataset = WellDataset(name="curve callout edge avoidance")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.array([5, 8, 12, 18]))
        )

        renderer = MatplotlibRenderer(
            style={
                "curve_callouts": {
                    "left_text_x": 0.0,
                    "lane_count": 1,
                }
            }
        )
        fig, ax = plt.subplots(figsize=(2.5, 6.0))
        try:
            window = type("Window", (), {"start": 1000.0, "stop": 1015.0})()
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[0],
                document,
                dataset,
                independent_curve_scales=False,
            )
            records = renderer._place_curve_callouts(
                ax,
                document.tracks[0],
                document,
                dataset,
                window,
                independent_curve_scales=False,
            )
            fig.canvas.draw()
            self.assertTrue(any(record.text_y is not None for record in records))
            placed = next(record for record in records if record.label == long_label)
            from matplotlib.transforms import blended_transform_factory

            text_transform = blended_transform_factory(ax.transAxes, ax.transData)
            text_bbox = renderer._measure_curve_callout_bbox(
                ax,
                renderer=fig.canvas.get_renderer(),
                label=placed.label,
                text_x=placed.text_x,
                text_y=placed.text_y,
                transform=text_transform,
                fontsize=placed.font_size,
                color=placed.color,
                fontweight=placed.font_weight,
                fontstyle=placed.font_style,
                horizontal_alignment=renderer._curve_callout_horizontal_alignment(
                    placed.placed_side or placed.side
                ),
            )
            axes_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
            padding_px = float(renderer.style["curve_callouts"]["edge_padding_px"])
            self.assertGreaterEqual(text_bbox.x0, axes_bbox.x0 + padding_px - 0.5)
            self.assertLessEqual(text_bbox.x1, axes_bbox.x1 - padding_px + 0.5)
            self.assertGreaterEqual(text_bbox.y0, axes_bbox.y0 + padding_px - 0.5)
            self.assertLessEqual(text_bbox.y1, axes_bbox.y1 - padding_px + 0.5)
        finally:
            plt.close(fig)

    def test_between_curves_fill_adds_single_collection(self) -> None:
        document = document_from_mapping(
            {
                "name": "between curves fill",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "porosity",
                        "title": "Porosity",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "B",
                                    "color": "#22c55e",
                                    "alpha": 0.3,
                                },
                            },
                            {
                                "kind": "curve",
                                "channel": "B",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                            },
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
        dataset = WellDataset(name="fill")
        dataset.add_channel(ScalarChannel("A", depth, "m", "pu", values=np.array([10, 20, 35, 50])))
        dataset.add_channel(ScalarChannel("B", depth, "m", "pu", values=np.array([20, 25, 40, 65])))

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            track = document.tracks[0]
            element = track.elements[0]
            renderer._draw_curve(
                ax,
                track,
                element,
                document,
                dataset,
                independent_curve_scales=True,
            )
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_track_header_draws_fill_indicator_row(self) -> None:
        document = document_from_mapping(
            {
                "name": "fill header row",
                "page": {"size": "A4", "track_header_height_mm": 12.0},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "crossover",
                        "title": "Track 3",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "label": "Density",
                                "scale": {"kind": "linear", "min": 1.7, "max": 2.7},
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "B",
                                    "label": "Gas Effect",
                                    "color": "#f2bf16",
                                },
                            },
                            {
                                "kind": "curve",
                                "channel": "B",
                                "label": "Neutron",
                                "scale": {"kind": "linear", "min": 1.7, "max": 2.7},
                            },
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset = WellDataset(name="header fill")
        dataset.add_channel(
            ScalarChannel("A", depth, "m", "g/cm3", values=np.array([2.1, 2.2, 2.3]))
        )
        dataset.add_channel(
            ScalarChannel("B", depth, "m", "g/cm3", values=np.array([2.0, 2.15, 2.35]))
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            renderer._draw_track_header(ax, document.tracks[0], document, dataset)
            self.assertGreaterEqual(len(ax.patches), 1)
            self.assertIn("Gas Effect", {text.get_text() for text in ax.texts})
        finally:
            plt.close(fig)

    def test_between_curves_crossover_fill_adds_two_collections(self) -> None:
        document = document_from_mapping(
            {
                "name": "between curves crossover",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "porosity",
                        "title": "Porosity",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "B",
                                    "color": "#999999",
                                    "crossover": {
                                        "enabled": True,
                                        "left_color": "#22c55e",
                                        "right_color": "#ef4444",
                                    },
                                },
                            },
                            {
                                "kind": "curve",
                                "channel": "B",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                            },
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
        dataset = WellDataset(name="crossover")
        dataset.add_channel(ScalarChannel("A", depth, "m", "pu", values=np.array([10, 60, 70, 30])))
        dataset.add_channel(ScalarChannel("B", depth, "m", "pu", values=np.array([30, 20, 50, 40])))

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            track = document.tracks[0]
            element = track.elements[0]
            renderer._draw_curve(
                ax,
                track,
                element,
                document,
                dataset,
                independent_curve_scales=True,
            )
            self.assertEqual(len(ax.collections), 2)
        finally:
            plt.close(fig)

    def test_between_curves_fill_rejects_mismatched_scales(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset = WellDataset(name="mismatch")
        dataset.add_channel(ScalarChannel("A", depth, "m", "pu", values=np.array([10, 20, 30])))
        dataset.add_channel(ScalarChannel("B", depth, "m", "pu", values=np.array([0.1, 0.2, 0.3])))
        document = document_from_mapping(
            {
                "name": "between curves mismatch",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "porosity",
                        "title": "Porosity",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "B",
                                },
                            },
                            {
                                "kind": "curve",
                                "channel": "B",
                                "scale": {"kind": "linear", "min": 0, "max": 1},
                            },
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            with self.assertRaises(TemplateValidationError):
                renderer._draw_curve(
                    ax,
                    document.tracks[0],
                    document.tracks[0].elements[0],
                    document,
                    dataset,
                    independent_curve_scales=True,
                )
        finally:
            plt.close(fig)

    def test_between_curves_fill_selects_matching_scale_for_duplicate_target_channel(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
        dataset = WellDataset(name="duplicate target")
        dataset.add_channel(ScalarChannel("A", depth, "m", "mV", values=np.array([2, 4, 6, 8])))
        dataset.add_channel(ScalarChannel("B", depth, "m", "mV", values=np.array([1, 3, 5, 7])))
        document = document_from_mapping(
            {
                "name": "duplicate target scale",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "cbl",
                        "title": "CBL",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                            },
                            {
                                "kind": "curve",
                                "channel": "B",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                            },
                            {
                                "kind": "curve",
                                "channel": "A",
                                "scale": {"kind": "linear", "min": 0, "max": 10},
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "B",
                                    "color": "#22c55e",
                                },
                            },
                            {
                                "kind": "curve",
                                "channel": "B",
                                "scale": {"kind": "linear", "min": 0, "max": 10},
                            },
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[2],
                document,
                dataset,
                independent_curve_scales=True,
            )
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_between_instances_fill_supports_same_channel_with_different_scales(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
        dataset = WellDataset(name="same channel instances")
        dataset.add_channel(ScalarChannel("CBL", depth, "m", "mV", values=np.array([2, 4, 6, 8])))
        document = document_from_mapping(
            {
                "name": "between instances",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "cbl",
                        "title": "CBL",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "id": "cbl_0_100",
                                "channel": "CBL",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "fill": {
                                    "kind": "between_instances",
                                    "other_element_id": "cbl_0_10",
                                    "color": "#d1d5db",
                                },
                            },
                            {
                                "kind": "curve",
                                "id": "cbl_0_10",
                                "channel": "CBL",
                                "scale": {"kind": "linear", "min": 0, "max": 10},
                            },
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[0],
                document,
                dataset,
                independent_curve_scales=True,
            )
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_to_lower_limit_fill_adds_single_collection(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
        dataset = WellDataset(name="lower limit fill")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.array([20, 40, 60, 80]))
        )
        document = document_from_mapping(
            {
                "name": "lower limit fill",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "linear", "min": 0, "max": 100},
                                "fill": {
                                    "kind": "to_lower_limit",
                                    "color": "#22c55e",
                                },
                            }
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[0],
                document,
                dataset,
                independent_curve_scales=False,
            )
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_baseline_split_fill_adds_two_collections_and_baseline(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
        dataset = WellDataset(name="baseline split")
        dataset.add_channel(
            ScalarChannel("NPHI", depth, "m", "ft3/ft3", values=np.array([0.05, 0.20, 0.10, 0.25]))
        )
        document = document_from_mapping(
            {
                "name": "baseline split fill",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "nphi",
                        "title": "NPHI",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "NPHI",
                                "scale": {"kind": "linear", "min": 0.0, "max": 0.45},
                                "fill": {
                                    "kind": "baseline_split",
                                    "alpha": 0.3,
                                    "baseline": {
                                        "value": 0.15,
                                        "lower_color": "#22c55e",
                                        "upper_color": "#ef4444",
                                        "line_color": "#111111",
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            renderer._draw_curve(
                ax,
                document.tracks[0],
                document.tracks[0].elements[0],
                document,
                dataset,
                independent_curve_scales=False,
            )
            self.assertEqual(len(ax.collections), 2)
            self.assertEqual(len(ax.lines), 2)
        finally:
            plt.close(fig)

    def test_baseline_split_header_draws_indicator_and_label(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset = WellDataset(name="baseline header")
        dataset.add_channel(
            ScalarChannel("NPHI", depth, "m", "ft3/ft3", values=np.array([0.05, 0.20, 0.10]))
        )
        document = document_from_mapping(
            {
                "name": "baseline header",
                "page": {"size": "A4", "track_header_height_mm": 12.0},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "nphi",
                        "title": "NPHI",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "NPHI",
                                "scale": {"kind": "linear", "min": 0.0, "max": 0.45},
                                "fill": {
                                    "kind": "baseline_split",
                                    "label": "Gas Effect",
                                    "baseline": {
                                        "value": 0.15,
                                        "lower_color": "#22c55e",
                                        "upper_color": "#ef4444",
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        fig, ax = plt.subplots()
        try:
            renderer._draw_track_header(ax, document.tracks[0], document, dataset)
            self.assertIn("Gas Effect", {text.get_text() for text in ax.texts})
            self.assertGreaterEqual(len(ax.patches), 1)
        finally:
            plt.close(fig)

    def test_baseline_split_header_uses_baseline_position_and_reverse_orientation(self) -> None:
        depth = np.array([1000.0, 1001.0, 1002.0], dtype=float)
        dataset = WellDataset(name="baseline orientation")
        dataset.add_channel(
            ScalarChannel("TT", depth, "m", "us", values=np.array([280.0, 320.0, 295.0]))
        )
        document = document_from_mapping(
            {
                "name": "baseline orientation",
                "page": {"size": "A4", "track_header_height_mm": 12.0},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "tt",
                        "title": "TT",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "TT",
                                "scale": {
                                    "kind": "linear",
                                    "min": 200,
                                    "max": 400,
                                    "reverse": True,
                                },
                                "fill": {
                                    "kind": "baseline_split",
                                    "baseline": {
                                        "value": 300,
                                        "lower_color": "#22c55e",
                                        "upper_color": "#ef4444",
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        )

        renderer = MatplotlibRenderer()
        segments = renderer._curve_fill_header_segments(
            document.tracks[0],
            document.tracks[0].elements[0],
            document,
            dataset,
            independent_curve_scales=False,
        )
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0][0], 0.0)
        self.assertAlmostEqual(segments[0][1], 0.5)
        self.assertEqual(segments[0][2], "#ef4444")
        self.assertAlmostEqual(segments[1][0], 0.5)
        self.assertAlmostEqual(segments[1][1], 1.0)
        self.assertEqual(segments[1][2], "#22c55e")

    def test_vdl_raster_profile_normalizes_trace_amplitude(self) -> None:
        document = document_from_mapping(
            {
                "name": "vdl profile",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "normalization": "auto",
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1001.0, 1002.0])
        values = np.array(
            [
                [1.0, -2.0, 2.0],
                [10.0, -10.0, 5.0],
                [4.0, -1.0, 0.5],
            ],
            dtype=float,
        )
        dataset = WellDataset(name="vdl")
        dataset.add_channel(
            RasterChannel(
                "VDL",
                depth,
                "m",
                "amp",
                values=values,
                sample_axis=np.array([0.0, 1.0, 2.0], dtype=float),
            )
        )
        renderer = MatplotlibRenderer()
        track = document.tracks[0]
        element = track.elements[0]
        channel = dataset.get_channel("VDL")
        assert isinstance(channel, RasterChannel)
        _, normalized = renderer._prepare_raster_display_data(
            channel.depth,
            channel.values,
            element,
            target="waveform",
        )
        self.assertAlmostEqual(float(np.nanmax(np.abs(normalized[0]))), 1.0)
        self.assertAlmostEqual(float(np.nanmax(np.abs(normalized[1]))), 1.0)
        self.assertAlmostEqual(float(np.nanmax(np.abs(normalized[2]))), 1.0)
        limits = renderer._resolve_raster_color_limits(normalized, element)
        assert limits is not None
        self.assertAlmostEqual(abs(float(limits[0])), abs(float(limits[1])), places=6)
        self.assertGreater(float(limits[1]), 0.9)

    def test_vdl_raster_display_sorts_depth_and_removes_trace_bias(self) -> None:
        document = document_from_mapping(
            {
                "name": "vdl sorting",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "normalization": "none",
                            }
                        ],
                    }
                ],
            }
        )
        depth = np.array([1002.0, 1001.0, 1000.0])
        values = np.array(
            [
                [11.0, 12.0, 13.0],
                [6.0, 7.0, 8.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=float,
        )
        dataset = WellDataset(name="vdl")
        dataset.add_channel(
            RasterChannel(
                "VDL",
                depth,
                "m",
                "amp",
                values=values,
                sample_axis=np.array([200.0, 700.0, 1200.0], dtype=float),
            )
        )
        renderer = MatplotlibRenderer()
        element = document.tracks[0].elements[0]
        channel = dataset.get_channel("VDL")
        assert isinstance(channel, RasterChannel)
        prepared_depth, prepared_values = renderer._prepare_raster_display_data(
            channel.depth,
            channel.values,
            element,
            target="raster",
        )
        np.testing.assert_array_equal(prepared_depth, np.array([1000.0, 1001.0, 1002.0]))
        np.testing.assert_allclose(
            prepared_values,
            np.array(
                [
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ]
            ),
        )

    def test_raster_window_crops_to_selected_sample_axis_range(self) -> None:
        document = document_from_mapping(
            {
                "name": "vdl crop",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "x_scale": {"kind": "linear", "min": 200, "max": 1200},
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "sample_axis": {
                                    "unit": "us",
                                    "source_origin": 40,
                                    "source_step": 10,
                                    "min": 200,
                                    "max": 1200,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        channel = RasterChannel(
            "VDL",
            np.array([1000.0, 1001.0], dtype=float),
            "m",
            "amp",
            values=np.vstack([np.arange(256, dtype=float), np.arange(256, dtype=float)]),
            sample_axis=np.arange(256, dtype=float),
        )
        element = document.tracks[0].elements[0]
        sample_axis = renderer._resolved_raster_sample_axis(channel, element)
        clipped_axis, clipped_values = renderer._clip_raster_columns_to_window(
            sample_axis,
            channel.values,
            axis_min=200.0,
            axis_max=1200.0,
        )
        self.assertEqual(clipped_axis[0], 200.0)
        self.assertEqual(clipped_axis[-1], 1200.0)
        self.assertEqual(clipped_axis.shape[0], 101)
        self.assertEqual(clipped_values.shape, (2, 101))
        np.testing.assert_array_equal(clipped_values[0], np.arange(16, 117, dtype=float))

    def test_vdl_profile_uses_split_auto_normalization_modes(self) -> None:
        document = document_from_mapping(
            {
                "name": "vdl profile split",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 40,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "vdl",
                                "normalization": "auto",
                                "waveform_normalization": "auto",
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        renderer = MatplotlibRenderer()
        self.assertEqual(
            renderer._resolve_raster_normalization(element, target="raster"),
            "global_maxabs",
        )
        self.assertEqual(
            renderer._resolve_raster_normalization(element, target="waveform"),
            "trace_maxabs",
        )

    def test_log_grid_scale_mode_auto_uses_cycles_from_scale_bounds(self) -> None:
        document = document_from_mapping(
            {
                "name": "log cycles",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "rt",
                        "title": "RT",
                        "kind": "normal",
                        "width_mm": 30,
                        "x_scale": {"kind": "log", "min": 2, "max": 2000},
                        "grid": {
                            "vertical": {
                                "main": {"scale": "logarithmic", "spacing_mode": "scale"},
                                "secondary": {"scale": "logarithmic", "spacing_mode": "scale"},
                            }
                        },
                        "elements": [{"kind": "curve", "channel": "RT"}],
                    }
                ],
            }
        )
        depth = np.array([1000.0, 1001.0, 1002.0])
        dataset = WellDataset(name="sample")
        dataset.add_channel(
            ScalarChannel(
                "RT",
                depth,
                "m",
                "ohm.m",
                values=np.array([2.0, 20.0, 2000.0]),
            )
        )
        renderer = MatplotlibRenderer()
        main_lines, secondary_lines = renderer._vertical_grid_fractions(document.tracks[0], dataset)
        self.assertEqual(len(main_lines), 3)
        self.assertGreater(len(secondary_lines), 3)

    def test_render_documents_auto_uses_strip_pages_for_multisection_pdf(self) -> None:
        document = document_from_mapping(
            {
                "name": "multi",
                "page": {"size": "A4", "continuous": True},
                "depth": {"unit": "m", "scale": "1:200"},
                "depth_range": [1000.0, 1200.0],
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 20,
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        dataset = WellDataset(name="sample")

        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "multisection.pdf"
            result = renderer.render_documents((document, document), dataset, output_path=output)
            self.assertTrue(output.exists())
            # Without auto-strip this would be one continuous page per section (2 pages total).
            self.assertGreater(result.page_count, 2)

    def test_render_array_track_with_colorbar_and_sample_axis(self) -> None:
        document = document_from_mapping(
            {
                "name": "array track",
                "page": {"size": "A4", "continuous": True},
                "depth": {"unit": "m", "scale": "1:200"},
                "depth_range": [1000.0, 1020.0],
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "reference", "width_mm": 18},
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 42,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "style": {"colormap": "bone"},
                                "color_limits": [-1.0, 1.0],
                                "colorbar": {"enabled": True, "label": "VDL amp"},
                                "sample_axis": {
                                    "enabled": True,
                                    "label": "Azimuth (deg)",
                                    "ticks": 7,
                                },
                                "waveform": {"enabled": True, "stride": 4, "max_traces": 80},
                            }
                        ],
                    },
                ],
            }
        )
        depth = np.linspace(1000.0, 1020.0, 120)
        azimuth = np.linspace(0.0, 360.0, 72)
        values = np.sin(depth[:, None] / 10.0) * np.cos(np.deg2rad(azimuth))[None, :]
        dataset = WellDataset(name="array")
        dataset.add_channel(
            RasterChannel(
                "VDL",
                depth,
                "m",
                "amplitude",
                values=values,
                sample_axis=azimuth,
                sample_unit="deg",
                sample_label="azimuth",
            )
        )
        renderer = MatplotlibRenderer()
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "array_track.pdf"
            result = renderer.render(document, dataset, output_path=output)
            self.assertTrue(output.exists())
            self.assertEqual(result.page_count, 1)

    def test_array_plot_sample_axis_is_suppressed_when_bottom_header_exists(self) -> None:
        document = document_from_mapping(
            {
                "name": "array footer header suppression",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "depth_range": [1000.0, 1120.0],
                "tracks": [
                    {
                        "id": "vdl",
                        "title": "VDL",
                        "kind": "array",
                        "width_mm": 42,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "sample_axis": {"enabled": True, "label": "Time (us)"},
                            }
                        ],
                    }
                ],
            }
        )
        renderer = MatplotlibRenderer()
        dataset = WellDataset(name="array footer header suppression")
        layouts = LayoutEngine().layout(document, dataset)
        self.assertGreater(len(layouts), 1)
        track = document.tracks[0]
        self.assertTrue(renderer._should_draw_array_plot_sample_axis(layouts[0], track))
        self.assertFalse(renderer._should_draw_array_plot_sample_axis(layouts[-1], track))

    def test_waveform_selection_is_consistent_across_windows(self) -> None:
        document = document_from_mapping(
            {
                "name": "waveform selection",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "array",
                        "title": "Array",
                        "kind": "array",
                        "width_mm": 40,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "waveform",
                                "waveform": {"enabled": True, "stride": 3, "max_traces": 10},
                            }
                        ],
                    }
                ],
            }
        )
        waveform = document.tracks[0].elements[0].waveform
        depth = np.linspace(1000.0, 1099.0, 100)
        renderer = MatplotlibRenderer()

        first_window = renderer._select_waveform_indices(
            depth,
            window_top=1000.0,
            window_base=1049.0,
            waveform=waveform,
        )
        second_window = renderer._select_waveform_indices(
            depth,
            window_top=1050.0,
            window_base=1099.0,
            waveform=waveform,
        )

        expected = np.arange(depth.size, dtype=int)[:: waveform.stride]
        downsample = int(np.ceil(expected.size / waveform.max_traces))
        expected = expected[::downsample]
        expected_first = expected[(depth[expected] >= 1000.0) & (depth[expected] <= 1049.0)]
        expected_second = expected[(depth[expected] >= 1050.0) & (depth[expected] <= 1099.0)]

        np.testing.assert_array_equal(first_window, expected_first)
        np.testing.assert_array_equal(second_window, expected_second)

    def test_waveform_selection_anchors_from_top_depth_for_descending_index(self) -> None:
        document = document_from_mapping(
            {
                "name": "waveform descending",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "array",
                        "title": "Array",
                        "kind": "array",
                        "width_mm": 40,
                        "elements": [
                            {
                                "kind": "raster",
                                "channel": "VDL",
                                "profile": "waveform",
                                "waveform": {"enabled": True, "stride": 4},
                            }
                        ],
                    }
                ],
            }
        )
        waveform = document.tracks[0].elements[0].waveform
        depth = np.linspace(1100.0, 1000.0, 101)  # descending storage order
        renderer = MatplotlibRenderer()

        selected = renderer._select_waveform_indices(
            depth,
            window_top=1000.0,
            window_base=1100.0,
            waveform=waveform,
        )
        top_depth_index = int(np.argmin(depth))
        self.assertIn(top_depth_index, selected.tolist())
        depth_selected = depth[selected]
        expected = np.sort(np.argsort(depth, kind="mergesort")[:: waveform.stride])
        np.testing.assert_array_equal(selected, expected)
        self.assertTrue(np.all(np.isfinite(depth_selected)))


if __name__ == "__main__":
    unittest.main()
