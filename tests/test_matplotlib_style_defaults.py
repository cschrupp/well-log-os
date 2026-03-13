from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_renderer_uses_yaml_defaults(self) -> None:
        renderer = MatplotlibRenderer()
        self.assertEqual(renderer.style["track"]["frame_linewidth"], 0.8)
        self.assertEqual(renderer.style["grid"]["depth_major_alpha"], 0.9)
        self.assertEqual(renderer.style["track"]["reference_grid_mode"], "edge_ticks")
        self.assertEqual(renderer.style["track"]["reference_label_align"], "center")
        self.assertEqual(renderer.style["track"]["reference_label_x"], 0.5)
        self.assertEqual(renderer.style["track_header"]["paired_scale_text_offset_ratio"], 0.08)
        self.assertEqual(renderer.style["track_header"]["division_tick_count"], 5)
        self.assertEqual(renderer.style["track_header"]["title_align"], "left")
        self.assertEqual(renderer.style["track_header"]["title_x"], 0.03)
        self.assertTrue(renderer.style["section_title"]["enabled"])
        self.assertEqual(renderer.style["section_title"]["height_mm"], 6.0)
        self.assertEqual(renderer.style["raster"]["colorbar_width_ratio"], 0.06)
        self.assertEqual(renderer.style["raster"]["sample_axis_tick_labelsize"], 5.0)
        self.assertEqual(renderer.style["raster"]["header_colorbar_bar_height_ratio"], 0.26)

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
