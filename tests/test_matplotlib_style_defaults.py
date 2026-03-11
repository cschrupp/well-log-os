from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from well_log_os import (
    CurveElement,
    CurveHeaderDisplaySpec,
    GridScaleKind,
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


if __name__ == "__main__":
    unittest.main()
