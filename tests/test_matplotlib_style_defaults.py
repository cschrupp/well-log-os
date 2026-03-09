from __future__ import annotations

import unittest

import numpy as np

from well_log_os import (
    CurveElement,
    CurveHeaderDisplaySpec,
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

    def test_renderer_style_override_deep_merges(self) -> None:
        renderer = MatplotlibRenderer(
            style={
                "track": {"frame_color": "#111111"},
                "track_header": {"separator_linewidth": 0.5},
            }
        )
        self.assertEqual(renderer.style["track"]["frame_color"], "#111111")
        self.assertEqual(renderer.style["track"]["frame_linewidth"], 0.8)
        self.assertEqual(renderer.style["track_header"]["separator_linewidth"], 0.5)
        self.assertEqual(renderer.style["track_header"]["background_color"], "#e8e8e8")

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


if __name__ == "__main__":
    unittest.main()
