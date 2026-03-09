from __future__ import annotations

import unittest

import numpy as np

from well_log_os import LayoutEngine, ScalarChannel, WellDataset, document_from_mapping
from well_log_os.errors import LayoutError


class LayoutTests(unittest.TestCase):
    def build_dataset(self) -> WellDataset:
        depth = np.linspace(1000.0, 1105.0, 300)
        dataset = WellDataset(name="synthetic")
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.linspace(20, 120, depth.size))
        )
        return dataset

    def test_layout_paginates_depth_range(self) -> None:
        document = document_from_mapping(
            {
                "name": "paginate",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16},
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "curve",
                        "width_mm": 25,
                        "x_scale": {"kind": "linear", "min": 0, "max": 150},
                        "elements": [{"kind": "curve", "channel": "GR"}],
                    },
                ],
            }
        )
        layout = LayoutEngine().layout(document, self.build_dataset())
        self.assertGreaterEqual(len(layout), 2)
        self.assertEqual(layout[0].track_frames[1].track.id, "gr")

    def test_track_width_overflow_raises(self) -> None:
        document = document_from_mapping(
            {
                "name": "overflow",
                "page": {"size": "A4", "margin_left_mm": 5, "margin_right_mm": 5},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 50},
                    {"id": "a", "title": "A", "kind": "curve", "width_mm": 100, "elements": []},
                    {"id": "b", "title": "B", "kind": "curve", "width_mm": 100, "elements": []},
                ],
            }
        )
        with self.assertRaises(LayoutError):
            LayoutEngine().track_frames(document)

    def test_continuous_mode_produces_single_tall_page(self) -> None:
        document = document_from_mapping(
            {
                "name": "continuous",
                "page": {"size": "A4", "continuous": True},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16},
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "curve",
                        "width_mm": 25,
                        "x_scale": {"kind": "linear", "min": 0, "max": 150},
                        "elements": [{"kind": "curve", "channel": "GR"}],
                    },
                ],
            }
        )
        layout = LayoutEngine().layout(document, self.build_dataset())
        self.assertEqual(len(layout), 1)
        self.assertGreater(layout[0].page.height_mm, 500)
        self.assertAlmostEqual(layout[0].depth_window.start, 1000.0, places=3)
        self.assertAlmostEqual(layout[0].depth_window.stop, 1105.0, places=3)

    def test_track_header_frames_align_with_track_frames(self) -> None:
        document = document_from_mapping(
            {
                "name": "headers",
                "page": {"size": "A4", "header_height_mm": 20, "track_header_height_mm": 9},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16},
                    {"id": "gr", "title": "GR", "kind": "curve", "width_mm": 25, "elements": []},
                ],
            }
        )
        page_layout = LayoutEngine().layout(document, self.build_dataset())[0]
        self.assertEqual(len(page_layout.track_header_frames), len(page_layout.track_frames))
        self.assertAlmostEqual(
            page_layout.track_header_frames[0].frame.y_mm,
            page_layout.page.margin_top_mm + page_layout.page.header_height_mm,
        )
        self.assertAlmostEqual(
            page_layout.track_header_frames[0].frame.height_mm,
            page_layout.page.track_header_height_mm,
        )
        self.assertAlmostEqual(
            page_layout.track_header_frames[1].frame.x_mm,
            page_layout.track_frames[1].frame.x_mm,
        )
        self.assertAlmostEqual(
            page_layout.track_header_frames[1].frame.width_mm,
            page_layout.track_frames[1].frame.width_mm,
        )

    def test_track_headers_only_render_on_layout_start_and_end(self) -> None:
        document = document_from_mapping(
            {
                "name": "headers once",
                "page": {"size": "A4", "header_height_mm": 20, "track_header_height_mm": 9},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16},
                    {"id": "gr", "title": "GR", "kind": "curve", "width_mm": 25, "elements": []},
                ],
            }
        )
        layouts = LayoutEngine().layout(document, self.build_dataset())
        self.assertGreaterEqual(len(layouts), 2)

        first = layouts[0]
        last = layouts[-1]
        self.assertEqual(len(first.track_header_top_frames), len(first.track_frames))
        self.assertEqual(len(first.track_header_bottom_frames), 0)
        self.assertEqual(len(last.track_header_bottom_frames), len(last.track_frames))

        if len(layouts) > 2:
            middle = layouts[1]
            self.assertEqual(len(middle.track_header_top_frames), 0)
            self.assertEqual(len(middle.track_header_bottom_frames), 0)


if __name__ == "__main__":
    unittest.main()
