from __future__ import annotations

import unittest

from well_log_os.errors import TemplateValidationError
from well_log_os.model import (
    CurveElement,
    NumberFormatKind,
    RasterElement,
    TrackHeaderObjectKind,
    TrackKind,
)
from well_log_os.templates import document_from_mapping


class TemplateTests(unittest.TestCase):
    def test_curve_element_can_use_value_labels_mode(self) -> None:
        document = document_from_mapping(
            {
                "name": "value labels",
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
                                "render_mode": "value_labels",
                                "value_labels": {
                                    "step": 10,
                                    "format": "fixed",
                                    "precision": 1,
                                    "font_size": 6.0,
                                    "horizontal_alignment": "center",
                                },
                                "header_display": {
                                    "show_name": True,
                                    "show_unit": False,
                                    "show_limits": True,
                                    "show_color": False,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        self.assertEqual(element.render_mode, "value_labels")
        self.assertEqual(element.value_labels.step, 10.0)
        self.assertEqual(element.value_labels.number_format, NumberFormatKind.FIXED)
        self.assertTrue(element.header_display.show_name)
        self.assertFalse(element.header_display.show_unit)
        self.assertTrue(element.header_display.show_limits)
        self.assertFalse(element.header_display.show_color)

    def test_reference_track_can_define_layout_axis(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference axis",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200", "major_step": 10, "minor_step": 2},
                "tracks": [
                    {
                        "id": "ref",
                        "title": "Reference",
                        "kind": "reference",
                        "width_mm": 16,
                        "reference": {
                            "define_layout": True,
                            "unit": "ft",
                            "scale_ratio": 500,
                            "major_step": 50,
                            "secondary_grid": {"display": True, "line_count": 5},
                        },
                    }
                ],
            }
        )
        self.assertEqual(document.depth_axis.unit, "ft")
        self.assertEqual(document.depth_axis.scale_ratio, 500)
        self.assertEqual(document.depth_axis.major_step, 50.0)
        self.assertAlmostEqual(document.depth_axis.minor_step, 10.0)

    def test_reference_track_accepts_curve_overlay(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference overlay",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "ref",
                        "title": "Reference",
                        "kind": "reference",
                        "width_mm": 20,
                        "elements": [{"kind": "curve", "channel": "CBL"}],
                    }
                ],
            }
        )
        self.assertEqual(document.tracks[0].kind, TrackKind.REFERENCE)
        self.assertIsInstance(document.tracks[0].elements[0], CurveElement)

    def test_image_track_accepts_raster_and_curve_overlay(self) -> None:
        document = document_from_mapping(
            {
                "name": "image overlay",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16},
                    {
                        "id": "image",
                        "title": "Image",
                        "kind": "image",
                        "width_mm": 40,
                        "x_scale": {"kind": "linear", "min": 0, "max": 360},
                        "elements": [
                            {"kind": "raster", "channel": "FMI"},
                            {"kind": "curve", "channel": "FRACTURE_INTENSITY"},
                        ],
                    },
                ],
            }
        )
        image_track = document.tracks[1]
        self.assertEqual(image_track.kind, TrackKind.IMAGE)
        self.assertIsInstance(image_track.elements[0], RasterElement)
        self.assertIsInstance(image_track.elements[1], CurveElement)

    def test_curve_track_rejects_raster_elements(self) -> None:
        with self.assertRaises(ValueError):
            document_from_mapping(
                {
                    "name": "bad curve track",
                    "page": {"size": "A4"},
                    "depth": {"unit": "m", "scale": "1:200"},
                    "tracks": [
                        {
                            "id": "bad",
                            "title": "Bad",
                            "kind": "curve",
                            "width_mm": 40,
                            "elements": [{"kind": "raster", "channel": "FMI"}],
                        }
                    ],
                }
            )

    def test_invalid_element_kind_raises_template_error(self) -> None:
        with self.assertRaises(TemplateValidationError):
            document_from_mapping(
                {
                    "name": "bad",
                    "page": {"size": "A4"},
                    "depth": {"unit": "m", "scale": "1:200"},
                    "tracks": [
                        {
                            "id": "track",
                            "title": "Track",
                            "kind": "curve",
                            "width_mm": 20,
                            "elements": [{"kind": "unknown", "channel": "GR"}],
                        }
                    ],
                }
            )

    def test_markers_and_zones_parse_from_template(self) -> None:
        document = document_from_mapping(
            {
                "name": "annotations",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "markers": [{"depth": 1002.5, "label": "Top A", "color": "#ff0000"}],
                "zones": [{"top": 1005, "base": 1012, "label": "Zone 1"}],
                "tracks": [{"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16}],
            }
        )
        self.assertEqual(len(document.markers), 1)
        self.assertEqual(document.markers[0].label, "Top A")
        self.assertAlmostEqual(document.markers[0].depth, 1002.5)
        self.assertEqual(len(document.zones), 1)
        self.assertEqual(document.zones[0].label, "Zone 1")
        self.assertAlmostEqual(document.zones[0].top, 1005.0)
        self.assertAlmostEqual(document.zones[0].base, 1012.0)

    def test_invalid_markers_or_zones_raise_template_error(self) -> None:
        with self.assertRaises(TemplateValidationError):
            document_from_mapping(
                {
                    "name": "bad markers",
                    "page": {"size": "A4"},
                    "depth": {"unit": "m", "scale": "1:200"},
                    "markers": {"depth": 1000},
                    "tracks": [{"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16}],
                }
            )
        with self.assertRaises(TemplateValidationError):
            document_from_mapping(
                {
                    "name": "bad zones",
                    "page": {"size": "A4"},
                    "depth": {"unit": "m", "scale": "1:200"},
                    "zones": [{"top": 1010, "base": 1000}],
                    "tracks": [{"id": "depth", "title": "Depth", "kind": "depth", "width_mm": 16}],
                }
            )

    def test_track_header_objects_parse_with_reserved_space(self) -> None:
        document = document_from_mapping(
            {
                "name": "header objects",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "Gamma Ray",
                        "kind": "curve",
                        "width_mm": 30,
                        "track_header": {
                            "objects": [
                                {"kind": "title", "enabled": True, "line_units": 1},
                                {"kind": "scale", "enabled": False, "reserve_space": True},
                                {"kind": "legend", "enabled": True, "line_units": 2},
                                {
                                    "kind": "divisions",
                                    "enabled": True,
                                    "reserve_space": True,
                                    "line_units": 1,
                                },
                            ]
                        },
                        "elements": [{"kind": "curve", "channel": "GR"}],
                    }
                ],
            }
        )
        objects = document.tracks[0].header.objects
        self.assertEqual(len(objects), 4)
        self.assertEqual(objects[0].kind, TrackHeaderObjectKind.TITLE)
        self.assertEqual(objects[1].kind, TrackHeaderObjectKind.SCALE)
        self.assertFalse(objects[1].enabled)
        self.assertTrue(objects[1].reserve_space)
        self.assertEqual(objects[2].line_units, 2)
        self.assertEqual(objects[3].kind, TrackHeaderObjectKind.DIVISIONS)

    def test_invalid_track_header_configuration_raises_template_error(self) -> None:
        with self.assertRaises(TemplateValidationError):
            document_from_mapping(
                {
                    "name": "bad header objects",
                    "page": {"size": "A4"},
                    "depth": {"unit": "m", "scale": "1:200"},
                    "tracks": [
                        {
                            "id": "gr",
                            "title": "Gamma Ray",
                            "kind": "curve",
                            "width_mm": 30,
                            "track_header": {
                                "objects": [
                                    {"kind": "title"},
                                    {"kind": "title"},
                                ]
                            },
                            "elements": [{"kind": "curve", "channel": "GR"}],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
