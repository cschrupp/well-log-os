from __future__ import annotations

import unittest

from well_log_os.errors import TemplateValidationError
from well_log_os.model import (
    AnnotationArrowSpec,
    AnnotationGlyphSpec,
    AnnotationIntervalSpec,
    AnnotationMarkerSpec,
    AnnotationTextSpec,
    CurveElement,
    CurveFillKind,
    GridDisplayMode,
    GridScaleKind,
    GridSpacingMode,
    NumberFormatKind,
    RasterElement,
    ScaleKind,
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
        self.assertFalse(element.header_display.wrap_name)
        self.assertFalse(element.wrap)

    def test_curve_element_can_enable_header_name_wrap(self) -> None:
        document = document_from_mapping(
            {
                "name": "header name wrap",
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
                                "header_display": {
                                    "wrap_name": True,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertTrue(element.header_display.wrap_name)

    def test_curve_element_can_parse_callouts(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve callouts",
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
                                "callouts": [
                                    {
                                        "depth": 1005,
                                        "label": "GR Sand",
                                        "side": "right",
                                        "placement": "top",
                                        "text_x": 0.76,
                                        "depth_offset": -2,
                                        "distance_from_top": 1.5,
                                        "distance_from_bottom": 2.5,
                                        "every": 5,
                                        "font_size": 7.2,
                                        "arrow": True,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        self.assertEqual(len(element.callouts), 1)
        self.assertEqual(element.callouts[0].label, "GR Sand")
        self.assertEqual(element.callouts[0].side, "right")
        self.assertEqual(element.callouts[0].placement, "top")
        self.assertAlmostEqual(element.callouts[0].text_x or 0.0, 0.76)
        self.assertAlmostEqual(element.callouts[0].distance_from_top or 0.0, 1.5)
        self.assertAlmostEqual(element.callouts[0].distance_from_bottom or 0.0, 2.5)
        self.assertAlmostEqual(element.callouts[0].every or 0.0, 5.0)

    def test_document_can_parse_report_block(self) -> None:
        document = document_from_mapping(
            {
                "name": "report block",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "header": {
                    "report": {
                        "provider_name": "Company",
                        "general_fields": [
                            {"key": "company", "label": "Company", "value": "University of Utah"},
                            {"key": "well", "label": "Well", "source_key": "WELL"},
                        ],
                        "service_titles": [
                            {
                                "value": "Cement Bond Log",
                                "font_size": 15.0,
                                "auto_adjust": True,
                                "bold": True,
                                "alignment": "center",
                            },
                            {
                                "value": "Variable Density Log",
                                "italic": True,
                                "alignment": "right",
                            },
                        ],
                        "detail": {
                            "kind": "cased_hole",
                            "column_titles": ["Measured", "Recorded"],
                            "rows": [
                                {
                                    "label_cells": ["Run", "Down"],
                                    "columns": [
                                        {"cells": ["06-Oct-2021"]},
                                        {
                                            "cells": [
                                                {
                                                    "value": "06-Oct-2021",
                                                    "background_color": "#fff46b",
                                                    "text_color": "#cc0000",
                                                    "font_weight": "bold",
                                                }
                                            ]
                                        },
                                    ],
                                },
                                {"label": "Bottom Log Interval", "values": ["8540 ft", "8540 ft"]},
                            ],
                        },
                        "tail_enabled": True,
                    }
                },
                "tracks": [
                    {
                        "id": "depth",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "reference": {"define_layout": True, "unit": "m", "scale_ratio": 200},
                    }
                ],
            }
        )
        self.assertIsNotNone(document.header.report)
        assert document.header.report is not None
        self.assertEqual(document.header.report.provider_name, "Company")
        self.assertEqual(document.header.report.service_titles[0].value.value, "Cement Bond Log")
        self.assertAlmostEqual(document.header.report.service_titles[0].font_size or 0.0, 15.0)
        self.assertTrue(document.header.report.service_titles[0].bold)
        self.assertEqual(document.header.report.service_titles[0].alignment, "center")
        self.assertTrue(document.header.report.service_titles[1].italic)
        self.assertEqual(document.header.report.service_titles[1].alignment, "right")
        self.assertEqual(document.header.report.detail.kind.value, "cased_hole")
        self.assertEqual(len(document.header.report.detail.rows[0].label_cells), 2)
        self.assertEqual(
            document.header.report.detail.rows[0].columns[1].cells[0].background_color,
            "#fff46b",
        )
        self.assertTrue(document.header.report.tail_enabled)

    def test_curve_element_can_enable_log_wrap(self) -> None:
        document = document_from_mapping(
            {
                "name": "wrapped log curve",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "rt",
                        "title": "RT",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "RT",
                                "scale": {"kind": "log", "min": 2, "max": 200},
                                "wrap": {"enabled": True, "color": "#ff5500"},
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        self.assertTrue(element.wrap)
        self.assertEqual(element.wrap_color, "#ff5500")

    def test_curve_element_can_parse_between_curves_fill(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve fill",
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
                                "channel": "NPHI",
                                "scale": {"kind": "linear", "min": 0, "max": 0.45},
                                "fill": {
                                    "kind": "between_curves",
                                    "other_channel": "DPHI",
                                    "label": "Gas Effect",
                                    "color": "#22c55e",
                                    "alpha": 0.3,
                                    "crossover": {
                                        "enabled": True,
                                        "left_color": "#22c55e",
                                        "right_color": "#ef4444",
                                    },
                                },
                            },
                            {
                                "kind": "curve",
                                "channel": "DPHI",
                                "scale": {"kind": "linear", "min": 0, "max": 0.45},
                            },
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        assert element.fill is not None
        self.assertEqual(element.fill.kind, CurveFillKind.BETWEEN_CURVES)
        self.assertEqual(element.fill.other_channel, "DPHI")
        self.assertEqual(element.fill.label, "Gas Effect")
        self.assertEqual(element.fill.color, "#22c55e")
        self.assertAlmostEqual(element.fill.alpha or 0.0, 0.3)
        self.assertTrue(element.fill.crossover.enabled)
        self.assertEqual(element.fill.crossover.left_color, "#22c55e")
        self.assertEqual(element.fill.crossover.right_color, "#ef4444")

    def test_curve_element_can_parse_between_instances_fill(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve instance fill",
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
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        self.assertEqual(element.id, "cbl_0_100")
        assert element.fill is not None
        self.assertEqual(element.fill.kind, CurveFillKind.BETWEEN_INSTANCES)
        self.assertEqual(element.fill.other_element_id, "cbl_0_10")

    def test_curve_element_can_parse_limit_fill(self) -> None:
        document = document_from_mapping(
            {
                "name": "curve limit fill",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "curve",
                        "title": "Curve",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "fill": {
                                    "kind": "to_lower_limit",
                                    "label": "Shale Fill",
                                    "color": "#22c55e",
                                },
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        assert element.fill is not None
        self.assertEqual(element.fill.kind, CurveFillKind.TO_LOWER_LIMIT)
        self.assertEqual(element.fill.label, "Shale Fill")
        self.assertEqual(element.fill.color, "#22c55e")

    def test_curve_element_can_parse_baseline_split_fill(self) -> None:
        document = document_from_mapping(
            {
                "name": "baseline split fill",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "curve",
                        "title": "Curve",
                        "kind": "normal",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "NPHI",
                                "fill": {
                                    "kind": "baseline_split",
                                    "label": "Gas Effect",
                                    "alpha": 0.35,
                                    "baseline": {
                                        "value": 0.15,
                                        "lower_color": "#22c55e",
                                        "upper_color": "#ef4444",
                                        "line_color": "#222222",
                                        "line_width": 0.7,
                                        "line_style": ":",
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        assert element.fill is not None
        self.assertEqual(element.fill.kind, CurveFillKind.BASELINE_SPLIT)
        assert element.fill.baseline is not None
        self.assertAlmostEqual(element.fill.baseline.value, 0.15)
        self.assertEqual(element.fill.baseline.lower_color, "#22c55e")
        self.assertEqual(element.fill.baseline.upper_color, "#ef4444")
        self.assertEqual(element.fill.baseline.line_color, "#222222")

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

    def test_reference_track_can_parse_reference_curve_overlay(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference overlay config",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "ref",
                        "title": "Reference",
                        "kind": "reference",
                        "width_mm": 20,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "CBL",
                                "reference_overlay": {
                                    "mode": "indicator",
                                    "lane_start": 0.7,
                                    "lane_end": 0.9,
                                    "tick_side": "right",
                                    "tick_length_ratio": 0.15,
                                    "threshold": 1.5,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        self.assertIsNotNone(element.reference_overlay)
        assert element.reference_overlay is not None
        self.assertEqual(element.reference_overlay.mode, "indicator")
        self.assertAlmostEqual(element.reference_overlay.lane_start or 0.0, 0.7)
        self.assertAlmostEqual(element.reference_overlay.lane_end or 0.0, 0.9)
        self.assertEqual(element.reference_overlay.tick_side, "right")
        self.assertAlmostEqual(element.reference_overlay.tick_length_ratio or 0.0, 0.15)
        self.assertAlmostEqual(element.reference_overlay.threshold or 0.0, 1.5)

    def test_reference_track_can_parse_reference_events(self) -> None:
        document = document_from_mapping(
            {
                "name": "reference events",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "ref",
                        "title": "Reference",
                        "kind": "reference",
                        "width_mm": 20,
                        "reference": {
                            "events": [
                                {
                                    "depth": 1001.5,
                                    "label": "Casing Foot",
                                    "tick_side": "left",
                                    "tick_length_ratio": 0.14,
                                    "text_side": "right",
                                    "text_x": 0.3,
                                }
                            ]
                        },
                    }
                ],
            }
        )
        reference = document.tracks[0].reference
        assert reference is not None
        self.assertEqual(len(reference.events), 1)
        event = reference.events[0]
        self.assertAlmostEqual(event.depth, 1001.5)
        self.assertEqual(event.label, "Casing Foot")
        self.assertEqual(event.tick_side, "left")
        self.assertAlmostEqual(event.tick_length_ratio or 0.0, 0.14)
        self.assertEqual(event.text_side, "right")
        self.assertAlmostEqual(event.text_x or 0.0, 0.3)

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
                            {
                                "kind": "raster",
                                "channel": "FMI",
                                "label": "VDL VariableDensity",
                                "profile": "vdl",
                                "normalization": "trace_maxabs",
                                "waveform_normalization": "trace_maxabs",
                                "clip_percentiles": [1, 99],
                                "show_raster": True,
                                "raster_alpha": 0.45,
                                "colorbar": {
                                    "enabled": True,
                                    "label": "VDL amp",
                                    "position": "header",
                                },
                                "sample_axis": {
                                    "enabled": True,
                                    "label": "Azimuth (deg)",
                                    "unit": "deg",
                                    "ticks": 7,
                                    "source_origin": 40,
                                    "source_step": 10,
                                    "min": 200,
                                    "max": 1200,
                                },
                                "waveform": {
                                    "enabled": True,
                                    "stride": 5,
                                    "amplitude_scale": 0.4,
                                    "color": "#663399",
                                    "line_width": 0.25,
                                    "fill": True,
                                    "positive_fill_color": "#000000",
                                    "negative_fill_color": "#ffffff",
                                    "invert_fill_polarity": False,
                                    "max_traces": 300,
                                },
                            },
                            {"kind": "curve", "channel": "FRACTURE_INTENSITY"},
                        ],
                    },
                ],
            }
        )
        image_track = document.tracks[1]
        self.assertEqual(image_track.kind, TrackKind.IMAGE)
        self.assertIsInstance(image_track.elements[0], RasterElement)
        self.assertEqual(image_track.elements[0].label, "VDL VariableDensity")
        self.assertEqual(image_track.elements[0].profile, "vdl")
        self.assertEqual(image_track.elements[0].normalization, "trace_maxabs")
        self.assertEqual(image_track.elements[0].waveform_normalization, "trace_maxabs")
        self.assertEqual(image_track.elements[0].clip_percentiles, (1.0, 99.0))
        self.assertTrue(image_track.elements[0].show_raster)
        self.assertEqual(image_track.elements[0].raster_alpha, 0.45)
        self.assertTrue(image_track.elements[0].colorbar_enabled)
        self.assertEqual(image_track.elements[0].colorbar_label, "VDL amp")
        self.assertEqual(image_track.elements[0].colorbar_position, "header")
        self.assertTrue(image_track.elements[0].sample_axis_enabled)
        self.assertEqual(image_track.elements[0].sample_axis_label, "Azimuth (deg)")
        self.assertEqual(image_track.elements[0].sample_axis_unit, "deg")
        self.assertEqual(image_track.elements[0].sample_axis_tick_count, 7)
        self.assertEqual(image_track.elements[0].sample_axis_source_origin, 40.0)
        self.assertEqual(image_track.elements[0].sample_axis_source_step, 10.0)
        self.assertEqual(image_track.elements[0].sample_axis_min, 200.0)
        self.assertEqual(image_track.elements[0].sample_axis_max, 1200.0)
        self.assertTrue(image_track.elements[0].waveform.enabled)
        self.assertEqual(image_track.elements[0].waveform.stride, 5)
        self.assertEqual(image_track.elements[0].waveform.amplitude_scale, 0.4)
        self.assertEqual(image_track.elements[0].waveform.color, "#663399")
        self.assertEqual(image_track.elements[0].waveform.line_width, 0.25)
        self.assertTrue(image_track.elements[0].waveform.fill)
        self.assertEqual(image_track.elements[0].waveform.positive_fill_color, "#000000")
        self.assertEqual(image_track.elements[0].waveform.negative_fill_color, "#ffffff")
        self.assertFalse(image_track.elements[0].waveform.invert_fill_polarity)
        self.assertEqual(image_track.elements[0].waveform.max_traces, 300)
        self.assertIsInstance(image_track.elements[1], CurveElement)

    def test_waveform_profile_defaults_to_waveform_only_mode(self) -> None:
        document = document_from_mapping(
            {
                "name": "waveform profile",
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
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, RasterElement)
        self.assertEqual(element.profile, "waveform")
        self.assertFalse(element.show_raster)
        self.assertTrue(element.waveform.enabled)

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

    def test_annotation_track_can_parse_interval_and_text_objects(self) -> None:
        document = document_from_mapping(
            {
                "name": "annotation objects",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "lith",
                        "title": "Lithofacies",
                        "kind": "annotation",
                        "width_mm": 20,
                        "annotations": [
                            {
                                "kind": "interval",
                                "top": 1000,
                                "base": 1010,
                                "text": "shale",
                                "fill_color": "#2047a3",
                                "text_orientation": "vertical",
                            },
                            {
                                "kind": "text",
                                "top": 1010,
                                "base": 1015,
                                "text": "Detailed zone description",
                                "wrap": True,
                            },
                        ],
                    }
                ],
            }
        )
        track = document.tracks[0]
        self.assertEqual(track.kind, TrackKind.ANNOTATION)
        self.assertEqual(len(track.annotations), 2)
        self.assertIsInstance(track.annotations[0], AnnotationIntervalSpec)
        self.assertEqual(track.annotations[0].text, "shale")
        self.assertIsInstance(track.annotations[1], AnnotationTextSpec)
        self.assertEqual(track.annotations[1].text, "Detailed zone description")

    def test_annotation_track_can_parse_marker_arrow_and_glyph_objects(self) -> None:
        document = document_from_mapping(
            {
                "name": "annotation extra objects",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "ann",
                        "title": "Annotations",
                        "kind": "annotation",
                        "width_mm": 20,
                        "annotations": [
                            {
                                "kind": "marker",
                                "depth": 1005,
                                "x": 0.2,
                                "shape": "triangle_right",
                                "label": "Casing Foot",
                                "priority": 150,
                                "label_mode": "dedicated_lane",
                                "label_lane_start": 0.75,
                                "label_lane_end": 0.95,
                            },
                            {
                                "kind": "arrow",
                                "start_depth": 1008,
                                "end_depth": 1012,
                                "start_x": 0.75,
                                "end_x": 0.35,
                                "label": "Flow",
                            },
                            {
                                "kind": "glyph",
                                "depth": 1014,
                                "glyph": "CF",
                                "lane_start": 0.0,
                                "lane_end": 0.3,
                            },
                        ],
                    }
                ],
            }
        )
        track = document.tracks[0]
        self.assertEqual(len(track.annotations), 3)
        self.assertIsInstance(track.annotations[0], AnnotationMarkerSpec)
        self.assertEqual(track.annotations[0].priority, 150)
        self.assertEqual(track.annotations[0].label_mode.value, "dedicated_lane")
        self.assertIsInstance(track.annotations[1], AnnotationArrowSpec)
        self.assertIsInstance(track.annotations[2], AnnotationGlyphSpec)

    def test_non_annotation_track_rejects_annotation_objects(self) -> None:
        with self.assertRaises(ValueError):
            document_from_mapping(
                {
                    "name": "bad annotation placement",
                    "page": {"size": "A4"},
                    "depth": {"unit": "m", "scale": "1:200"},
                    "tracks": [
                        {
                            "id": "gr",
                            "title": "GR",
                            "kind": "normal",
                            "width_mm": 30,
                            "annotations": [{"kind": "text", "depth": 1000, "text": "bad"}],
                        }
                    ],
                }
            )

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

    def test_track_grid_parses_vertical_and_horizontal_properties(self) -> None:
        document = document_from_mapping(
            {
                "name": "grid options",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "curve",
                        "width_mm": 30,
                        "grid": {
                            "display": "below",
                            "horizontal": {
                                "display": "above",
                                "main": {"visible": True, "thickness": 0.8, "color": "#111111"},
                                "secondary": {"visible": False},
                            },
                            "vertical": {
                                "display": "below",
                                "main": {
                                    "visible": True,
                                    "line_count": 4,
                                    "thickness": 0.7,
                                    "color": "#222222",
                                    "scale": "exponential",
                                    "spacing_mode": "scale",
                                },
                                "secondary": {
                                    "visible": True,
                                    "line_count": 3,
                                    "thickness": 0.4,
                                    "scale": "tangential",
                                    "spacing_mode": "count",
                                },
                            },
                        },
                        "elements": [{"kind": "curve", "channel": "GR"}],
                    }
                ],
            }
        )
        grid = document.tracks[0].grid
        self.assertEqual(grid.horizontal_display, GridDisplayMode.ABOVE)
        self.assertTrue(grid.horizontal_major_visible)
        self.assertFalse(grid.horizontal_minor_visible)
        self.assertAlmostEqual(grid.horizontal_major_thickness or 0.0, 0.8)
        self.assertEqual(grid.vertical_display, GridDisplayMode.BELOW)
        self.assertEqual(grid.vertical_main_line_count, 4)
        self.assertEqual(grid.vertical_secondary_line_count, 3)
        self.assertEqual(grid.vertical_main_scale, GridScaleKind.LOGARITHMIC)
        self.assertEqual(grid.vertical_secondary_scale, GridScaleKind.TANGENTIAL)
        self.assertEqual(grid.vertical_main_spacing_mode, GridSpacingMode.SCALE)
        self.assertEqual(grid.vertical_secondary_spacing_mode, GridSpacingMode.COUNT)

    def test_curve_scale_parses_tangential_alias(self) -> None:
        document = document_from_mapping(
            {
                "name": "tangential curve",
                "page": {"size": "A4"},
                "depth": {"unit": "m", "scale": "1:200"},
                "tracks": [
                    {
                        "id": "gr",
                        "title": "GR",
                        "kind": "curve",
                        "width_mm": 30,
                        "elements": [
                            {
                                "kind": "curve",
                                "channel": "GR",
                                "scale": {"kind": "tangent", "min": 0, "max": 150},
                            }
                        ],
                    }
                ],
            }
        )
        element = document.tracks[0].elements[0]
        self.assertIsInstance(element, CurveElement)
        assert element.scale is not None
        self.assertEqual(element.scale.kind, ScaleKind.TANGENTIAL)

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
