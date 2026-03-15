from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import yaml

from well_log_os.errors import TemplateValidationError
from well_log_os.logfile import (
    build_document_for_logfile,
    build_documents_for_logfile,
    load_datasets_for_logfile,
    load_logfile,
    logfile_from_mapping,
)
from well_log_os.model import (
    AnnotationIntervalSpec,
    AnnotationTextSpec,
    CurveFillKind,
    GridScaleKind,
    RasterChannel,
    ScalarChannel,
    ScaleKind,
    WellDataset,
)


def build_mapping() -> dict:
    return {
        "version": 1,
        "name": "Test Logfile",
        "data": {"source_path": "sample.las", "source_format": "auto"},
        "render": {"backend": "matplotlib", "output_path": "out.pdf", "dpi": 300},
        "document": {
            "name": "{WELL} Layout",
            "page": {"size": "A4", "continuous": True},
            "depth": {"unit": "m", "scale": "1:200", "major_step": 20, "minor_step": 5},
            "header": {
                "title": "{WELL}",
                "subtitle": "{SOURCE_FILENAME}",
                "fields": [{"label": "Well", "source_key": "WELL"}],
            },
            "footer": {"lines": ["Generated from {SOURCE_FILENAME}"]},
            "markers": [],
            "zones": [],
            "layout": {
                "heading": {"enabled": True},
                "comments": [],
                "log_sections": [
                    {
                        "id": "main",
                        "title": "Main Log Section",
                        "subtitle": "Service Interval",
                        "tracks": [
                            {
                                "id": "gr",
                                "title": "GR",
                                "kind": "normal",
                                "width_mm": 28,
                                "position": 1,
                            },
                            {
                                "id": "depth",
                                "title": "Depth",
                                "kind": "reference",
                                "width_mm": 16,
                                "position": 2,
                                "reference": {
                                    "define_layout": True,
                                    "unit": "m",
                                    "scale_ratio": 200,
                                    "major_step": 20,
                                    "secondary_grid": {"display": True, "line_count": 4},
                                },
                            },
                            {
                                "id": "rt",
                                "title": "RT",
                                "kind": "normal",
                                "width_mm": 28,
                                "position": 3,
                            },
                        ],
                    }
                ],
                "tail": {"enabled": True},
            },
            "bindings": {
                "on_missing": "skip",
                "channels": [
                    {
                        "channel": "GR",
                        "track_id": "gr",
                        "kind": "curve",
                        "style": {"color": "#1b5e20"},
                    },
                    {
                        "channel": "RT",
                        "track_id": "rt",
                        "kind": "curve",
                        "style": {"color": "#0d47a1"},
                    },
                ],
            },
        },
    }


class LogFileTests(unittest.TestCase):
    def build_dataset(self) -> WellDataset:
        depth = np.linspace(1000.0, 1020.0, 50)
        sample_axis = np.linspace(0.0, 360.0, 36)
        raster_values = np.sin(depth[:, None] / 8.0) * np.cos(np.deg2rad(sample_axis))[None, :]
        dataset = WellDataset(name="sample", well_metadata={"WELL": "TEST-1"})
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.linspace(20, 120, depth.size))
        )
        dataset.add_channel(
            ScalarChannel(
                "RT", depth, "m", "ohm.m", values=np.exp(np.linspace(0.1, 5.0, depth.size))
            )
        )
        dataset.add_channel(
            RasterChannel(
                "VDL",
                depth,
                "m",
                "amplitude",
                values=raster_values,
                sample_axis=sample_axis,
                sample_unit="deg",
                sample_label="azimuth",
            )
        )
        return dataset

    def build_dataset_gr_only(self) -> WellDataset:
        depth = np.linspace(1000.0, 1020.0, 50)
        dataset = WellDataset(name="main", well_metadata={"WELL": "MAIN-1"})
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.linspace(20, 120, depth.size))
        )
        return dataset

    def build_dataset_rt_only(self) -> WellDataset:
        depth = np.linspace(1000.0, 1020.0, 50)
        dataset = WellDataset(name="repeat", well_metadata={"WELL": "REPEAT-1"})
        dataset.add_channel(
            ScalarChannel("RT", depth, "m", "ohm.m", values=np.linspace(2.0, 50.0, depth.size))
        )
        return dataset

    def test_logfile_builds_document_from_layout_bindings(self) -> None:
        spec = logfile_from_mapping(build_mapping())
        self.assertIsNone(spec.render_continuous_strip_page_height_mm)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(document.name, "TEST-1 Layout")
        self.assertEqual(document.header.title, "TEST-1")
        self.assertEqual(document.header.subtitle, "example_input.las")
        self.assertEqual(len(document.tracks), 3)
        self.assertEqual(document.tracks[0].id, "gr")
        self.assertEqual(document.tracks[1].id, "depth")
        self.assertEqual(document.tracks[2].id, "rt")
        self.assertEqual(document.tracks[0].elements[0].channel, "GR")
        self.assertEqual(document.tracks[2].elements[0].channel, "RT")
        self.assertIn("layout_sections", document.metadata)
        active_section = document.metadata["layout_sections"]["active_section"]
        self.assertEqual(active_section["title"], "Main Log Section")
        self.assertEqual(active_section["subtitle"], "Service Interval")

    def test_logfile_builds_multisection_documents(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"] = [
            {
                "id": "main",
                "title": "Main Section",
                "tracks": [
                    {
                        "id": "depth_main",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "position": 1,
                        "reference": {"define_layout": True, "unit": "m", "scale_ratio": 200},
                    },
                    {
                        "id": "gr_main",
                        "title": "GR",
                        "kind": "normal",
                        "width_mm": 28,
                        "position": 2,
                    },
                ],
            },
            {
                "id": "aux",
                "title": "Aux Section",
                "tracks": [
                    {
                        "id": "depth_aux",
                        "title": "Depth",
                        "kind": "reference",
                        "width_mm": 16,
                        "position": 1,
                        "reference": {"define_layout": True, "unit": "m", "scale_ratio": 200},
                    },
                    {
                        "id": "rt_aux",
                        "title": "RT",
                        "kind": "normal",
                        "width_mm": 28,
                        "position": 2,
                    },
                ],
            },
        ]
        payload["document"]["bindings"]["channels"] = [
            {"channel": "GR", "track_id": "gr_main", "kind": "curve"},
            {"channel": "RT", "track_id": "rt_aux", "kind": "curve"},
        ]
        spec = logfile_from_mapping(payload)
        documents = build_documents_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["layout_sections"]["active_section"]["id"], "main")
        self.assertEqual(documents[1].metadata["layout_sections"]["active_section"]["id"], "aux")
        self.assertEqual([track.id for track in documents[0].tracks], ["depth_main", "gr_main"])
        self.assertEqual([track.id for track in documents[1].tracks], ["depth_aux", "rt_aux"])
        self.assertEqual(documents[0].tracks[1].elements[0].channel, "GR")
        self.assertEqual(documents[1].tracks[1].elements[0].channel, "RT")

    def test_logfile_requires_section_for_ambiguous_track_id_bindings(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"] = [
            {
                "id": "main",
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "reference", "width_mm": 16},
                    {"id": "curve", "title": "Curve", "kind": "normal", "width_mm": 28},
                ],
            },
            {
                "id": "aux",
                "tracks": [
                    {"id": "depth", "title": "Depth", "kind": "reference", "width_mm": 16},
                    {"id": "curve", "title": "Curve", "kind": "normal", "width_mm": 28},
                ],
            },
        ]
        payload["document"]["bindings"]["channels"] = [
            {"channel": "GR", "track_id": "curve", "kind": "curve"},
        ]
        spec = logfile_from_mapping(payload)
        with self.assertRaises(TemplateValidationError):
            build_documents_for_logfile(
                spec,
                self.build_dataset(),
                source_path=Path("example_input.las"),
            )

    def test_multisection_build_uses_section_specific_datasets(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"] = [
            {
                "id": "main",
                "tracks": [
                    {"id": "depth_main", "title": "Depth", "kind": "reference", "width_mm": 16},
                    {"id": "gr_main", "title": "GR", "kind": "normal", "width_mm": 28},
                ],
            },
            {
                "id": "repeat",
                "tracks": [
                    {"id": "depth_repeat", "title": "Depth", "kind": "reference", "width_mm": 16},
                    {"id": "rt_repeat", "title": "RT", "kind": "normal", "width_mm": 28},
                ],
            },
        ]
        payload["document"]["bindings"]["channels"] = [
            {"section": "main", "channel": "GR", "track_id": "gr_main", "kind": "curve"},
            {"section": "repeat", "channel": "RT", "track_id": "rt_repeat", "kind": "curve"},
        ]
        spec = logfile_from_mapping(payload)
        documents = build_documents_for_logfile(
            spec,
            {
                "main": self.build_dataset_gr_only(),
                "repeat": self.build_dataset_rt_only(),
            },
            source_path={
                "main": Path("main_run.las"),
                "repeat": Path("repeat_run.las"),
            },
        )
        self.assertEqual(documents[0].tracks[1].elements[0].channel, "GR")
        self.assertEqual(documents[1].tracks[1].elements[0].channel, "RT")
        self.assertEqual(documents[0].header.subtitle, "main_run.las")
        self.assertEqual(documents[1].header.subtitle, "repeat_run.las")

    @patch("well_log_os.logfile.load_las")
    def test_load_datasets_for_logfile_supports_section_data_overrides(self, mock_load_las) -> None:
        payload = build_mapping()
        payload["data"] = {"source_path": "main.las", "source_format": "las"}
        payload["document"]["layout"]["log_sections"] = [
            {
                "id": "main",
                "tracks": [
                    {"id": "depth_main", "title": "Depth", "kind": "reference", "width_mm": 16}
                ],
            },
            {
                "id": "repeat",
                "data": {"source_path": "repeat.las", "source_format": "las"},
                "tracks": [
                    {"id": "depth_repeat", "title": "Depth", "kind": "reference", "width_mm": 16}
                ],
            },
        ]
        spec = logfile_from_mapping(payload)
        dataset_main = WellDataset(name="main")
        dataset_repeat = WellDataset(name="repeat")
        mock_load_las.side_effect = [dataset_main, dataset_repeat]

        datasets_by_section, source_paths_by_section = load_datasets_for_logfile(
            spec,
            base_dir=Path("/tmp/project"),
        )

        self.assertEqual(datasets_by_section["main"], dataset_main)
        self.assertEqual(datasets_by_section["repeat"], dataset_repeat)
        self.assertEqual(source_paths_by_section["main"], Path("/tmp/project/main.las").resolve())
        self.assertEqual(
            source_paths_by_section["repeat"],
            Path("/tmp/project/repeat.las").resolve(),
        )

    @patch("well_log_os.logfile.load_las")
    def test_load_datasets_for_logfile_supports_section_first_sources(self, mock_load_las) -> None:
        payload = build_mapping()
        payload.pop("data")
        payload["document"]["layout"]["log_sections"] = [
            {
                "id": "main",
                "data": {"source_path": "main.las", "source_format": "las"},
                "tracks": [
                    {"id": "depth_main", "title": "Depth", "kind": "reference", "width_mm": 16}
                ],
            },
            {
                "id": "repeat",
                "data": {"source_path": "repeat.las", "source_format": "las"},
                "tracks": [
                    {"id": "depth_repeat", "title": "Depth", "kind": "reference", "width_mm": 16}
                ],
            },
        ]
        spec = logfile_from_mapping(payload)
        dataset_main = WellDataset(name="main")
        dataset_repeat = WellDataset(name="repeat")
        mock_load_las.side_effect = [dataset_main, dataset_repeat]

        datasets_by_section, source_paths_by_section = load_datasets_for_logfile(
            spec,
            base_dir=Path("/tmp/project"),
        )

        self.assertEqual(datasets_by_section["main"], dataset_main)
        self.assertEqual(datasets_by_section["repeat"], dataset_repeat)
        self.assertEqual(source_paths_by_section["main"], Path("/tmp/project/main.las").resolve())
        self.assertEqual(
            source_paths_by_section["repeat"],
            Path("/tmp/project/repeat.las").resolve(),
        )

    def test_missing_data_sources_raise_when_root_data_absent(self) -> None:
        payload = build_mapping()
        payload.pop("data")
        spec = logfile_from_mapping(payload)
        with self.assertRaises(TemplateValidationError):
            load_datasets_for_logfile(spec)

    def test_invalid_logfile_configuration_raises(self) -> None:
        payload = build_mapping()
        del payload["document"]["bindings"]["channels"][0]["track_id"]
        with self.assertRaises(TemplateValidationError):
            logfile_from_mapping(payload)

    def test_layout_and_bindings_are_required(self) -> None:
        payload = build_mapping()
        del payload["document"]["layout"]
        with self.assertRaises(TemplateValidationError):
            logfile_from_mapping(payload)

    def test_schema_error_reports_invalid_dpi_path(self) -> None:
        payload = build_mapping()
        payload["render"]["dpi"] = 0
        with self.assertRaises(TemplateValidationError) as ctx:
            logfile_from_mapping(payload)
        self.assertIn("$.render.dpi", str(ctx.exception))

    def test_logfile_parses_continuous_strip_page_height(self) -> None:
        payload = build_mapping()
        payload["render"]["continuous_strip_page_height_mm"] = 279.4
        spec = logfile_from_mapping(payload)
        self.assertEqual(spec.render_continuous_strip_page_height_mm, 279.4)

    def test_logfile_parses_matplotlib_style_overrides(self) -> None:
        payload = build_mapping()
        payload["render"]["matplotlib"] = {
            "style": {
                "track": {"x_tick_labelsize": 7.5},
                "track_header": {"background_color": "#ffffff"},
            }
        }
        spec = logfile_from_mapping(payload)
        self.assertEqual(spec.render_matplotlib["style"]["track"]["x_tick_labelsize"], 7.5)
        self.assertEqual(
            spec.render_matplotlib["style"]["track_header"]["background_color"],
            "#ffffff",
        )

    def test_reference_track_config_controls_layout_axis(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][1]["reference"] = {
            "define_layout": True,
            "unit": "ft",
            "scale_ratio": 500,
            "major_step": 50,
            "secondary_grid": {"display": True, "line_count": 5},
            "number_format": {"format": "automatic", "precision": 1},
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(document.depth_axis.unit, "ft")
        self.assertEqual(document.depth_axis.scale_ratio, 500)
        self.assertEqual(document.depth_axis.major_step, 50.0)
        self.assertAlmostEqual(document.depth_axis.minor_step, 10.0)

    def test_track_positions_allow_reordering_in_layout(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["position"] = 2
        payload["document"]["layout"]["log_sections"][0]["tracks"][1]["position"] = 1
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(document.tracks[0].id, "depth")
        self.assertEqual(document.tracks[1].id, "gr")

    def test_binding_can_render_curve_values_as_labels(self) -> None:
        payload = build_mapping()
        payload["document"]["bindings"]["channels"][0]["render_mode"] = "value_labels"
        payload["document"]["bindings"]["channels"][0]["value_labels"] = {
            "step": 5,
            "format": "fixed",
            "precision": 1,
            "font_size": 6,
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertEqual(curve.render_mode, "value_labels")
        self.assertEqual(curve.value_labels.step, 5.0)

    def test_binding_can_parse_curve_callouts(self) -> None:
        payload = build_mapping()
        payload["document"]["bindings"]["channels"][0]["callouts"] = [
            {
                "depth": 1005,
                "label": "GR Sand",
                "side": "right",
                "placement": "bottom",
                "text_x": 0.78,
                "depth_offset": -2,
                "distance_from_top": 1.0,
                "distance_from_bottom": 2.0,
                "every": 4,
            }
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertEqual(len(curve.callouts), 1)
        self.assertEqual(curve.callouts[0].label, "GR Sand")
        self.assertEqual(curve.callouts[0].side, "right")
        self.assertEqual(curve.callouts[0].placement, "bottom")
        self.assertAlmostEqual(curve.callouts[0].text_x or 0.0, 0.78)
        self.assertAlmostEqual(curve.callouts[0].distance_from_top or 0.0, 1.0)
        self.assertAlmostEqual(curve.callouts[0].distance_from_bottom or 0.0, 2.0)
        self.assertAlmostEqual(curve.callouts[0].every or 0.0, 4.0)

    def test_binding_can_parse_reference_curve_overlay(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["kind"] = "reference"
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["reference"] = {
            "define_layout": True,
            "unit": "m",
            "scale_ratio": 200,
        }
        payload["document"]["bindings"]["channels"][0]["reference_overlay"] = {
            "mode": "ticks",
            "tick_side": "right",
            "tick_length_ratio": 0.18,
            "threshold": 5.0,
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertIsNotNone(curve.reference_overlay)
        assert curve.reference_overlay is not None
        self.assertEqual(curve.reference_overlay.mode, "ticks")
        self.assertEqual(curve.reference_overlay.tick_side, "right")
        self.assertAlmostEqual(curve.reference_overlay.tick_length_ratio or 0.0, 0.18)
        self.assertAlmostEqual(curve.reference_overlay.threshold or 0.0, 5.0)

    def test_reference_track_can_parse_reference_events(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["kind"] = "reference"
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["reference"] = {
            "define_layout": True,
            "unit": "m",
            "scale_ratio": 200,
            "events": [
                {
                    "depth": 1002.0,
                    "label": "Readings Start",
                    "tick_side": "right",
                    "tick_length_ratio": 0.16,
                    "text_side": "left",
                    "text_x": 0.72,
                }
            ],
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        reference = document.tracks[0].reference
        assert reference is not None
        self.assertEqual(len(reference.events), 1)
        event = reference.events[0]
        self.assertAlmostEqual(event.depth, 1002.0)
        self.assertEqual(event.label, "Readings Start")
        self.assertEqual(event.tick_side, "right")
        self.assertAlmostEqual(event.tick_length_ratio or 0.0, 0.16)
        self.assertEqual(event.text_side, "left")
        self.assertAlmostEqual(event.text_x or 0.0, 0.72)

    def test_binding_can_enable_log_wrap(self) -> None:
        payload = build_mapping()
        payload["document"]["bindings"]["channels"][1]["scale"] = {
            "kind": "log",
            "min": 2,
            "max": 200,
        }
        payload["document"]["bindings"]["channels"][1]["wrap"] = {
            "enabled": True,
            "color": "#ff5500",
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[2].elements[0]
        self.assertEqual(curve.scale.kind, ScaleKind.LOG)
        self.assertTrue(curve.wrap)
        self.assertEqual(curve.wrap_color, "#ff5500")

    def test_binding_can_parse_between_curves_fill(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"] = [
            {
                "id": "porosity",
                "title": "Porosity",
                "kind": "normal",
                "width_mm": 28,
                "position": 1,
            }
        ]
        payload["document"]["bindings"]["channels"] = [
            {
                "channel": "GR",
                "track_id": "porosity",
                "kind": "curve",
                "label": "NPHI",
                "scale": {"kind": "linear", "min": 0, "max": 150},
                "fill": {
                    "kind": "between_curves",
                    "other_channel": "RT",
                    "label": "Gas Effect",
                    "color": "#22c55e",
                    "alpha": 0.25,
                    "crossover": {
                        "enabled": True,
                        "left_color": "#22c55e",
                        "right_color": "#ef4444",
                    },
                },
            },
            {
                "channel": "RT",
                "track_id": "porosity",
                "kind": "curve",
                "label": "DPHI",
                "scale": {"kind": "linear", "min": 0, "max": 150},
            },
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        assert curve.fill is not None
        self.assertEqual(curve.fill.kind, CurveFillKind.BETWEEN_CURVES)
        self.assertEqual(curve.fill.other_channel, "RT")
        self.assertEqual(curve.fill.label, "Gas Effect")
        self.assertEqual(curve.fill.color, "#22c55e")
        self.assertTrue(curve.fill.crossover.enabled)
        self.assertEqual(curve.fill.crossover.left_color, "#22c55e")
        self.assertEqual(curve.fill.crossover.right_color, "#ef4444")

    def test_binding_can_parse_between_instances_fill(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"] = [
            {
                "id": "cbl",
                "title": "CBL",
                "kind": "normal",
                "width_mm": 28,
                "position": 1,
            }
        ]
        payload["document"]["bindings"]["channels"] = [
            {
                "id": "cbl_0_100",
                "channel": "GR",
                "track_id": "cbl",
                "kind": "curve",
                "scale": {"kind": "linear", "min": 0, "max": 150},
                "fill": {
                    "kind": "between_instances",
                    "other_element_id": "cbl_0_10",
                    "color": "#d1d5db",
                },
            },
            {
                "id": "cbl_0_10",
                "channel": "GR",
                "track_id": "cbl",
                "kind": "curve",
                "scale": {"kind": "linear", "min": 0, "max": 10},
            },
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertEqual(curve.id, "cbl_0_100")
        assert curve.fill is not None
        self.assertEqual(curve.fill.kind, CurveFillKind.BETWEEN_INSTANCES)
        self.assertEqual(curve.fill.other_element_id, "cbl_0_10")

    def test_binding_can_parse_limit_fill(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"] = [
            {
                "id": "gr_fill",
                "title": "GR",
                "kind": "normal",
                "width_mm": 28,
                "position": 1,
            }
        ]
        payload["document"]["bindings"]["channels"] = [
            {
                "channel": "GR",
                "track_id": "gr_fill",
                "kind": "curve",
                "fill": {
                    "kind": "to_upper_limit",
                    "label": "Sand Fill",
                    "color": "#f59e0b",
                },
            }
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        assert curve.fill is not None
        self.assertEqual(curve.fill.kind, CurveFillKind.TO_UPPER_LIMIT)
        self.assertEqual(curve.fill.label, "Sand Fill")
        self.assertEqual(curve.fill.color, "#f59e0b")

    def test_binding_can_parse_baseline_split_fill(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"] = [
            {
                "id": "porosity",
                "title": "Porosity",
                "kind": "normal",
                "width_mm": 28,
                "position": 1,
            }
        ]
        payload["document"]["bindings"]["channels"] = [
            {
                "channel": "GR",
                "track_id": "porosity",
                "kind": "curve",
                "fill": {
                    "kind": "baseline_split",
                    "label": "Gas Effect",
                    "alpha": 0.3,
                    "baseline": {
                        "value": 70,
                        "lower_color": "#22c55e",
                        "upper_color": "#ef4444",
                        "line_color": "#111111",
                        "line_width": 0.5,
                    },
                },
            }
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        assert curve.fill is not None
        self.assertEqual(curve.fill.kind, CurveFillKind.BASELINE_SPLIT)
        assert curve.fill.baseline is not None
        self.assertAlmostEqual(curve.fill.baseline.value, 70.0)
        self.assertEqual(curve.fill.baseline.lower_color, "#22c55e")
        self.assertEqual(curve.fill.baseline.upper_color, "#ef4444")

    def test_raster_binding_parses_array_display_options(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][2]["kind"] = "array"
        payload["document"]["bindings"]["channels"] = [
            {
                "channel": "VDL",
                "track_id": "rt",
                "kind": "raster",
                "label": "VDL VariableDensity",
                "profile": "vdl",
                "normalization": "trace_maxabs",
                "waveform_normalization": "trace_maxabs",
                "clip_percentiles": [1, 99],
                "show_raster": True,
                "raster_alpha": 0.45,
                "style": {"colormap": "bone"},
                "color_limits": [-1.0, 1.0],
                "colorbar": {"enabled": True, "label": "VDL amp", "position": "header"},
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
                    "amplitude_scale": 0.45,
                    "color": "#663399",
                    "line_width": 0.22,
                    "fill": True,
                    "positive_fill_color": "#000000",
                    "negative_fill_color": "#ffffff",
                    "invert_fill_polarity": False,
                    "max_traces": 250,
                },
            }
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        raster = document.tracks[2].elements[0]
        self.assertEqual(raster.label, "VDL VariableDensity")
        self.assertEqual(raster.profile, "vdl")
        self.assertEqual(raster.normalization, "trace_maxabs")
        self.assertEqual(raster.waveform_normalization, "trace_maxabs")
        self.assertEqual(raster.clip_percentiles, (1.0, 99.0))
        self.assertTrue(raster.show_raster)
        self.assertEqual(raster.raster_alpha, 0.45)
        self.assertTrue(raster.colorbar_enabled)
        self.assertEqual(raster.colorbar_label, "VDL amp")
        self.assertEqual(raster.colorbar_position, "header")
        self.assertTrue(raster.sample_axis_enabled)
        self.assertEqual(raster.sample_axis_label, "Azimuth (deg)")
        self.assertEqual(raster.sample_axis_unit, "deg")
        self.assertEqual(raster.sample_axis_tick_count, 7)
        self.assertEqual(raster.sample_axis_source_origin, 40.0)
        self.assertEqual(raster.sample_axis_source_step, 10.0)
        self.assertEqual(raster.sample_axis_min, 200.0)
        self.assertEqual(raster.sample_axis_max, 1200.0)
        self.assertTrue(raster.waveform.enabled)
        self.assertEqual(raster.waveform.stride, 5)
        self.assertEqual(raster.waveform.amplitude_scale, 0.45)
        self.assertEqual(raster.waveform.color, "#663399")
        self.assertEqual(raster.waveform.line_width, 0.22)
        self.assertTrue(raster.waveform.fill)
        self.assertEqual(raster.waveform.positive_fill_color, "#000000")
        self.assertEqual(raster.waveform.negative_fill_color, "#ffffff")
        self.assertFalse(raster.waveform.invert_fill_polarity)
        self.assertEqual(raster.waveform.max_traces, 250)
        self.assertEqual(raster.color_limits, (-1.0, 1.0))

    def test_waveform_profile_defaults_to_waveform_only_mode(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][2]["kind"] = "array"
        payload["document"]["bindings"]["channels"] = [
            {
                "channel": "VDL",
                "track_id": "rt",
                "kind": "raster",
                "profile": "waveform",
            }
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        raster = document.tracks[2].elements[0]
        self.assertEqual(raster.profile, "waveform")
        self.assertFalse(raster.show_raster)
        self.assertTrue(raster.waveform.enabled)

    def test_logfile_parses_track_grid_scale_modes(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["grid"] = {
            "vertical": {
                "main": {"line_count": 4, "scale": "exponential"},
                "secondary": {"line_count": 3, "scale": "tangential"},
            }
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        grid = document.tracks[0].grid
        self.assertEqual(grid.vertical_main_scale, GridScaleKind.LOGARITHMIC)
        self.assertEqual(grid.vertical_secondary_scale, GridScaleKind.TANGENTIAL)

    def test_logfile_parses_tangential_curve_scale_kind(self) -> None:
        payload = build_mapping()
        payload["document"]["bindings"]["channels"][0]["scale"] = {
            "kind": "tangential",
            "min": 0,
            "max": 150,
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertEqual(curve.scale.kind, ScaleKind.TANGENTIAL)

    def test_binding_can_configure_curve_header_display(self) -> None:
        payload = build_mapping()
        payload["document"]["bindings"]["channels"][0]["header_display"] = {
            "show_name": False,
            "show_unit": False,
            "show_limits": True,
            "show_color": False,
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertFalse(curve.header_display.show_name)
        self.assertFalse(curve.header_display.show_unit)
        self.assertTrue(curve.header_display.show_limits)
        self.assertFalse(curve.header_display.show_color)
        self.assertFalse(curve.header_display.wrap_name)

    def test_binding_can_enable_curve_header_name_wrap(self) -> None:
        payload = build_mapping()
        payload["document"]["bindings"]["channels"][0]["header_display"] = {
            "wrap_name": True,
        }
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        curve = document.tracks[0].elements[0]
        self.assertTrue(curve.header_display.wrap_name)

    def test_layout_can_parse_annotation_track_objects(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"].append(
            {
                "id": "lith",
                "title": "Lithofacies",
                "kind": "annotation",
                "width_mm": 18,
                "position": 4,
                "annotations": [
                    {
                        "kind": "interval",
                        "top": 1000,
                        "base": 1010,
                        "text": "shale",
                    },
                    {
                        "kind": "text",
                        "depth": 1005,
                        "text": "laminated",
                    },
                ],
            }
        )
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        track = document.tracks[3]
        self.assertEqual(track.kind.value, "annotation")
        self.assertEqual(len(track.annotations), 2)
        self.assertIsInstance(track.annotations[0], AnnotationIntervalSpec)
        self.assertIsInstance(track.annotations[1], AnnotationTextSpec)

    def test_non_annotation_track_rejects_annotation_objects_in_logfile(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"][0]["annotations"] = [
            {"kind": "text", "depth": 1000, "text": "bad"}
        ]
        with self.assertRaises(TemplateValidationError):
            logfile_from_mapping(payload)

    def test_bindings_can_group_multiple_curves_in_one_track(self) -> None:
        payload = build_mapping()
        payload["document"]["layout"]["log_sections"][0]["tracks"] = [
            {
                "id": "depth",
                "title": "Depth",
                "kind": "reference",
                "width_mm": 16,
                "position": 2,
                "reference": {"define_layout": True, "unit": "m", "scale_ratio": 200},
            },
            {
                "id": "combo",
                "title": "GR / RT",
                "kind": "normal",
                "width_mm": 28,
                "position": 1,
            },
        ]
        payload["document"]["bindings"]["channels"] = [
            {"channel": "GR", "track_id": "combo", "kind": "curve", "style": {"color": "#1b5e20"}},
            {"channel": "RT", "track_id": "combo", "kind": "curve", "style": {"color": "#0d47a1"}},
        ]
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(len(document.tracks), 2)
        combo = document.tracks[0]
        self.assertEqual(combo.id, "combo")
        self.assertEqual(combo.title, "GR / RT")
        self.assertEqual(len(combo.elements), 2)
        self.assertIsNone(combo.x_scale)
        self.assertEqual(combo.elements[0].channel, "GR")
        self.assertEqual(combo.elements[1].channel, "RT")

    def test_page_spacing_fields_are_supported_in_logfile_yaml(self) -> None:
        payload = build_mapping()
        payload["document"]["page"]["margin_left_mm"] = 2.5
        payload["document"]["page"]["track_gap_mm"] = 1.25
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertAlmostEqual(document.page.margin_left_mm, 2.5)
        self.assertAlmostEqual(document.page.track_gap_mm, 1.25)

    def test_load_logfile_merges_template_yaml_with_savefile_overrides(self) -> None:
        template_payload = {
            "render": {"backend": "matplotlib", "output_path": "base.pdf", "dpi": 300},
            "document": {
                "name": "{WELL} Template",
                "page": {"size": "A4", "continuous": True},
                "depth": {"unit": "m", "scale": "1:200"},
                "header": {"title": "{WELL}", "subtitle": "{SOURCE_FILENAME}", "fields": []},
                "footer": {"lines": []},
                "markers": [],
                "zones": [],
                "layout": {
                    "heading": {"enabled": True},
                    "comments": [],
                    "log_sections": [
                        {
                            "id": "main",
                            "tracks": [
                                {
                                    "id": "depth",
                                    "title": "Depth",
                                    "kind": "reference",
                                    "width_mm": 16,
                                    "position": 1,
                                    "reference": {
                                        "define_layout": True,
                                        "unit": "m",
                                        "scale_ratio": 200,
                                    },
                                }
                            ],
                        }
                    ],
                    "tail": {"enabled": True},
                },
                "bindings": {"on_missing": "skip", "channels": []},
            },
        }
        savefile_payload = {
            "template": {"path": "../templates/base.log.yaml"},
            "version": 1,
            "name": "From Savefile",
            "data": {"source_path": "job.las", "source_format": "auto"},
            "render": {"output_path": "job.pdf", "dpi": 350},
            "document": {
                "layout": {
                    "log_sections": [
                        {
                            "id": "main",
                            "tracks": [
                                {
                                    "id": "depth",
                                    "title": "Depth",
                                    "kind": "reference",
                                    "width_mm": 16,
                                    "position": 1,
                                    "reference": {
                                        "define_layout": True,
                                        "unit": "m",
                                        "scale_ratio": 200,
                                    },
                                },
                                {
                                    "id": "combo",
                                    "title": "Combo",
                                    "kind": "normal",
                                    "width_mm": 28,
                                    "position": 2,
                                },
                            ],
                        }
                    ]
                },
                "bindings": {
                    "channels": [
                        {"channel": "GR", "track_id": "combo", "kind": "curve"},
                        {"channel": "RT", "track_id": "combo", "kind": "curve"},
                    ]
                },
            },
        }

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            templates_dir = root / "templates"
            examples_dir = root / "examples"
            templates_dir.mkdir(parents=True)
            examples_dir.mkdir(parents=True)
            template_path = templates_dir / "base.log.yaml"
            savefile_path = examples_dir / "job.log.yaml"
            template_path.write_text(yaml.safe_dump(template_payload), encoding="utf-8")
            savefile_path.write_text(yaml.safe_dump(savefile_payload), encoding="utf-8")

            spec = load_logfile(savefile_path)

        self.assertEqual(spec.name, "From Savefile")
        self.assertEqual(spec.render_output_path, "job.pdf")
        self.assertEqual(spec.render_dpi, 350)
        self.assertEqual(spec.document["bindings"]["channels"][0]["channel"], "GR")
        self.assertEqual(spec.document["bindings"]["channels"][0]["track_id"], "combo")


if __name__ == "__main__":
    unittest.main()
