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
from well_log_os.model import GridScaleKind, ScalarChannel, ScaleKind, WellDataset


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
        dataset = WellDataset(name="sample", well_metadata={"WELL": "TEST-1"})
        dataset.add_channel(
            ScalarChannel("GR", depth, "m", "gAPI", values=np.linspace(20, 120, depth.size))
        )
        dataset.add_channel(
            ScalarChannel(
                "RT", depth, "m", "ohm.m", values=np.exp(np.linspace(0.1, 5.0, depth.size))
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
