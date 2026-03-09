from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from well_log_os.errors import TemplateValidationError
from well_log_os.logfile import build_document_for_logfile, load_logfile, logfile_from_mapping
from well_log_os.model import ScalarChannel, WellDataset


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
        },
        "auto_tracks": {
            "on_missing": "skip",
            "max_tracks": 2,
            "depth_track": {"id": "depth", "title": "Depth", "width_mm": 16},
            "tracks": [
                {
                    "channel": "GR",
                    "configure": {
                        "id": "gr",
                        "width_mm": 28,
                        "title_template": "{mnemonic} [{unit}]",
                        "style": {
                            "color": "#1b5e20",
                            "line_width": 0.9,
                            "line_style": "-",
                            "opacity": 1.0,
                        },
                        "grid": {
                            "major": True,
                            "minor": True,
                            "major_alpha": 0.35,
                            "minor_alpha": 0.15,
                        },
                        "scale": {
                            "kind": "auto",
                            "percentile_low": 2,
                            "percentile_high": 98,
                            "log_ratio_threshold": 200,
                            "min_positive": 1e-6,
                        },
                    },
                },
                {
                    "channel": "RT",
                    "configure": {
                        "id": "rt",
                        "width_mm": 28,
                        "title_template": "{mnemonic} [{unit}]",
                        "style": {
                            "color": "#0d47a1",
                            "line_width": 0.9,
                            "line_style": "-",
                            "opacity": 1.0,
                        },
                        "grid": {
                            "major": True,
                            "minor": True,
                            "major_alpha": 0.35,
                            "minor_alpha": 0.15,
                        },
                        "scale": {
                            "kind": "auto",
                            "percentile_low": 2,
                            "percentile_high": 98,
                            "log_ratio_threshold": 200,
                            "min_positive": 1e-6,
                        },
                    },
                },
            ],
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

    def test_logfile_builds_document_without_hardcoded_tracks(self) -> None:
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
        self.assertEqual(len(document.tracks), 3)  # depth + 2 selected scalar tracks
        self.assertEqual(document.tracks[1].elements[0].channel, "GR")
        self.assertEqual(document.tracks[2].elements[0].channel, "RT")

    def test_invalid_logfile_configuration_raises(self) -> None:
        payload = build_mapping()
        del payload["auto_tracks"]["tracks"][0]["configure"]["style"]["color"]
        with self.assertRaises(TemplateValidationError):
            logfile_from_mapping(payload)

    def test_track_configure_is_required_without_default(self) -> None:
        payload = build_mapping()
        del payload["auto_tracks"]["tracks"][0]["configure"]
        with self.assertRaises(TemplateValidationError) as ctx:
            logfile_from_mapping(payload)
        self.assertIn("auto_tracks.tracks[0].configure is required", str(ctx.exception))

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

    def test_default_configure_supports_string_track_entries(self) -> None:
        payload = build_mapping()
        payload["auto_tracks"]["default_configure"] = deepcopy(
            payload["auto_tracks"]["tracks"][0]["configure"]
        )
        payload["auto_tracks"]["tracks"] = [
            "GR",
            {
                "channel": "RT",
                "configure": {
                    "style": {"color": "#ff5500"},
                },
            },
        ]

        spec = logfile_from_mapping(payload)
        self.assertEqual(spec.auto_tracks["tracks"][0]["channel"], "GR")
        self.assertEqual(spec.auto_tracks["tracks"][0]["configure"]["width_mm"], 28)
        self.assertEqual(spec.auto_tracks["tracks"][1]["configure"]["style"]["color"], "#ff5500")

    def test_depth_track_reference_config_controls_layout_axis(self) -> None:
        payload = build_mapping()
        payload["auto_tracks"]["depth_track"]["reference"] = {
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
        payload["auto_tracks"]["depth_track"]["position"] = 2
        payload["auto_tracks"]["tracks"][0]["configure"]["position"] = 1
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(document.tracks[0].id, "gr")
        self.assertEqual(document.tracks[1].id, "depth")

    def test_auto_track_can_render_curve_values_as_labels(self) -> None:
        payload = build_mapping()
        payload["auto_tracks"]["tracks"][0]["configure"]["curve_render_mode"] = "value_labels"
        payload["auto_tracks"]["tracks"][0]["configure"]["value_labels"] = {
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
        curve = document.tracks[1].elements[0]
        self.assertEqual(curve.render_mode, "value_labels")
        self.assertEqual(curve.value_labels.step, 5.0)

    def test_auto_tracks_can_group_multiple_curves_in_one_track(self) -> None:
        payload = build_mapping()
        payload["auto_tracks"]["tracks"][0]["configure"]["id"] = "combo"
        payload["auto_tracks"]["tracks"][0]["configure"]["title"] = "GR / RT"
        payload["auto_tracks"]["tracks"][1]["configure"]["id"] = "combo"
        payload["auto_tracks"]["tracks"][1]["configure"]["style"]["color"] = "#0d47a1"
        spec = logfile_from_mapping(payload)
        document = build_document_for_logfile(
            spec,
            self.build_dataset(),
            source_path=Path("example_input.las"),
        )
        self.assertEqual(len(document.tracks), 2)  # depth + grouped combo
        combo = document.tracks[1]
        self.assertEqual(combo.id, "combo")
        self.assertEqual(combo.title, "GR / RT")
        self.assertEqual(len(combo.elements), 2)
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
            },
            "auto_tracks": {
                "on_missing": "skip",
                "max_tracks": 8,
                "depth_track": {"id": "depth", "title": "Depth", "width_mm": 16},
                "default_configure": {
                    "width_mm": 28,
                    "title_template": "{mnemonic} [{unit}]",
                    "style": {"color": "#1b5e20"},
                    "grid": {"major": True, "minor": True},
                    "scale": {"kind": "auto", "percentile_low": 2, "percentile_high": 98},
                },
                "tracks": [],
            },
        }
        savefile_payload = {
            "template": {"path": "../templates/base.log.yaml"},
            "version": 1,
            "name": "From Savefile",
            "data": {"source_path": "job.las", "source_format": "auto"},
            "render": {"output_path": "job.pdf", "dpi": 350},
            "auto_tracks": {
                "tracks": [
                    "GR",
                    {"channel": "RT", "configure": {"style": {"color": "#0d47a1"}}},
                ]
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
        self.assertEqual(spec.auto_tracks["tracks"][0]["channel"], "GR")
        self.assertEqual(spec.auto_tracks["tracks"][0]["configure"]["width_mm"], 28)
        self.assertEqual(spec.auto_tracks["tracks"][1]["configure"]["style"]["color"], "#0d47a1")


if __name__ == "__main__":
    unittest.main()
