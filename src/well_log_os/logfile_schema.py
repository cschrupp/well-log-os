from __future__ import annotations

from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .errors import TemplateValidationError

LOGFILE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "well_log_os Logfile",
    "type": "object",
    "required": ["version", "name", "data", "render", "document", "auto_tracks"],
    "additionalProperties": False,
    "properties": {
        "version": {"type": "integer", "const": 1},
        "name": {"type": "string", "minLength": 1},
        "data": {
            "type": "object",
            "required": ["source_path"],
            "additionalProperties": False,
            "properties": {
                "source_path": {"type": "string", "minLength": 1},
                "source_format": {"type": "string", "enum": ["auto", "las", "dlis"]},
            },
        },
        "render": {
            "type": "object",
            "required": ["output_path", "dpi"],
            "additionalProperties": False,
            "properties": {
                "backend": {"type": "string", "enum": ["matplotlib", "plotly"]},
                "output_path": {"type": "string", "minLength": 1},
                "dpi": {"type": "integer", "minimum": 1},
            },
        },
        "document": {
            "type": "object",
            "required": ["page"],
            "properties": {
                "name": {"type": "string"},
                "page": {"$ref": "#/$defs/documentPage"},
                "depth": {"$ref": "#/$defs/documentDepth"},
                "header": {"$ref": "#/$defs/documentHeader"},
                "footer": {"$ref": "#/$defs/documentFooter"},
                "markers": {"$ref": "#/$defs/documentMarkers"},
                "zones": {"$ref": "#/$defs/documentZones"},
            },
            "additionalProperties": True,
        },
        "auto_tracks": {"$ref": "#/$defs/autoTracks"},
    },
    "$defs": {
        "documentPage": {
            "type": "object",
            "properties": {
                "size": {"type": "string", "minLength": 1},
                "width_mm": {"type": "number", "exclusiveMinimum": 0},
                "height_mm": {"type": "number", "exclusiveMinimum": 0},
                "orientation": {"type": "string", "enum": ["portrait", "landscape"]},
                "continuous": {"type": "boolean"},
                "margin_left_mm": {"type": "number"},
                "margin_right_mm": {"type": "number"},
                "margin_top_mm": {"type": "number"},
                "margin_bottom_mm": {"type": "number"},
                "header_height_mm": {"type": "number"},
                "track_header_height_mm": {"type": "number"},
                "footer_height_mm": {"type": "number"},
            },
            "anyOf": [
                {"required": ["size"]},
                {"required": ["width_mm", "height_mm"]},
            ],
            "additionalProperties": True,
        },
        "documentDepth": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "minLength": 1},
                "scale": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "number"},
                    ]
                },
                "major_step": {"type": "number"},
                "minor_step": {"type": "number"},
            },
            "additionalProperties": True,
        },
        "documentHeader": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["label", "source_key"],
                        "properties": {
                            "label": {"type": "string"},
                            "source_key": {"type": "string"},
                            "default": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": True,
        },
        "documentFooter": {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "additionalProperties": True,
        },
        "documentMarkers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["depth"],
                "properties": {
                    "depth": {"type": "number"},
                    "label": {"type": "string"},
                    "color": {"type": "string"},
                    "line_style": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "documentZones": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["top", "base"],
                "properties": {
                    "top": {"type": "number"},
                    "base": {"type": "number"},
                    "label": {"type": "string"},
                    "fill_color": {"type": "string"},
                    "alpha": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
        "autoTracks": {
            "type": "object",
            "required": ["depth_track", "tracks"],
            "additionalProperties": False,
            "properties": {
                "on_missing": {"type": "string", "enum": ["skip", "error"]},
                "max_tracks": {"type": "integer", "minimum": 1},
                "depth_track": {"$ref": "#/$defs/depthTrack"},
                "default_configure": {"$ref": "#/$defs/trackConfigure"},
                "tracks": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"$ref": "#/$defs/autoTrackEntry"},
                        ]
                    },
                },
            },
        },
        "depthTrack": {
            "type": "object",
            "required": ["id", "title", "width_mm"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1},
                "width_mm": {"type": "number", "exclusiveMinimum": 0},
                "track_header": {"$ref": "#/$defs/trackHeader"},
            },
        },
        "autoTrackEntry": {
            "type": "object",
            "required": ["channel"],
            "additionalProperties": False,
            "properties": {
                "channel": {"type": "string", "minLength": 1},
                "required": {"type": "boolean"},
                "configure": {"$ref": "#/$defs/trackConfigurePatch"},
            },
        },
        "trackConfigure": {
            "type": "object",
            "required": ["width_mm", "style"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "title": {"type": "string"},
                "title_template": {"type": "string"},
                "width_mm": {"type": "number", "exclusiveMinimum": 0},
                "style": {"$ref": "#/$defs/style"},
                "grid": {"$ref": "#/$defs/grid"},
                "scale": {"$ref": "#/$defs/autoTrackScale"},
                "track_header": {"$ref": "#/$defs/trackHeader"},
            },
        },
        "style": {
            "type": "object",
            "required": ["color"],
            "additionalProperties": False,
            "properties": {
                "color": {"type": "string", "minLength": 1},
                "line_width": {"type": "number", "exclusiveMinimum": 0},
                "line_style": {"type": "string"},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                "fill_color": {"type": "string"},
                "fill_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "colormap": {"type": "string"},
            },
        },
        "stylePatch": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "color": {"type": "string", "minLength": 1},
                "line_width": {"type": "number", "exclusiveMinimum": 0},
                "line_style": {"type": "string"},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                "fill_color": {"type": "string"},
                "fill_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "colormap": {"type": "string"},
            },
        },
        "grid": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "major": {"type": "boolean"},
                "minor": {"type": "boolean"},
                "major_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "minor_alpha": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "autoTrackScale": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["auto", "linear", "log"]},
                "min": {"type": "number"},
                "max": {"type": "number"},
                "reverse": {"type": "boolean"},
                "percentile_low": {"type": "number"},
                "percentile_high": {"type": "number"},
                "log_ratio_threshold": {"type": "number"},
                "min_positive": {"type": "number", "exclusiveMinimum": 0},
            },
            "allOf": [
                {
                    "if": {
                        "anyOf": [
                            {"required": ["min"]},
                            {"required": ["max"]},
                        ]
                    },
                    "then": {"required": ["min", "max"]},
                }
            ],
        },
        "autoTrackScalePatch": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["auto", "linear", "log"]},
                "min": {"type": "number"},
                "max": {"type": "number"},
                "reverse": {"type": "boolean"},
                "percentile_low": {"type": "number"},
                "percentile_high": {"type": "number"},
                "log_ratio_threshold": {"type": "number"},
                "min_positive": {"type": "number", "exclusiveMinimum": 0},
            },
            "allOf": [
                {
                    "if": {
                        "anyOf": [
                            {"required": ["min"]},
                            {"required": ["max"]},
                        ]
                    },
                    "then": {"required": ["min", "max"]},
                }
            ],
        },
        "trackConfigurePatch": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "title": {"type": "string"},
                "title_template": {"type": "string"},
                "width_mm": {"type": "number", "exclusiveMinimum": 0},
                "style": {"$ref": "#/$defs/stylePatch"},
                "grid": {"$ref": "#/$defs/grid"},
                "scale": {"$ref": "#/$defs/autoTrackScalePatch"},
                "track_header": {"$ref": "#/$defs/trackHeader"},
            },
        },
        "trackHeader": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/trackHeaderObjectSlot"},
                    "minItems": 1,
                },
                "title": {"$ref": "#/$defs/trackHeaderObjectConfig"},
                "scale": {"$ref": "#/$defs/trackHeaderObjectConfig"},
                "legend": {"$ref": "#/$defs/trackHeaderObjectConfig"},
            },
            "additionalProperties": False,
        },
        "trackHeaderObjectSlot": {
            "type": "object",
            "required": ["kind"],
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["title", "scale", "legend"]},
                "enabled": {"type": "boolean"},
                "reserve_space": {"type": "boolean"},
                "line_units": {"type": "integer", "minimum": 1},
            },
        },
        "trackHeaderObjectConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean"},
                "reserve_space": {"type": "boolean"},
                "line_units": {"type": "integer", "minimum": 1},
            },
        },
    },
}

_LOGFILE_SCHEMA_VALIDATOR = Draft202012Validator(LOGFILE_JSON_SCHEMA)


def get_logfile_json_schema() -> dict[str, Any]:
    """Return a copy of the JSON Schema used for logfile validation."""
    return deepcopy(LOGFILE_JSON_SCHEMA)


def _format_error_path(error_path: Any) -> str:
    path = "$"
    for part in error_path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def validate_logfile_mapping(data: Any) -> None:
    """Validate logfile YAML mapping data against the JSON Schema."""
    errors = sorted(
        _LOGFILE_SCHEMA_VALIDATOR.iter_errors(data),
        key=lambda error: (_format_error_path(error.absolute_path), error.message),
    )
    if not errors:
        return

    details = "\n".join(
        f"- {_format_error_path(error.absolute_path)}: {error.message}" for error in errors
    )
    raise TemplateValidationError(f"Logfile schema validation failed:\n{details}")
