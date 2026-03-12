from __future__ import annotations

from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .errors import TemplateValidationError

LOGFILE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "well_log_os Logfile",
    "type": "object",
    "required": ["version", "name", "render", "document"],
    "additionalProperties": False,
    "properties": {
        "version": {"type": "integer", "const": 1},
        "name": {"type": "string", "minLength": 1},
        "data": {"$ref": "#/$defs/dataSource"},
        "render": {
            "type": "object",
            "required": ["output_path", "dpi"],
            "additionalProperties": False,
            "properties": {
                "backend": {"type": "string", "enum": ["matplotlib", "plotly"]},
                "output_path": {"type": "string", "minLength": 1},
                "dpi": {"type": "integer", "minimum": 1},
                "continuous_strip_page_height_mm": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
                "matplotlib": {"$ref": "#/$defs/matplotlibRender"},
            },
        },
        "document": {
            "type": "object",
            "required": ["page", "layout", "bindings"],
            "properties": {
                "name": {"type": "string"},
                "page": {"$ref": "#/$defs/documentPage"},
                "depth": {"$ref": "#/$defs/documentDepth"},
                "header": {"$ref": "#/$defs/documentHeader"},
                "footer": {"$ref": "#/$defs/documentFooter"},
                "markers": {"$ref": "#/$defs/documentMarkers"},
                "zones": {"$ref": "#/$defs/documentZones"},
                "layout": {"$ref": "#/$defs/documentLayout"},
                "bindings": {"$ref": "#/$defs/documentBindings"},
            },
            "additionalProperties": True,
        },
    },
    "$defs": {
        "dataSource": {
            "type": "object",
            "required": ["source_path"],
            "additionalProperties": False,
            "properties": {
                "source_path": {"type": "string", "minLength": 1},
                "source_format": {"type": "string", "enum": ["auto", "las", "dlis"]},
            },
        },
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
                "track_gap_mm": {"type": "number"},
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
        "documentLayout": {
            "type": "object",
            "required": ["log_sections"],
            "additionalProperties": False,
            "properties": {
                "heading": {"type": "object"},
                "comments": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "tail": {"type": "object"},
                "log_sections": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/layoutLogSection"},
                },
            },
        },
        "layoutLogSection": {
            "type": "object",
            "required": ["id", "tracks"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "data": {"$ref": "#/$defs/dataSource"},
                "tracks": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/layoutTrack"},
                },
            },
        },
        "layoutTrack": {
            "type": "object",
            "required": ["id", "width_mm"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "title": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": [
                        "reference",
                        "normal",
                        "array",
                        "annotation",
                        "depth",
                        "curve",
                        "image",
                    ],
                },
                "width_mm": {"type": "number", "exclusiveMinimum": 0},
                "position": {"type": "integer", "minimum": 1},
                "x_scale": {"$ref": "#/$defs/trackScale"},
                "grid": {"$ref": "#/$defs/grid"},
                "track_header": {"$ref": "#/$defs/trackHeader"},
                "reference": {"$ref": "#/$defs/referenceTrack"},
            },
        },
        "documentBindings": {
            "type": "object",
            "required": ["channels"],
            "additionalProperties": False,
            "properties": {
                "on_missing": {"type": "string", "enum": ["skip", "error"]},
                "channels": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/documentChannelBinding"},
                },
            },
        },
        "documentChannelBinding": {
            "type": "object",
            "required": ["channel", "track_id"],
            "additionalProperties": False,
            "properties": {
                "section": {"type": "string", "minLength": 1},
                "channel": {"type": "string", "minLength": 1},
                "track_id": {"type": "string", "minLength": 1},
                "required": {"type": "boolean"},
                "kind": {"type": "string", "enum": ["curve", "raster"]},
                "label": {"type": "string"},
                "style": {"$ref": "#/$defs/stylePatch"},
                "scale": {"$ref": "#/$defs/trackScale"},
                "wrap": {"$ref": "#/$defs/curveWrap"},
                "render_mode": {"type": "string", "enum": ["line", "value_labels"]},
                "value_labels": {"$ref": "#/$defs/curveValueLabels"},
                "header_display": {"$ref": "#/$defs/curveHeaderDisplay"},
                "interpolation": {"type": "string"},
                "profile": {"type": "string", "enum": ["generic", "vdl", "waveform"]},
                "normalization": {
                    "type": "string",
                    "enum": ["auto", "none", "trace_maxabs", "global_maxabs"],
                },
                "waveform_normalization": {
                    "type": "string",
                    "enum": ["auto", "none", "trace_maxabs", "global_maxabs"],
                },
                "show_raster": {"type": "boolean"},
                "raster_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "colorbar": {"$ref": "#/$defs/rasterColorbar"},
                "sample_axis": {"$ref": "#/$defs/rasterSampleAxis"},
                "waveform": {"$ref": "#/$defs/rasterWaveform"},
                "color_limits": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "clip_percentiles": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "matplotlibRender": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "style": {"$ref": "#/$defs/matplotlibStyle"},
            },
        },
        "matplotlibStyle": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "header": {"$ref": "#/$defs/matplotlibStyleHeader"},
                "footer": {"$ref": "#/$defs/matplotlibStyleFooter"},
                "section_title": {"$ref": "#/$defs/matplotlibStyleSectionTitle"},
                "track_header": {"$ref": "#/$defs/matplotlibStyleTrackHeader"},
                "track": {"$ref": "#/$defs/matplotlibStyleTrack"},
                "grid": {"$ref": "#/$defs/matplotlibStyleGrid"},
                "markers": {"$ref": "#/$defs/matplotlibStyleMarkers"},
                "raster": {"$ref": "#/$defs/matplotlibStyleRaster"},
            },
        },
        "matplotlibStyleSectionTitle": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean"},
                "height_mm": {"type": "number", "minimum": 0},
                "background_color": {"type": "string", "minLength": 1},
                "border_color": {"type": "string", "minLength": 1},
                "border_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "title_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "subtitle_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "title_color": {"type": "string", "minLength": 1},
                "subtitle_color": {"type": "string", "minLength": 1},
                "title_y": {"type": "number", "minimum": 0, "maximum": 1},
                "subtitle_y": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "matplotlibStyleHeader": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title_x": {"type": "number"},
                "title_y": {"type": "number"},
                "title_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "subtitle_x": {"type": "number"},
                "subtitle_y": {"type": "number"},
                "subtitle_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "field_x": {"type": "number"},
                "field_start_y": {"type": "number"},
                "field_step_y": {"type": "number"},
                "field_fontsize": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "matplotlibStyleFooter": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "line_x": {"type": "number"},
                "line_start_y": {"type": "number"},
                "line_step_y": {"type": "number"},
                "line_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "page_x": {"type": "number"},
                "page_y": {"type": "number"},
                "page_fontsize": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "matplotlibStyleTrackHeader": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "background_color": {"type": "string", "minLength": 1},
                "separator_color": {"type": "string", "minLength": 1},
                "separator_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "slot_top": {"type": "number"},
                "slot_bottom": {"type": "number"},
                "font_scale_factor": {"type": "number", "exclusiveMinimum": 0},
                "title_min_pt": {"type": "number", "exclusiveMinimum": 0},
                "title_max_pt": {"type": "number", "exclusiveMinimum": 0},
                "scale_min_pt": {"type": "number", "exclusiveMinimum": 0},
                "scale_max_pt": {"type": "number", "exclusiveMinimum": 0},
                "legend_empty_min_pt": {"type": "number", "exclusiveMinimum": 0},
                "legend_empty_max_pt": {"type": "number", "exclusiveMinimum": 0},
                "legend_row_min_pt": {"type": "number", "exclusiveMinimum": 0},
                "legend_row_max_pt": {"type": "number", "exclusiveMinimum": 0},
                "title_x": {"type": "number"},
                "title_align": {"type": "string", "enum": ["left", "center", "right"]},
                "text_x": {"type": "number"},
                "legend_line_start": {"type": "number"},
                "legend_line_end": {"type": "number"},
                "legend_text_x": {"type": "number"},
                "legend_label_width_ratio": {"type": "number", "exclusiveMinimum": 0},
                "legend_char_width_ratio": {"type": "number", "exclusiveMinimum": 0},
                "legend_min_chars": {"type": "integer", "minimum": 1},
                "legend_line_min_width": {"type": "number", "exclusiveMinimum": 0},
                "scale_left_x": {"type": "number"},
                "scale_unit_x": {"type": "number"},
                "scale_right_x": {"type": "number"},
                "paired_scale_text_offset_ratio": {"type": "number", "minimum": 0},
                "division_tick_count": {"type": "integer", "minimum": 2},
                "division_tick_length_ratio": {"type": "number", "minimum": 0},
                "division_tick_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "division_tick_color": {"type": "string", "minLength": 1},
                "division_axis_y_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "division_label_y_ratio": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "matplotlibStyleTrack": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "background_color": {"type": "string", "minLength": 1},
                "depth_background_color": {"type": "string", "minLength": 1},
                "depth_tick_labelsize": {"type": "number", "exclusiveMinimum": 0},
                "depth_tick_color": {"type": "string", "minLength": 1},
                "x_tick_labelsize": {"type": "number", "exclusiveMinimum": 0},
                "frame_color": {"type": "string", "minLength": 1},
                "frame_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "marker_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "tangential_spread": {"type": "number", "exclusiveMinimum": 0},
                "reference_grid_mode": {
                    "type": "string",
                    "enum": ["full", "edge_ticks"],
                },
                "reference_major_tick_length_ratio": {"type": "number", "exclusiveMinimum": 0},
                "reference_minor_tick_length_ratio": {"type": "number", "exclusiveMinimum": 0},
                "reference_tick_color": {"type": "string", "minLength": 1},
                "reference_tick_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "reference_label_x": {"type": "number"},
                "reference_label_align": {
                    "type": "string",
                    "enum": ["left", "center", "right"],
                },
                "reference_label_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "reference_label_color": {"type": "string", "minLength": 1},
                "reference_label_fontfamily": {"type": ["string", "null"]},
                "reference_label_fontweight": {"type": "string"},
                "reference_label_fontstyle": {"type": "string"},
            },
        },
        "matplotlibStyleGrid": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x_major_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "x_minor_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "depth_major_color": {"type": "string", "minLength": 1},
                "depth_major_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "depth_major_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "depth_minor_color": {"type": "string", "minLength": 1},
                "depth_minor_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "depth_minor_alpha": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "matplotlibStyleMarkers": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "callout_anchor_x": {"type": "number"},
                "callout_text_x": {"type": "number"},
                "callout_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "callout_text_color": {"type": "string", "minLength": 1},
                "callout_arrow_style": {"type": "string", "minLength": 1},
                "callout_arrow_linewidth": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "matplotlibStyleRaster": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "colorbar_width_ratio": {"type": "number", "exclusiveMinimum": 0},
                "colorbar_pad_ratio": {"type": "number", "minimum": 0},
                "colorbar_tick_labelsize": {"type": "number", "exclusiveMinimum": 0},
                "colorbar_label_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "colorbar_tick_color": {"type": "string", "minLength": 1},
                "colorbar_label_color": {"type": "string", "minLength": 1},
                "sample_axis_tick_labelsize": {"type": "number", "exclusiveMinimum": 0},
                "sample_axis_label_fontsize": {"type": "number", "exclusiveMinimum": 0},
                "sample_axis_tick_color": {"type": "string", "minLength": 1},
                "sample_axis_label_color": {"type": "string", "minLength": 1},
                "sample_axis_label_pad": {"type": "number"},
                "header_colorbar_text_color": {"type": "string", "minLength": 1},
                "header_colorbar_border_color": {"type": "string", "minLength": 1},
                "header_colorbar_border_linewidth": {"type": "number", "exclusiveMinimum": 0},
                "header_colorbar_bar_center_y_ratio": {"type": "number", "minimum": 0},
                "header_colorbar_bar_height_ratio": {"type": "number", "minimum": 0},
                "header_colorbar_label_y_ratio": {"type": "number", "minimum": 0},
            },
        },
        "referenceTrack": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "axis": {"type": "string", "enum": ["depth", "time"]},
                "define_layout": {"type": "boolean"},
                "unit": {"type": "string", "minLength": 1},
                "scale_ratio": {"type": "integer", "minimum": 1},
                "major_step": {"type": "number", "exclusiveMinimum": 0},
                "minor_step": {"type": "number", "exclusiveMinimum": 0},
                "values_orientation": {"type": "string", "enum": ["horizontal", "vertical"]},
                "secondary_grid": {"$ref": "#/$defs/referenceTrackSecondaryGrid"},
                "header": {"$ref": "#/$defs/referenceTrackHeader"},
                "number_format": {"$ref": "#/$defs/referenceTrackNumberFormat"},
            },
        },
        "referenceTrackSecondaryGrid": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "display": {"type": "boolean"},
                "line_count": {"type": "integer", "minimum": 1},
            },
        },
        "referenceTrackHeader": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "display_unit": {"type": "boolean"},
                "display_scale": {"type": "boolean"},
                "display_annotations": {"type": "boolean"},
            },
        },
        "referenceTrackNumberFormat": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["automatic", "fixed", "scientific", "concise"],
                },
                "precision": {"type": "integer", "minimum": 0},
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
                "display": {"$ref": "#/$defs/gridDisplayMode"},
                "major": {"type": "boolean"},
                "minor": {"type": "boolean"},
                "major_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "minor_alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "horizontal": {"$ref": "#/$defs/horizontalGrid"},
                "vertical": {"$ref": "#/$defs/verticalGrid"},
            },
        },
        "gridDisplayMode": {
            "type": ["string", "boolean"],
            "enum": ["below", "above", "none", True, False],
        },
        "gridScaleKind": {
            "type": "string",
            "enum": ["linear", "logarithmic", "log", "exponential", "tangential", "tangent"],
        },
        "horizontalGridLine": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visible": {"type": "boolean"},
                "color": {"type": "string", "minLength": 1},
                "thickness": {"type": "number", "exclusiveMinimum": 0},
                "alpha": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "horizontalGrid": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "display": {"$ref": "#/$defs/gridDisplayMode"},
                "main": {"$ref": "#/$defs/horizontalGridLine"},
                "secondary": {"$ref": "#/$defs/horizontalGridLine"},
            },
        },
        "verticalGridLine": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visible": {"type": "boolean"},
                "line_count": {"type": "integer", "minimum": 1},
                "thickness": {"type": "number", "exclusiveMinimum": 0},
                "color": {"type": "string", "minLength": 1},
                "alpha": {"type": "number", "minimum": 0, "maximum": 1},
                "scale": {"$ref": "#/$defs/gridScaleKind"},
                "spacing_mode": {"type": "string", "enum": ["count", "manual", "scale", "auto"]},
            },
        },
        "verticalGrid": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "display": {"$ref": "#/$defs/gridDisplayMode"},
                "main": {"$ref": "#/$defs/verticalGridLine"},
                "secondary": {"$ref": "#/$defs/verticalGridLine"},
            },
        },
        "trackScale": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["auto", "linear", "log", "logarithmic", "tangential", "tangent"],
                },
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
        "curveHeaderDisplay": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "show_name": {"type": "boolean"},
                "show_unit": {"type": "boolean"},
                "show_limits": {"type": "boolean"},
                "show_color": {"type": "boolean"},
            },
        },
        "curveWrap": {
            "anyOf": [
                {"type": "boolean"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "color": {"type": "string", "minLength": 1},
                    },
                },
            ]
        },
        "rasterColorbar": {
            "anyOf": [
                {"type": "boolean"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "label": {"type": "string", "minLength": 1},
                        "position": {"type": "string", "enum": ["right", "header"]},
                    },
                },
            ]
        },
        "rasterSampleAxis": {
            "anyOf": [
                {"type": "boolean"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "label": {"type": "string", "minLength": 1},
                        "unit": {"type": "string", "minLength": 1},
                        "ticks": {"type": "integer", "minimum": 2},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
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
            ]
        },
        "rasterWaveform": {
            "anyOf": [
                {"type": "boolean"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "stride": {"type": "integer", "minimum": 1},
                        "amplitude_scale": {"type": "number", "exclusiveMinimum": 0},
                        "color": {"type": "string", "minLength": 1},
                        "line_width": {"type": "number", "exclusiveMinimum": 0},
                        "fill": {"type": "boolean"},
                        "positive_fill_color": {"type": "string", "minLength": 1},
                        "negative_fill_color": {"type": "string", "minLength": 1},
                        "invert_fill_polarity": {"type": "boolean"},
                        "max_traces": {"type": "integer", "minimum": 1},
                    },
                },
            ]
        },
        "curveValueLabels": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "step": {"type": "number", "exclusiveMinimum": 0},
                "format": {
                    "type": "string",
                    "enum": ["automatic", "fixed", "scientific", "concise"],
                },
                "precision": {"type": "integer", "minimum": 0},
                "color": {"type": "string", "minLength": 1},
                "font_size": {"type": "number", "exclusiveMinimum": 0},
                "font_family": {"type": "string", "minLength": 1},
                "font_weight": {"type": "string"},
                "font_style": {"type": "string"},
                "horizontal_alignment": {
                    "type": "string",
                    "enum": ["left", "center", "right"],
                },
                "vertical_alignment": {
                    "type": "string",
                    "enum": ["top", "center", "bottom"],
                },
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
                "divisions": {"$ref": "#/$defs/trackHeaderObjectConfig"},
            },
            "additionalProperties": False,
        },
        "trackHeaderObjectSlot": {
            "type": "object",
            "required": ["kind"],
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["title", "scale", "legend", "divisions"]},
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
