# Rendering Workings

This document explains how `well_log_os` builds a final rendered log from:
- template defaults
- savefile overrides
- renderer defaults

## 1) Configuration Layers

At render time, configuration is resolved in this order:

1. Built-in renderer defaults from:
   - `src/well_log_os/renderers/matplotlib_defaults.yaml`
2. Template values (`template.path` target YAML).
3. Savefile values (job-specific YAML).

For Matplotlib style settings, the effective style is:

`matplotlib_defaults.yaml` + `render.matplotlib.style` (deep merge)

Where savefile/template style keys override only the keys they define.

## 2) Logfile Resolution Flow

Pipeline entrypoint:
- `render_from_logfile(...)` in `src/well_log_os/pipeline.py`

Flow:

1. Load and merge template + savefile (`load_logfile`).
2. Validate against JSON schema (`logfile_schema.py`).
3. Load source data (`.las` / `.dlis`) into `WellDataset`.
4. Build `LogDocument` from document config + auto-track rules.
5. Build renderer with backend options:
   - `dpi`
   - `continuous_strip_page_height_mm`
   - `style` from `render.matplotlib.style`
6. Render to file/figures.

## 3) Matplotlib Style Sections

`render.matplotlib.style` supports these top-level sections:

- `header`
- `footer`
- `track_header`
- `track`
- `grid`
- `markers`

Each section can override only the values you care about.

Reference-track number/tick controls live under `render.matplotlib.style.track`, including:
- `reference_grid_mode` (`full` or `edge_ticks`)
- `reference_major_tick_length_ratio`, `reference_minor_tick_length_ratio`
- `reference_tick_color`, `reference_tick_linewidth`
- `reference_label_x`, `reference_label_align`
- `reference_label_fontsize`, `reference_label_color`
- `reference_label_fontfamily`, `reference_label_fontweight`, `reference_label_fontstyle`

Example:

```yaml
render:
  backend: matplotlib
  output_path: ../workspace/renders/job.pdf
  dpi: 300
  matplotlib:
    style:
      track_header:
        background_color: "#efefef"
        separator_linewidth: 0.4
      track:
        x_tick_labelsize: 7.5
      grid:
        depth_major_linewidth: 0.8
```

## 4) Why Defaults Are in YAML

Benefits:
- no large hardcoded style blocks inside renderer code
- easier to tune visual defaults without touching logic
- consistent base style for UI/template generation later

## 5) Current Boundaries

- This configuration controls visual/layout styling.
- Some backend internals remain code-level (for example Matplotlib PDF rc settings).
- Track content rules remain model-driven:
  - array tracks accept raster + scalar overlays
  - normal and reference tracks do not accept raster elements

## 6) Track Types

The document model supports these track types:

- `reference`: layout reference axis track (depth/time semantics), can host scalar overlays.
- `normal`: single-value-per-index curves.
- `array`: array/raster channels with optional scalar overlays.
- `annotation`: reserved track type for annotation-focused display.

Backward-compatible aliases are accepted in configs:

- `depth` -> `reference`
- `curve` -> `normal`
- `image` -> `array`

## 7) Reference Track Contract

A `reference` track is not only visual: it can define the layout reference axis.

- If `reference.define_layout: true`, its axis fields update the document layout axis:
  - `unit`
  - `scale_ratio`
  - `major_step`
  - `minor_step` (or `major_step / secondary_grid.line_count` when omitted)
- Main/secondary reference grids are rendered from this configuration.
- Reference values are drawn inside the track area (not outside the frame).
- Header display can be controlled with:
  - `reference.header.display_unit`
  - `reference.header.display_scale`
  - `reference.header.display_annotations`
