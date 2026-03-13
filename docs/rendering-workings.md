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
4. Build one `LogDocument` per `document.layout.log_sections[*]` from `document.layout` + `document.bindings`.
5. Build renderer with backend options:
   - `dpi`
   - `continuous_strip_page_height_mm`
   - `style` from `render.matplotlib.style`
6. Render to file/figures.

Track assembly is track-first:

- `document.layout.log_sections[*].tracks` defines the physical layout.
- `document.bindings.channels` assigns dataset channels into those tracks.
- Data-source routing:
  - each section can define source with `document.layout.log_sections[*].data`
  - root `data` is optional and acts as default only when section data is omitted
  - section data block supports `source_path` and optional `source_format` (`auto|las|dlis`)
- Multi-section binding routing:
  - if `binding.section` is set, binding is applied to that section only
  - if `binding.section` is omitted, `track_id` must be unique across sections
  - ambiguous `track_id` across sections requires explicit `binding.section`

In layout/bindings mode, section placeholders are available:

- `document.layout.heading`
- `document.layout.comments`
- `document.layout.log_sections`
- `document.layout.tail`

Each section is rendered in sequence into the same output artifact (matplotlib backend).

## 3) Matplotlib Style Sections

`render.matplotlib.style` supports these top-level sections:

- `header`
- `footer`
- `section_title`
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

## 6) Page Spacing Controls

Track placement spacing is controlled by page config, not renderer hardcoding:

- `document.page.margin_left_mm`: first track start offset from the page origin.
- `document.page.track_gap_mm`: spacing between adjacent track frames.

Defaults are:

- `margin_left_mm: 0`
- `track_gap_mm: 0`

## 7) Track Types

The document model supports these track types:

- `reference`: layout reference axis track (depth/time semantics), can host scalar overlays.
- `normal`: single-value-per-index curves.
- `array`: array/raster channels with optional scalar overlays.
- `annotation`: reserved track type for annotation-focused display.

Backward-compatible aliases are accepted in configs:

- `depth` -> `reference`
- `curve` -> `normal`
- `image` -> `array`

## 8) Reference Track Contract

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

## 9) Multi-Curve Track Bindings

Assign multiple curve bindings to the same `track_id` to render multi-curve overlays in one track.

Track-header legend space auto-fits to curve count:

- legend slot line units are expanded to at least the number of curves
- page `track_header_height_mm` is increased when needed to preserve readable legend rows
- multi-curve headers render per-curve blocks (name row + scale row) with curve-colored separators
- each curve can control header visibility via `document.bindings.channels[*].header_display`:
  - `show_name`, `show_unit`, `show_limits`, `show_color`
- curve `scale.kind` supports `linear`, `log`/`logarithmic`, and `tangential`
- in paired mode, each curve is ordered as `name` then `scale` immediately below.
- paired-mode spacing can be tuned with `render.matplotlib.style.track_header.paired_scale_text_offset_ratio`.
- track-header title alignment is configurable with `render.matplotlib.style.track_header.title_align` and `title_x`.
- optional `track_header.divisions` object renders header tick values in its own reserved line.
- top x-axis labels are hidden in the plot area so scale/division text stays inside header slots.
- each `layout.log_sections[*]` may define:
  - `title` (required to render the section banner)
  - `subtitle` (optional)
- section banners are drawn as full-width boxed titles across the track span.

## 10) Track Grid Modes

Track grids can now be configured per track with horizontal/vertical blocks:

- `tracks[*].grid.horizontal`
- `tracks[*].grid.vertical`

Vertical grid scales support:

- `linear`
- `logarithmic` (aliases: `log`, `exponential`)
- `tangential` (alias: `tangent`)

Main and secondary vertical grids can each set:

- `visible`
- `line_count`
- `thickness`
- `color`
- `alpha`
- `scale`
- `spacing_mode` (`count`/`manual` or `scale`/`auto`)

For log curves, `spacing_mode: scale` + `scale: logarithmic` derives vertical grid cycles from the
actual scale bounds (for example `2->200` vs `2->2000`), including start-value effects (`1` vs `2`).

Recommended patterns:

```yaml
# Auto from scale bounds (recommended for log tracks)
grid:
  vertical:
    main:
      scale: logarithmic
      spacing_mode: scale
    secondary:
      scale: logarithmic
      spacing_mode: scale
```

```yaml
# Fixed/manual density (same line count regardless of min/max)
grid:
  vertical:
    main:
      scale: logarithmic
      line_count: 4
      spacing_mode: count
    secondary:
      scale: logarithmic
      line_count: 4
      spacing_mode: count
```

See [examples/log_scale_options.log.yaml](../examples/log_scale_options.log.yaml) for a
real-data 4-track comparison (`0-100` linear, `2-200` log, `2-2000` log, and tangential).

Curve-level log wrap is available in bindings with:

```yaml
document:
  bindings:
    channels:
      - channel: RT
        track_id: rt_wrap
        kind: curve
        scale: { kind: log, min: 2, max: 200 }
        wrap:
          enabled: true
          color: "#ef4444"
```

Wrapping applies to curves in `reference`, `normal`, and `array` tracks.
It folds out-of-range curve values into the configured scale interval and can
render wrapped sections in a separate color (`wrap.color`).

## 11) Array Display Options

Raster bindings in array tracks support:

- `profile`: `generic`, `vdl`, or `waveform`
- `normalization`: `auto`, `none`, `trace_maxabs`, `global_maxabs`
- `colorbar`: `true/false` or object `{ enabled, label, position }`
- `sample_axis`:
  `{ enabled, label, unit, ticks, min, max, source_origin, source_step }`
- `waveform`: `true/false` or object
  `{ enabled, stride, amplitude_scale, color, line_width, max_traces, fill,
  positive_fill_color, negative_fill_color, invert_fill_polarity }`
- existing raster options:
  - `style.colormap`
  - `interpolation`
  - `color_limits`
  - `clip_percentiles`

Profile semantics:

- `generic`: plain raster display with optional sample-axis labels/ticks.
- `vdl`: Variable Density Log density display using zero-centered clipping and grayscale mapping.
  With `gray_r`, negative amplitudes render white and positive amplitudes render black.
- `waveform`: waveform-only array display. Raster background is disabled by default and waveform
  overlay is enabled by default.

Sample-axis resolution order:

1. `binding.sample_axis.source_origin/source_step` from the logfile, when provided.
2. `RasterChannel.sample_axis` loaded from the source file.
3. DLIS tool/channel metadata-derived axis, when available.

For current DLIS VDL/WF1 support, the loader can derive micro-time axes from channel axes or tool
metadata such as digitizer sample interval. The renderer then clips the actual raster/waveform
columns to `sample_axis.min/max` before plotting. This is important: the selected time window is a
true crop, not a relabel of the full waveform width.

This also means end-user tuning remains valid and necessary for parity work. If a vendor-generated
log starts slightly earlier or later than our auto-derived axis, users should adjust:

- `sample_axis.source_origin`
- `sample_axis.source_step`

Example:

```yaml
document:
  bindings:
    channels:
      - channel: VDL
        track_id: vdl_array
        kind: raster
        profile: vdl
        normalization: auto
        style: { colormap: gray_r }
        interpolation: nearest
        clip_percentiles: [1, 99]
        colorbar:
          enabled: true
          label: Amplitude
          position: header
        sample_axis:
          enabled: false
          unit: us
          source_origin: 40
          source_step: 10
          min: 200
          max: 1200
          ticks: 7
        waveform:
          enabled: true
          stride: 5
          amplitude_scale: 0.35
          color: "#5b3f8c"
          line_width: 0.22
          fill: true
          positive_fill_color: "#000000"
          negative_fill_color: "#ffffff"
          invert_fill_polarity: true
          max_traces: 700
```

Reference example files:

- [examples/cbl_vdl_array_mvp.log.yaml](../examples/cbl_vdl_array_mvp.log.yaml)
- [examples/cbl_vdl_array_overlay.log.yaml](../examples/cbl_vdl_array_overlay.log.yaml)
- [examples/cbl_comparison_feet.log.yaml](../examples/cbl_comparison_feet.log.yaml)
- [examples/cbl_comparison_feet_full.log.yaml](../examples/cbl_comparison_feet_full.log.yaml)
- [examples/cbl_vdl_array_mvp_demo.py](../examples/cbl_vdl_array_mvp_demo.py)
