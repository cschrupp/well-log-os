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
- `curve_callouts`
- `grid`
- `markers`
- `raster`

Each section can override only the values you care about.

Reference-track number/tick controls live under `render.matplotlib.style.track`, including:
- `reference_grid_mode` (`full` or `edge_ticks`)
- `reference_major_tick_length_ratio`, `reference_minor_tick_length_ratio`
- `reference_tick_color`, `reference_tick_linewidth`
- `reference_label_x`, `reference_label_align`
- `reference_label_fontsize`, `reference_label_color`
- `reference_label_fontfamily`, `reference_label_fontweight`, `reference_label_fontstyle`
- `reference_overlay_curve_lane_start`, `reference_overlay_curve_lane_end`
- `reference_overlay_indicator_lane_start`, `reference_overlay_indicator_lane_end`
- `reference_overlay_tick_length_ratio`, `reference_overlay_threshold`

Curve-callout placement defaults live under `render.matplotlib.style.curve_callouts`, including:
- `left_text_x`, `right_text_x`
- `lane_count`, `lane_step_x`
- `edge_padding_px`, `curve_buffer_px`
- `default_depth_offset_steps`
- `top_distance_steps`, `bottom_distance_steps`
- `min_vertical_gap_steps`
- `font_size`, `font_weight`, `font_style`
- `arrow_style`, `arrow_linewidth`

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
- Reference tracks can host scalar overlay curves via `reference_overlay` on curve bindings:
  - `curve`: normalized slim curve inside a configured lane
  - `indicator`: narrow indicator trace inside a configured lane
  - `ticks`: thresholded edge ticks driven by scalar values
- Reference tracks can also host local non-channel events under `reference.events`.
- When `track_header.legend.enabled: true`, a reference-track header keeps the reference scale row
  and uses the legend slot for overlay curve properties (name + scale/unit rows).

## 9) Multi-Curve Track Bindings

Assign multiple curve bindings to the same `track_id` to render multi-curve overlays in one track.

Track-header legend space auto-fits to curve count:

- legend slot line units are expanded to at least the number of curves
- page `track_header_height_mm` is increased when needed to preserve readable legend rows
- multi-curve headers render per-curve blocks (name row + scale row) with curve-colored separators
- each curve can control header visibility via `document.bindings.channels[*].header_display`:
  - `show_name`, `show_unit`, `show_limits`, `show_color`, `wrap_name`
- `header_display.wrap_name: true` wraps curve labels to at most two centered lines at word
  boundaries; when disabled, labels keep the default truncation behavior.
- curve `scale.kind` supports `linear`, `log`/`logarithmic`, and `tangential`
- in paired mode, each curve is ordered as `name` then `scale` immediately below.
- paired-mode spacing can be tuned with `render.matplotlib.style.track_header.paired_scale_text_offset_ratio`.
- track-header title alignment is configurable with `render.matplotlib.style.track_header.title_align` and `title_x`.
- optional `track_header.divisions` object renders header tick values in its own reserved line.
- top x-axis labels are hidden in the plot area so scale/division text stays inside header slots.
- curve property groups keep a fixed vertical quota across tracks; short tracks do not stretch their
  header rows to fill the whole band.
- array-track property groups follow the same fixed-height behavior as curve headers.
- reference-track overlay properties stay inside the legend slot; the scale slot remains reserved
  for the reference axis text (for example `ft 1:240`).
- Narrow reference-track overlay legends can combine `header_display.wrap_name: true` with larger
  legend `line_units` to keep long overlay names readable without widening the track.
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

## 11) Curve Fill Modes

Curve fills are configured on individual curve bindings with `fill`.

Supported kinds:

- `between_curves`
  - fills between two channels in the same track
  - requires scale compatibility between the two rendered curves
- `between_instances`
  - fills between two specific rendered curve instances
  - target is resolved with `fill.other_element_id`
  - intended for cases like the same channel rendered twice with different scales
- `to_lower_limit`
  - fills from the curve to the lower bound of the active scale
- `to_upper_limit`
  - fills from the curve to the upper bound of the active scale
- `baseline_split`
  - draws a vertical baseline at `fill.baseline.value`
  - fills one side with `lower_color` and the other with `upper_color`

Important semantics:

- lower/upper limit fills are tied to scale bounds, not to screen left/right
- reversed axes still use the correct data meaning
- `between_curves` and `between_instances` optionally support `crossover`
- wrapped curves are not yet a general fill target

Example:

```yaml
document:
  bindings:
    channels:
      - channel: TT
        track_id: baseline_fill
        kind: curve
        scale: { kind: linear, min: 200, max: 400, reverse: true }
        fill:
          kind: baseline_split
          label: Baseline 300 us
          alpha: 0.35
          baseline:
            value: 300
            lower_color: "#1f9d55"
            upper_color: "#ef4444"
            line_color: "#111111"
            line_width: 0.6
            line_style: "--"
```

Track headers mirror fill behavior:

- fill preview rows are rendered in the header when a curve fill is present
- crossover fills preview the active left/right colors
- baseline fills preview the correct split position and orientation for the active scale

## 12) Curve Callouts

Curve callouts are configured on curve bindings with `callouts`.

Each callout can control:

- `depth`
- `label`
- `side`: `auto`, `left`, `right`
- `placement`: `inline`, `top`, `bottom`, `top_and_bottom`
- `text_x`
- `depth_offset`
- `distance_from_top`
- `distance_from_bottom`
- `every`
- text and arrow styling

Current placement model:

- labels render inline in the track body at the chosen/generated depth
- `top`, `bottom`, and `top_and_bottom` are section-relative depth generators
- `distance_from_top` / `distance_from_bottom` are interpreted in the current section index units
  (depth or time), not in physical units like mm
- `every` repeats from the section top/bottom anchors, not from each paginated page window
- edge overflow is a hard constraint
- label-label overlap is rejected
- curve overlap is treated as a soft penalty during candidate placement

Example:

```yaml
document:
  bindings:
    channels:
      - channel: CBL
        track_id: cbl
        kind: curve
        callouts:
          - depth: 672
            label: CBL
            placement: top_and_bottom
            distance_from_top: 500
            distance_from_bottom: 500
            every: 1000
            side: right
            text_x: 0.83
```

Reference-track events are configured separately under `tracks[*].reference.events`.
They are layout-local marker objects, not dataset channel bindings.

Supported event fields:

- `depth`
- `label`
- `color`, `line_style`, `line_width`
- `tick_side` (`left`, `right`, `both`)
- `tick_length_ratio`
- optional explicit `lane_start` / `lane_end`
- `text_side`, `text_x`, `depth_offset`
- `font_size`, `font_weight`, `font_style`
- `arrow`, `arrow_style`, `arrow_linewidth`

Event markers render only inside the reference track and are intended for one-off milestones such
as casing foot, readings start/stop, tool-state transitions, and manually curated depth flags.

## 13) Array Display Options

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
