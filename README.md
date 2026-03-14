# well_log_os

`well_log_os` is an open-source Python toolkit for building printable and interactive well-log layouts from LAS and DLIS data.

The project is intentionally renderer-first:
- normalize subsurface data into typed channels
- describe the sheet using templates and layout specs
- render the same document to static and interactive backends

## Status

This repository currently contains the current MVP baseline:
- normalized data objects for scalar, array, and raster channels
- a printable log document model with tracks, styles, headers, and footers
- YAML template loading
- a physical page layout engine
- optional `matplotlib` and `plotly` renderer backends
- optional LAS and DLIS ingestion adapters
- DLIS VDL/WF1-style array support with derived micro-time sample axes
- printable VDL density and waveform array rendering
- scale-aware curve fills, including crossover, limit, and baseline modes
- track-header fill indicators that mirror the actual plotted fill behavior
- in-track curve callouts with section-relative repetition and collision avoidance
- reference-track scalar overlay modes (`curve`, `indicator`, `ticks`)
- reference-track event objects for local markers such as casing foot or readings start

## Architecture

The package separates three layers:
- `WellDataset`: data and metadata normalized from LAS/DLIS inputs
- `LogDocument`: page, depth, track, and annotation specifications
- renderers: backend-specific drawing implementations that consume the same document

Track types are explicit: `reference`, `normal`, `array`, and `annotation`
(with compatibility aliases `depth`, `curve`, `image`).
Array tracks can host raster data and scalar overlays, while normal/reference tracks do not accept raster elements.
Reference tracks can host scalar overlay curves and local reference events while still defining the
layout axis.
Set `page.continuous: true` in templates to render a single continuous-depth PDF page.
Set `page.track_header_height_mm` to reserve a dedicated per-track header band.
Track headers now support explicit object slots (`title`, `scale`, `legend`) with `enabled`,
`reserve_space`, and `line_units` controls to prevent overlap while keeping fixed spacing.
Depth grid density in continuous mode is controlled by `depth.major_step` and `depth.minor_step`.
Use top-level `markers` and `zones` sections to draw formation and event annotations.

## Development Workflow

This project uses `uv` for environment management and dependency resolution.
The package continues to support Python `>=3.11`.
CI validates the project on Python `3.11`, `3.12`, `3.13`, and `3.14`.

Create or update the environment:

```bash
uv sync
```

Install with optional LAS ingestion and PDF output:

```bash
uv sync --extra las --extra pdf --extra units
```

With all optional backends:

```bash
uv sync --all-extras
```

Run tests:

```bash
uv run python -m unittest discover -s tests -v
```

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

## Example Template

See [examples/triple_combo.yaml](examples/triple_combo.yaml).
For scale/grid behavior examples, see [examples/log_scale_options.log.yaml](examples/log_scale_options.log.yaml).
For resistivity-style scales and wrapped log demo, see
[examples/resistivity_scale_conventions.log.yaml](examples/resistivity_scale_conventions.log.yaml).
For VDL density, waveform overlay, and feet-based comparison examples, see:
- [examples/cbl_vdl_array_mvp.log.yaml](examples/cbl_vdl_array_mvp.log.yaml)
- [examples/cbl_vdl_array_overlay.log.yaml](examples/cbl_vdl_array_overlay.log.yaml)
- [examples/cbl_comparison_feet.log.yaml](examples/cbl_comparison_feet.log.yaml)
- [examples/cbl_comparison_feet_full.log.yaml](examples/cbl_comparison_feet_full.log.yaml)
For fill and callout examples, see:
- [examples/fill_modes_showcase.log.yaml](examples/fill_modes_showcase.log.yaml)
- [examples/cbl_feature_showcase_full.log.yaml](examples/cbl_feature_showcase_full.log.yaml)
- [examples/curve_callouts_showcase.log.yaml](examples/curve_callouts_showcase.log.yaml)
- [examples/curve_callout_bands_showcase.log.yaml](examples/curve_callout_bands_showcase.log.yaml)
- [examples/curve_callout_bands_full.log.yaml](examples/curve_callout_bands_full.log.yaml)
- [examples/reference_track_overlays.log.yaml](examples/reference_track_overlays.log.yaml)

## Template + Savefile Model

`well_log_os` now supports YAML template inheritance for logfile configs.

- Put reusable layout defaults in template files, for example:
  - [templates/wireline_base.template.yaml](templates/wireline_base.template.yaml)
- Create per-job savefiles that reference templates:
  - [examples/cbl_main.log.yaml](examples/cbl_main.log.yaml)

Savefiles use:

```yaml
template:
  path: ../templates/wireline_base.template.yaml
```

Behavior:
- Savefile values override template values.
- Tracks are defined in `document.layout.log_sections[*].tracks`.
- Data sources are section-scoped via `document.layout.log_sections[*].data`.
  If top-level `data` is provided, it acts as a default for sections that do not set one.
- Channels are assigned in `document.bindings.channels` (`channel` + `track_id`).
- Curve scales support `linear`, `log`/`logarithmic`, and `tangential`.
- For log tracks, vertical grid can auto-follow scale bounds with:
  `grid.vertical.main.spacing_mode: scale` and `grid.vertical.secondary.spacing_mode: scale`.
  This adapts cycles and spacing for ranges like `2->200` vs `2->2000`, including non-decade starts.
- Use `spacing_mode: count` when you want fixed/manual line density independent of curve bounds.
- Curves support wrapping across curve-capable tracks (`reference`, `normal`, `array`):
  - `wrap: true` to enable with default curve color.
  - `wrap: { enabled: true, color: "#ef4444" }` to color wrapped segments explicitly.
- Curves support first-class fills:
  - `between_curves` for same-scale curve-vs-curve fills
  - `between_instances` for fills between specific rendered curve instances
  - `to_lower_limit` and `to_upper_limit` for fills to the active scale bounds
  - `baseline_split` for two-color fills around a vertical baseline
- Lower/upper limit fills are tied to the active scale bounds, not to the physical left/right side
  of the screen. Reversed scales still behave correctly.
- When you need a fill between two rendered copies of the same channel, assign explicit element ids:
  - `id: cbl_0_100`
  - `fill.other_element_id: cbl_0_10`
- Track headers render fill indicators that follow the same semantics as the plotted fill, including
  crossover splits and baseline orientation.
- Curves support in-track callouts via `callouts`:
  - inline labels at explicit depths
  - repeated labels from section `top`, `bottom`, or `top_and_bottom`
  - side, text position, font, arrow, and offset controls
  - hard edge avoidance, label-label avoidance, and soft curve-overlap avoidance
- Reference-track curve overlays support `reference_overlay`:
  - `mode: curve` for slim normalized overlay curves
  - `mode: indicator` for narrow indicator lanes
  - `mode: ticks` for thresholded event-tick rendering from scalar channels
- Reference-track headers can now keep the reference scale row while rendering overlay properties in
  the legend slot when `track_header.legend.enabled: true`.
- Curve header labels can opt into two-line wrapping with
  `document.bindings.channels[*].header_display.wrap_name: true`, which is useful for narrow track
  headers such as reference-track overlay legends.
- Reference tracks support local event objects under `reference.events` for one-off markers such as
  casing shoe, readings start, or tool-state transitions.
- Callout repetition is section-relative. `top`, `bottom`, and `top_and_bottom` generate repeated
  depths from the log section bounds, then render each label inline at those generated depths.
- Raster bindings support display controls:
  - `profile` (`generic`, `vdl`, or `waveform`)
  - `normalization` (`auto`, `none`, `trace_maxabs`, `global_maxabs`)
  - `colorbar` (`true/false` or `{ enabled, label, position }`)
  - `sample_axis`
    (`true/false` or `{ enabled, label, unit, ticks, min, max, source_origin, source_step }`)
  - `waveform`
    (`true/false` or
    `{ enabled, stride, amplitude_scale, color, line_width, max_traces, fill,
    positive_fill_color, negative_fill_color, invert_fill_polarity }`)
- Multiple curves per track are supported by assigning multiple bindings to the same `track_id`.
- Section placeholders are first-class in YAML:
  - `document.layout.heading`
  - `document.layout.comments`
  - `document.layout.log_sections`
  - `document.layout.tail`
- Template YAML files can be partial; the merged savefile result is what gets validated and rendered.
- Page spacing is YAML-configurable:
  - `document.page.margin_left_mm` (default: `0`)
  - `document.page.track_gap_mm` (default: `0`)
- Track-header legend space now auto-expands based on curve count in each track.
- For continuous logs in PDF viewers, set `render.continuous_strip_page_height_mm` to export
  depth-continuous strip segments without vertical blank gaps while keeping readability.
- Matplotlib visuals can be configured in YAML using `render.matplotlib.style` instead of
  hardcoded renderer values.
- For DLIS array channels, `sample_axis.min/max` crops the actual waveform/raster columns to the
  selected window. It does not relabel the full array.
- When DLIS tool metadata exposes micro-time sampling, the loader derives the sample axis
  automatically. When vendor output still needs alignment tuning, the final user can override:
  - `sample_axis.source_origin`
  - `sample_axis.source_step`

Example VDL binding with explicit user-tunable sample axis:

```yaml
document:
  bindings:
    channels:
      - section: main
        channel: VDL
        track_id: vdl
        kind: raster
        profile: vdl
        style:
          colormap: gray_r
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
          stride: 6
          amplitude_scale: 0.28
          line_width: 0.16
```

Example instance-targeted fill between two rendered copies of the same channel:

```yaml
document:
  bindings:
    channels:
      - section: main
        channel: CBL
        track_id: cbl_fill
        kind: curve
        id: cbl_0_100
        scale: { kind: linear, min: 0, max: 100 }
        fill:
          kind: between_instances
          other_element_id: cbl_0_10
          label: Scale Effect
          crossover:
            enabled: true
            left_color: "#1f9d55"
            right_color: "#d64545"
      - section: main
        channel: CBL
        track_id: cbl_fill
        kind: curve
        id: cbl_0_10
        scale: { kind: linear, min: 0, max: 10 }
```

Example curve callout with section-relative repetition:

```yaml
document:
  bindings:
    channels:
      - section: main
        channel: CBL
        track_id: cbl
        kind: curve
        scale: { kind: linear, min: 0, max: 100 }
        callouts:
          - depth: 672
            label: CBL
            placement: top_and_bottom
            distance_from_top: 500
            distance_from_bottom: 500
            every: 1000
            side: right
            text_x: 0.83
            font_size: 10.5
```

Example reference-track overlays and local events:

```yaml
document:
  layout:
    log_sections:
      - id: main
        tracks:
          - id: depth_overlay
            kind: reference
            width_mm: 20
            reference:
              axis: depth
              define_layout: true
              unit: ft
              scale_ratio: 240
              events:
                - depth: 678
                  label: Casing Foot
                  tick_side: right
                  text_side: left
                  text_x: 0.72
            track_header:
              objects:
                - kind: scale
                  enabled: true
                  line_units: 1
                - kind: legend
                  enabled: true
                  line_units: 6
  bindings:
    channels:
      - section: main
        channel: TT
        track_id: depth_overlay
        kind: curve
        reference_overlay:
          mode: curve
          lane_start: 0.06
          lane_end: 0.24
      - section: main
        channel: TENS
        track_id: depth_overlay
        kind: curve
        reference_overlay:
          mode: indicator
          lane_start: 0.78
          lane_end: 0.94
      - section: main
        channel: CBL
        track_id: depth_overlay
        kind: curve
        reference_overlay:
          mode: ticks
          tick_side: left
          tick_length_ratio: 0.08
          threshold: 100
```

```yaml
render:
  backend: matplotlib
  output_path: ../workspace/renders/job.pdf
  dpi: 300
  matplotlib:
    style:
      track_header:
        background_color: "#efefef"
      track:
        x_tick_labelsize: 7.5
      grid:
        depth_major_linewidth: 0.8
```

## Real Data Demo

Use the master loader (single command for any log-file YAML):

```bash
uv run python -m well_log_os.cli render examples/cbl_main.log.yaml
```

Validate a log-file against the JSON Schema before rendering:

```bash
uv run python -m well_log_os.cli validate examples/cbl_main.log.yaml
```

Optional output override:

```bash
uv run python -m well_log_os.cli render examples/cbl_main.log.yaml -o out.pdf
```

Convenience wrapper:

```bash
uv run examples/real_data_demo.py
```

Or pass a specific log file:

```bash
uv run examples/real_data_demo.py examples/cbl_main.log.yaml
```

Array-track demo with synthetic VDL data and logfile config:

```bash
uv run examples/cbl_vdl_array_mvp_demo.py
```

Use [templates/wireline_base.template.yaml](templates/wireline_base.template.yaml) as a reusable
layout template, then create/modify job savefiles like
[examples/cbl_main.log.yaml](examples/cbl_main.log.yaml).
Keep local input/output assets under:
- `workspace/data/` for LAS/DLIS
- `workspace/renders/` for generated PDF/HTML/JSON outputs
The entire `workspace/` folder is excluded from git.
Note: DLIS normalization now supports scalar channels plus VDL/WF1-style array channels with
derived sample axes. Exact micro-time origin can still be tuned per savefile when matching
vendor-generated logs.

## Project Memory

- [docs/decision-log.md](docs/decision-log.md): agreed architectural and product decisions.
- [docs/roadmap.md](docs/roadmap.md): phased development plan and near-term priorities.
- [docs/rendering-workings.md](docs/rendering-workings.md): rendering flow and style-resolution model.

## License

Apache-2.0
