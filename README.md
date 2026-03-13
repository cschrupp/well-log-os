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

## Architecture

The package separates three layers:
- `WellDataset`: data and metadata normalized from LAS/DLIS inputs
- `LogDocument`: page, depth, track, and annotation specifications
- renderers: backend-specific drawing implementations that consume the same document

Track types are explicit: `reference`, `normal`, `array`, and `annotation`
(with compatibility aliases `depth`, `curve`, `image`).
Array tracks can host raster data and scalar overlays, while normal/reference tracks do not accept raster elements.
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
