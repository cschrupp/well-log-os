# well_log_os

`well_log_os` is an open-source Python toolkit for building printable and interactive well-log layouts from LAS and DLIS data.

The project is intentionally renderer-first:
- normalize subsurface data into typed channels
- describe the sheet using templates and layout specs
- render the same document to static and interactive backends

## Status

This repository currently contains the initial scaffold:
- normalized data objects for scalar, array, and raster channels
- a printable log document model with tracks, styles, headers, and footers
- YAML template loading
- a physical page layout engine
- optional `matplotlib` and `plotly` renderer backends
- optional LAS and DLIS ingestion adapters

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
- `auto_tracks.tracks` uses channel-aware merging (match by `channel`).
- Track entries can be short (`- GR`) when `auto_tracks.default_configure` is defined.
- `auto_tracks` can group multiple curves in one track by reusing the same `configure.id`.
- Template YAML files can be partial; the merged savefile result is what gets validated and rendered.
- Page spacing is YAML-configurable:
  - `document.page.margin_left_mm` (default: `0`)
  - `document.page.track_gap_mm` (default: `0`)
- Track-header legend space now auto-expands based on curve count in each track.
- For continuous logs in PDF viewers, set `render.continuous_strip_page_height_mm` to export
  depth-continuous strip segments without vertical blank gaps while keeping readability.
- Matplotlib visuals can be configured in YAML using `render.matplotlib.style` instead of
  hardcoded renderer values.

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

Use [templates/wireline_base.template.yaml](templates/wireline_base.template.yaml) as a reusable
layout template, then create/modify job savefiles like
[examples/cbl_main.log.yaml](examples/cbl_main.log.yaml).
Keep local input/output assets under:
- `workspace/data/` for LAS/DLIS
- `workspace/renders/` for generated PDF/HTML/JSON outputs
The entire `workspace/` folder is excluded from git.
Note: LAS ingestion is implemented; DLIS normalization is still scaffolded.

## Project Memory

- [docs/decision-log.md](docs/decision-log.md): agreed architectural and product decisions.
- [docs/roadmap.md](docs/roadmap.md): phased development plan and near-term priorities.
- [docs/rendering-workings.md](docs/rendering-workings.md): rendering flow and style-resolution model.

## License

Apache-2.0
