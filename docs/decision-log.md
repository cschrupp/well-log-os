# well_log_os Decision Log

Last updated: 2026-03-09

## Purpose

This file records project decisions that should remain stable unless explicitly revised.

## Product and Scope Decisions

- Language: Python only.
- Primary objective: display/render quality over ingestion complexity.
- Outputs: both static printable output and interactive viewing.
- Growth path: start as a Python library, then extend to web apps.
- Domain focus: wireline logs, including triple-combo, image/raster logs, and petrophysical outputs.

## Architecture Decisions

- Renderer-first architecture with three layers:
  - `WellDataset` for normalized data
  - `LogDocument` for layout and visualization specs
  - renderer backends for output generation
- Templates are YAML-driven and define page geometry, depth behavior, tracks, and annotations.

## Data Model Decisions

- Image tracks are first-class tracks.
- Raster/image data must not be forced into curve tracks.
- Curve tracks reject raster elements.
- Curves can overlay inside image tracks.
- Log configurations are persisted in a dedicated YAML "log-file" spec (not hardcoded in scripts).
- Track layouts are configured first in YAML (`document.layout.log_sections[*].tracks`),
  then channels are assigned via `document.bindings.channels`.
- Rendering is executed through a master log-file pipeline/CLI, not per-file loaders.
- Log-file YAML is validated with a JSON Schema before parsing/building.
- CLI includes a dedicated `validate` command for logfile preflight checks.
- Log-file configs support YAML template inheritance (`template.path`) plus savefile overrides.
- Data-source ownership is section-first:
  - preferred source location is `document.layout.log_sections[*].data`
  - top-level `data` is optional default/fallback
- Track header data is modeled as typed objects (`title`, `scale`, `legend`), not ad-hoc text.
- Track header objects support:
  - `enabled` to show/hide content
  - `reserve_space` to preserve layout slot when hidden
  - `line_units` to control relative vertical allocation

## Rendering Decisions

- Continuous log-strip mode is supported (`page.continuous: true`) for single long-page PDFs.
- Track header band has dedicated physical space (`page.track_header_height_mm`).
- Depth grid cadence is template-controlled (`depth.major_step`, `depth.minor_step`).
- Markers and zones are top-level annotation objects.
- Header slot geometry is reserved to prevent overlaps.

## Tooling and Process Decisions

- Dependency and environment management: `uv`.
- Formatting and linting: `ruff`.
- Tests: `unittest` executed through `uv run`.
- License: Apache-2.0.
- Optional dependencies:
  - LAS: `lasio`, `welly`
  - DLIS: `dlisio`
  - static PDF rendering: `matplotlib`
  - interactive rendering: `plotly`

## Python Compatibility Decision

- Package metadata currently states Python `>=3.11`.
- CI validates 3.11, 3.12, 3.13, and 3.14.

## Revision Rule

When a decision changes, update this file with:

- previous decision
- new decision
- effective date
- rationale
