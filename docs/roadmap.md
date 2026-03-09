# well_log_os Roadmap

Last updated: 2026-03-09

## Scope Summary

Build and mature an open-source Python toolkit for high-quality well-log display from LAS/DLIS with:

- print-first PDF generation
- interactive viewing
- template-driven layout control

Decision history is tracked in `docs/decision-log.md`.

## Current Baseline

- YAML templates define page, depth axis, tracks, styles, annotations.
- Physical layout engine computes page windows and track frames.
- Matplotlib renderer supports:
  - printable output
  - continuous single-page PDF
  - depth grid + markers/zones
  - structured track header object slots
- Plotly renderer provides interactive visualization baseline.
- Synthetic example exists for fast iteration (`examples/synthetic_demo.py` + `examples/triple_combo.yaml`).
- Log-file schema validation is implemented (JSON Schema + CLI `validate`).
- YAML template/savefile inheritance is implemented for reusable log designs.

## Development Plan

### Phase A: Core Stabilization (current)

- Harden template schema validation.
- Expand renderer regression tests (layout + header behavior).
- Add more realistic template examples (triple combo, CBL-style strips).
- Keep parity between model and docs.

### Phase B: Rendering Quality

- Improve default typography and spacing for print fidelity.
- Add richer depth column behavior (interval labels, optional callout lanes).
- Improve raster controls (color limits, interpolation presets, palettes).
- Add configurable track border styles and divider systems.

### Phase C: Interactive Viewer Maturity

- Expand interactive controls (track visibility, zoom presets, annotations).
- Add synchronized depth cursor and track crosshair behavior.
- Support export/import of viewer state from templates.

### Phase D: I/O and Standardization

- Harden LAS and DLIS adapters for real-world edge cases.
- Verify dependency licenses and document compatibility notes.
- Define template schema versioning and migration notes.

### Phase E: Packaging and Ecosystem

- Improve docs with API examples and cookbook-style templates.
- Publish contributor guide and architecture notes.
- Prepare roadmap for web service/UI packaging.

## Immediate Next Tasks

- Add a schema reference page for YAML keys (especially `track_header.objects`).
- Add visual regression checks for header slot layout and non-overlap.
- Tune default header `line_units` and font scaling against real CBL examples.
- Add one more end-to-end sample with multiple image tracks + overlays.

## Working Principles

- Treat each visual element as a typed object, not ad-hoc text.
- Preserve physically meaningful layout dimensions.
- Keep model decisions explicit and test-backed.
- Prefer stable defaults, then add opt-in complexity.
