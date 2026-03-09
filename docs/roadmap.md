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
- Track taxonomy is explicit with compatibility aliases:
  - `reference` (`depth` alias), `normal` (`curve` alias), `array` (`image` alias), `annotation`.
- Reference tracks can define layout axis values (unit/scale/steps) and render reference labels inside track bounds.
- Matplotlib renderer supports:
  - printable output
  - continuous single-page PDF
  - depth grid + markers/zones
  - structured track header object slots
- Plotly renderer provides interactive visualization baseline.
- Synthetic example exists for fast iteration (`examples/synthetic_demo.py` + `examples/triple_combo.yaml`).
- Log-file schema validation is implemented (JSON Schema + CLI `validate`).
- YAML template/savefile inheritance is implemented for reusable log designs.

## CBL Parity Gaps (from comparison test, 2026-03-09)

Compared against `workspace/renders/CBL_log_example.Pdf`, the current renderer is missing:

- VDL image/raster lane support from DLIS normalization.
- Cover/disclaimer/contents pages and report-style front matter.
- Parameter-table sections (channel processing, depth zone, tool control).
- Advanced per-depth callouts/labels/arrows and event glyphs.
- Composite lane logic with custom legend/table blocks.
- Multi-page report composition mode (in addition to continuous strip mode).
- Richer visual theming and table/border styles for commercial-style output.

## Curve Properties Matrix (Techlog Comparison, 2026-03-09)

Implementable now (already in project):

- Horizontal limits (`min`/`max`) and scale type (`linear`/`log`) per curve.
- Inversion (`reverse`) per curve.
- Curve line color, thickness, style, opacity.
- Multi-curve track overlays with independent per-curve scaling.
- Per-curve header rows (name + scale/limits + unit).
- Per-curve header display toggles (`show_name`, `show_unit`, `show_limits`, `show_color`).
- Optional header `divisions` object for per-track scale ticks in a dedicated header line.
- Value-label mode with step, number format, precision, alignment, font.
- Header typography and frame styling through `render.matplotlib.style`.

Near-term additions (next phases):

- Points/symbol mode (`marker` type, size, optional color-by-variable).
- Area-fill modes (left/right, baseline/fill rules).
- Vertical thresholds and repeat/wrap display behavior.
- Per-curve decimation/display optimization policy.
- Per-curve number formatting for header limits (separate from label mode).

Longer-term / UI-centric:

- Full variable-management metadata blocks (family/alias/version history).
- Interactive graphical editing and audit/history panel.
- Per-curve reference-window controls (top/base limits in alternate units).

## Development Plan

### Phase A: Core Stabilization (current)

- Harden template schema validation.
- Expand renderer regression tests (layout + header behavior).
- Add more realistic template examples (triple combo, CBL-style strips).
- Keep parity between model and docs.

### Phase B: Rendering Quality

- Improve default typography and spacing for print fidelity.
- Add richer depth column behavior (interval labels, optional callout lanes).
- Add dedicated data-driven tick primitives for reference tracks (major/minor + event ticks).
- Add configurable reference-value placement/alignment policies inside track (left/center/right, collision handling).
- Improve raster controls (color limits, interpolation presets, palettes).
- Add configurable track border styles and divider systems.

### Phase C: Interactive Viewer Maturity

- Expand interactive controls (track visibility, zoom presets, annotations).
- Add synchronized depth cursor and track crosshair behavior.
- Support export/import of viewer state from templates.
- Add UI-ready edit model for reference-track properties (header toggles, number format, grids).

### Phase D: I/O and Standardization

- Harden LAS and DLIS adapters for real-world edge cases.
- Verify dependency licenses and document compatibility notes.
- Define template schema versioning and migration notes.
- Add explicit schema blocks for `reference`, `array`, and `annotation` track-specific configs.

### Phase E: Packaging and Ecosystem

- Improve docs with API examples and cookbook-style templates.
- Publish contributor guide and architecture notes.
- Prepare roadmap for web service/UI packaging.

## Immediate Next Tasks

- Complete reference-track properties from parity screenshots:
  - header orientation/alignment controls
  - values orientation modes
  - appearance block (track/header bg colors, font settings)
  - print/layout width semantics
- Add data-driven ticks on reference tracks:
  - ticks from selected channels/events
  - formatting and filtering rules
  - optional header summaries
- Implement explicit `array` track options beyond raster baseline:
  - sample-axis labeling
  - array-specific legends/colorbars
  - per-track colormap presets
- Implement first `annotation` track objects:
  - depth-linked text labels
  - arrows/glyph markers
  - reserved-space/no-overlap policies
- Add reference-track focused examples:
  - depth-reference with curve overlay
  - time-reference sample
  - mixed reference + array + normal layout
- Implement DLIS normalization for array/raster channels (VDL-first target).
- Add image-track template examples for CBL/VDL with curve overlays.
- Introduce report-section primitives (cover, disclaimer, contents, parameter tables).
- Add annotation primitives for callouts/arrows and depth-linked labels.
- Add a paginated report mode for mixed pages + log strips.
- Add style tokens/themes for branded tables, headers, and borders.
- Add a schema reference page for YAML keys (especially `track_header.objects`).
- Add visual regression checks for header slot layout and non-overlap.
- Tune default header `line_units` and font scaling against real CBL examples.
- Add one more end-to-end sample with multiple image tracks + overlays.

## Working Principles

- Treat each visual element as a typed object, not ad-hoc text.
- Preserve physically meaningful layout dimensions.
- Keep model decisions explicit and test-backed.
- Prefer stable defaults, then add opt-in complexity.
