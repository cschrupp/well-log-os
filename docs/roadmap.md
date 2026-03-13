# well_log_os Roadmap

Last updated: 2026-03-12

## Scope Summary

Build and mature an open-source Python toolkit for high-quality well-log display from LAS/DLIS with:

- print-first PDF generation
- interactive viewing
- template-driven layout control

Decision history is tracked in `docs/decision-log.md`.

## Current Baseline

- YAML templates define page, depth axis, tracks, styles, annotations.
- Logfile config is now track-first:
  - `document.layout.log_sections[*].tracks` defines layout
  - `document.bindings.channels[*]` assigns channels to tracks
- Legacy `auto_tracks` flow has been removed from the MVP path.
- Physical layout engine computes page windows and track frames.
- Track taxonomy is explicit with compatibility aliases:
  - `reference` (`depth` alias), `normal` (`curve` alias), `array` (`image` alias), `annotation`.
- Reference tracks can define layout axis values (unit/scale/steps) and render reference labels inside track bounds.
- Matplotlib renderer supports:
  - printable output
  - continuous single-page PDF
  - depth grid + markers/zones
  - structured track header object slots
  - array-lane raster controls (`colorbar`, `sample_axis`, waveform overlay)
  - waveform-only array rendering profile
  - VDL density rendering with grayscale amplitude mapping
  - DLIS-derived raster sample-axis metadata with savefile overrides
- Plotly renderer provides interactive visualization baseline.
- Synthetic example exists for fast iteration (`examples/synthetic_demo.py` + `examples/triple_combo.yaml`).
- Log-file schema validation is implemented (JSON Schema + CLI `validate`).
- YAML template/savefile inheritance is implemented for reusable log designs.
- Section placeholders exist in YAML for report composition:
  - `layout.heading`
  - `layout.comments`
  - `layout.log_sections`
  - `layout.tail`
- Data source routing is section-first with optional root fallback:
  - `layout.log_sections[*].data.source_path`
  - `layout.log_sections[*].data.source_format`

## MVP Architecture Gaps (layout/bindings model, 2026-03-10)

Needed next to complete the intended workflow:

- Render all `layout.log_sections` sequentially in one output artifact (implemented for matplotlib).
- Add real renderer primitives for `heading`, `comments`, and `tail` sections.
- Allow per-section depth windows and per-section page/layout settings.
- Add section-aware bindings (`binding.section`) in rendering, not only schema/validation.
- Add track-level default element properties to reduce binding repetition (style/scale/header display).
- Add validation for unbound required tracks and optional strict mode for empty tracks.
- Add CLI/report diagnostics that summarize section coverage and binding coverage.

## CBL Parity Gaps (from comparison test, 2026-03-09)

Compared against `workspace/renders/CBL_log_example.Pdf`, the current renderer is missing:

- Cover/disclaimer/contents pages and report-style front matter.
- Parameter-table sections (channel processing, depth zone, tool control).
- Advanced per-depth callouts/labels/arrows and event glyphs.
- Composite lane logic with custom legend/table blocks.
- Multi-page report composition mode (in addition to continuous strip mode).
- Richer visual theming and table/border styles for commercial-style output.
- Final-user calibration workflow for vendor-specific micro-time origin/width differences in VDLs.

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
- Baseline curve-wrap mode with per-segment color
  (`bindings.channels[*].wrap.enabled`, `bindings.channels[*].wrap.color`) for repeat-style display.

Near-term additions (next phases):

- Points/symbol mode (`marker` type, size, optional color-by-variable).
- Area-fill modes (left/right, baseline/fill rules).
- Vertical thresholds and advanced repeat/wrap controls (count, offset, style).
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
- Add explicit micro-time calibration helpers/presets for VDL sample-axis tuning.
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

- Extend multi-section composition engine:
  - render heading/comments/tail blocks between log sections
  - add section-specific break policies for continuous and paginated outputs
- Implement section object rendering:
  - heading blocks
  - comment/notes blocks
  - tail blocks
- Add per-section layout contracts:
  - optional `depth_range` per section
  - optional section-specific track-header height and spacing
- Add binding ergonomics:
  - track defaults (`curve_defaults` / `raster_defaults`)
  - optional per-track binding templates
  - strict validation mode for missing channels/empty tracks
- Add end-to-end examples using the new model:
  - single-section CBL
  - multi-section report with comments
  - mixed section layouts (reference + array + normal)
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
  - array-specific legends and advanced colorbar placement
  - per-track colormap presets
- Implement first `annotation` track objects:
  - depth-linked text labels
  - arrows/glyph markers
  - reserved-space/no-overlap policies
- Add reference-track focused examples:
  - depth-reference with curve overlay
  - time-reference sample
  - mixed reference + array + normal layout
- Add image-track template examples for CBL/VDL with curve overlays.
- Introduce report-section primitives (cover, disclaimer, contents, parameter tables).
- Add annotation primitives for callouts/arrows and depth-linked labels.
- Add a paginated report mode for mixed pages + log strips.
- Add style tokens/themes for branded tables, headers, and borders.
- Add a schema reference page for YAML keys (especially `track_header.objects`).
- Add visual regression checks for header slot layout and non-overlap.
- Tune default header `line_units` and font scaling against real CBL examples.
- Add one more end-to-end sample with multiple image tracks + overlays.
- Add savefile examples for VDL micro-time tuning against vendor outputs.

## Working Principles

- Treat each visual element as a typed object, not ad-hoc text.
- Preserve physically meaningful layout dimensions.
- Keep model decisions explicit and test-backed.
- Prefer stable defaults, then add opt-in complexity.
