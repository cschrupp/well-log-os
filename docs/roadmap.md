# well_log_os Roadmap

Last updated: 2026-03-18

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
  - fixed-height curve and array property groups in track headers
  - curve-owned fills:
    - `between_curves`
    - `between_instances`
    - `to_lower_limit`
    - `to_upper_limit`
    - `baseline_split`
  - track-header fill indicators that reflect actual fill semantics
  - in-track curve callouts with edge avoidance and curve-aware placement
  - section-relative repeated callouts from top/bottom anchors
  - reference-track scalar overlay modes (`curve`, `indicator`, `ticks`)
  - reference-track-local event objects for non-channel markers
  - reference-track headers that keep the layout scale row while exposing overlay properties
  - annotation-track `interval`, `text`, `marker`, `arrow`, and `glyph` rendering
  - annotation-track dedicated label lanes and collision-aware label placement for marker/arrow labels
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
  - `layout.remarks`
  - `layout.log_sections`
  - `layout.tail`
- Data source routing is section-first with optional root fallback:
  - `layout.log_sections[*].data.source_path`
  - `layout.log_sections[*].data.source_format`

## MVP Architecture Gaps (layout/bindings model, 2026-03-10)

Needed next to complete the intended workflow:

- Render all `layout.log_sections` sequentially in one output artifact (implemented for matplotlib).
- Refine report composition beyond the current `heading` / `remarks` / `tail` baseline.
- Allow per-section depth windows and per-section page/layout settings.
- Add section-aware bindings (`binding.section`) in rendering, not only schema/validation.
- Add track-level default element properties to reduce binding repetition (style/scale/header display).
- Add validation for unbound required tracks and optional strict mode for empty tracks.
- Add CLI/report diagnostics that summarize section coverage and binding coverage.

## Next Phase: Programmatic API and Dataset Ingestion (2026-03-18)

This is now the active next phase after the current report/track MVP baseline.

Core goals:

- allow researchers to add computed channels from numpy/pandas back into `WellDataset`
- allow logs to be composed in Python without hand-authoring YAML
- support full and partial renders for notebook workflows
- keep YAML as serialization, not the only authoring surface

Planned public modules:

- `well_log_os.api.dataset`
- `well_log_os.api.builder`
- `well_log_os.api.render`
- `well_log_os.api.serialize`

Planned delivery order:

1. dataset-ingestion API
2. pandas/numpy adapters
3. validation and alignment helpers
4. programmatic document builder
5. in-memory dataset-backed sections
6. full render API
7. partial render API
8. notebook-friendly outputs
9. examples and tests

Detailed checklist:

- [docs/programmatic-api-plan.md](programmatic-api-plan.md)

## CBL Parity Gaps (from comparison test, 2026-03-09)

Compared against `workspace/renders/CBL_log_example.Pdf`, the current renderer is missing:

- Cover/disclaimer/contents pages and report-style front matter.
- Richer remarks/notes layouts for the lower half of the first report page.
- Parameter-table sections (channel processing, depth zone, tool control).
- Advanced annotation packing for very dense tracks beyond the current dedicated-label-lane model.
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
- Per-curve header display toggles
  (`show_name`, `show_unit`, `show_limits`, `show_color`, `wrap_name`).
- Optional header `divisions` object for per-track scale ticks in a dedicated header line.
- Value-label mode with step, number format, precision, alignment, font.
- Header typography and frame styling through `render.matplotlib.style`.
- Baseline curve-wrap mode with per-segment color
  (`bindings.channels[*].wrap.enabled`, `bindings.channels[*].wrap.color`) for repeat-style display.
- Curve fill modes:
  - `between_curves`
  - `between_instances`
  - `to_lower_limit`
  - `to_upper_limit`
  - `baseline_split`
- Track-header fill indicators that preview the rendered fill behavior.
- In-track curve callouts with section-relative repetition from top/bottom anchors.

Near-term additions (next phases):

- Points/symbol mode (`marker` type, size, optional color-by-variable).
- Vertical thresholds and advanced repeat/wrap controls (count, offset, style).
- Per-curve decimation/display optimization policy.
- Per-curve number formatting for header limits (separate from label mode).
- Per-callout priority/required rules for dense tracks.
- Page-relative callout repetition mode, if needed alongside the current section-relative mode.

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

- Begin the programmatic API phase:
  - dataset-ingestion API for computed channels
  - pandas/numpy adapters
  - programmatic document builder
  - partial render API for section/track/window scopes
  - notebook-friendly output helpers
  - YAML round-trip support for builder-created documents
- Document and expand reference/depth track usage:
  - reference-track overlay YAML examples
  - full-length reference overlay example
  - compatibility notes for depth/time reference units versus overlaid curve units
- Improve reference-track local event handling:
  - optional collision-aware placement for event labels versus curve callouts
  - optional header summaries for local reference events
- Polish annotation-track examples and contracts:
  - clearer dedicated-lane examples for dense lithofacies/event tracks
  - optional multi-lane event-label examples
  - collision-aware placement notes in the schema reference
- Extend multi-section composition engine:
  - render heading/remarks/tail blocks between log sections
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
  - multi-section report with remarks
  - mixed section layouts (reference + array + normal)
- Complete reference-track properties from parity screenshots:
  - header orientation/alignment controls
  - values orientation modes
  - appearance block (track/header bg colors, font settings)
  - print/layout width semantics
- Implement explicit `array` track options beyond raster baseline:
  - array-specific legends and advanced colorbar placement
  - per-track colormap presets
- Add reference-track focused examples:
  - depth-reference with curve overlay
  - time-reference sample
  - mixed reference + array + normal layout
- Add image-track template examples for CBL/VDL with curve overlays.
- Introduce report-section primitives (cover, disclaimer, contents, parameter tables).
- Add annotation-track polish beyond the current object baseline:
  - richer label-lane contracts
  - optional repacking rules for dense interval/text/glyph compositions
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
