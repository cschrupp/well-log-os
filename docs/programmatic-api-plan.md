# Programmatic API Plan

Last updated: 2026-03-18

## Goal

Add a programmatic API so users can:

- ingest raw or computed log data from numpy/pandas results
- build report/layout objects without writing YAML by hand
- render full reports or partial views for notebooks and research workflows
- optionally serialize the resulting document back to YAML

This phase is intentionally separate from interactive-viewer work. The immediate target is a clean
Python API over the current data model and renderers.

## Design Principles

- The canonical configuration object is the in-memory model, not YAML.
- YAML remains a first-class serialization format, not the core architecture.
- Data ingestion, document composition, and rendering are separate layers.
- Notebook usage must not require temporary YAML or LAS/DLIS files.
- Added data must carry explicit reference-axis and unit information.

## Target Layering

### 1. Data Layer

Owns normalized datasets and channels.

Core objects:

- `WellDataset`
- `ScalarChannel`
- `ArrayChannel`
- `RasterChannel`

Responsibilities:

- file ingestion (`.las`, `.dlis`)
- computed-channel ingestion from numpy/pandas
- validation of shape, index, units, and metadata
- optional merge/reindex/unit-conversion helpers

### 2. Composition Layer

Owns layout and report composition.

Core objects:

- `LogDocument`
- report blocks (`heading`, `remarks`, `tail`)
- sections
- tracks
- bindings

Responsibilities:

- define how a report should look
- connect datasets/channels to tracks
- remain independent of file format and renderer

### 3. Render Layer

Owns final output generation.

Responsibilities:

- full-document render
- section render
- track render
- depth/time window render
- notebook-friendly image output

## MVP Scope

Implement in this order:

1. dataset ingestion API
2. programmatic document builder
3. full-document render API
4. partial render API
5. notebook-friendly outputs
6. YAML round-trip helpers

Out of scope for this phase:

- interactive Plotly/Bokeh editing
- web service endpoints
- collaborative session/state management
- aggressive render caching

## Concrete Implementation Checklist

### Phase 1. Stabilize the data boundary

Goal: make the existing dataset/channel model the formal ingestion target.

Tasks:

- Confirm `WellDataset` is the only internal dataset container.
- Confirm scalar and array/raster channels are the only ingest targets.
- Add a small validation entrypoint on datasets/channels.
- Document the required contract for computed channels:
  - numeric reference axis
  - explicit axis unit
  - monotonic axis
  - shape compatibility
  - explicit value unit when known

Likely files:

- [dataset.py](/home/user/projects/well_log_os/src/well_log_os/model/dataset.py)
- [channels.py](/home/user/projects/well_log_os/src/well_log_os/model/channels.py)

Acceptance:

- all ingestion paths produce the same channel classes now used by LAS/DLIS loaders

### Phase 2. Add a dataset-ingestion API

Goal: let users add computed results back into a dataset cleanly.

New module:

- `src/well_log_os/api/dataset.py`

Public surface:

- `DatasetBuilder`
- `create_dataset(...)`
- dataset/channel add helpers

Recommended methods:

- `add_curve(...)`
- `add_array(...)`
- `add_raster(...)`
- `add_series(...)`
- `add_dataframe(...)`
- `add_or_replace_channel(...)`
- `merge(...)`

Recommended API shape:

```python
from well_log_os.api.dataset import DatasetBuilder

ds = DatasetBuilder(name="processed").build()
ds.add_curve(
    mnemonic="PHIE",
    values=phi,
    index=depth_ft,
    index_unit="ft",
    value_unit="v/v",
)
```

Array example:

```python
ds.add_array(
    mnemonic="VDL_ENH",
    values=waveforms,
    index=depth_ft,
    index_unit="ft",
    sample_axis=time_us,
    sample_unit="us",
    value_unit="amplitude",
)
```

Important rule:

- use `index` in the public ingestion API, not hard-coded `depth`
- depth/time semantics belong to the composition layer

Acceptance:

- users can create valid internal datasets entirely from computed arrays

### Phase 3. Add pandas/numpy adapters

Goal: support common notebook workflows directly.

New functionality:

- `add_series(...)`
- `add_dataframe(...)`
- dataframe index or named index-column support

Required behavior:

- no silent unit guessing
- no silent assumption that dataframe index is depth
- support subset and regridded results as long as their own basis is valid

Examples:

```python
ds.add_dataframe(df, use_index=True, index_unit="ft")
```

```python
ds.add_dataframe(
    df,
    index_column="TIME",
    index_unit="ms",
    curves={"ATTN": {"value_unit": "dB"}},
)
```

Acceptance:

- a researcher can compute new curves in pandas and add them to a dataset in one step

### Phase 4. Add validation and alignment helpers

Goal: make computed-channel ingestion safe.

Hard errors:

- missing index
- non-numeric index
- shape mismatch
- empty arrays
- missing sample axis for array/raster channels

Warnings:

- duplicate index samples
- non-uniform sampling
- null-heavy channels
- unexpected descending order

Explicit transforms:

- `sort_index()`
- `convert_index_unit(...)`
- `reindex_to(...)`
- `validate()`

Important rule:

- require a valid own basis, not a shared global basis
- alignment to other channels must be explicit

Acceptance:

- invalid computed data is rejected early and clearly

### Phase 5. Add serialization helpers

Goal: make YAML a round-trip format, not the only authoring path.

New module:

- `src/well_log_os/api/serialize.py`

Functions:

- `document_to_dict(...)`
- `document_from_dict(...)`
- `document_to_yaml(...)`
- `document_from_yaml(...)`

Requirements:

- work from in-memory model objects
- preserve report/layout structure
- support saving builder-created documents as YAML

Acceptance:

- Python-built documents can be saved as YAML and loaded back

### Phase 6. Add a programmatic document builder

Goal: let users compose logs in Python without writing YAML-shaped dicts.

New module:

- `src/well_log_os/api/builder.py`

Public objects:

- `LogBuilder`
- `SectionBuilder`
- `TrackBuilder`

Recommended usage:

```python
from well_log_os.api import LogBuilder

builder = LogBuilder(template="wireline_base")
builder.set_depth_range(8290, 8460, unit="ft")
builder.set_heading(...)
builder.add_remarks(...)
section = builder.add_section("main", dataset=ds_main)
section.add_reference_track(...)
section.add_curve(...)
section.add_raster(...)
doc = builder.build()
```

Important rule:

- the API should speak in domain terms, not expose raw YAML structure by default

Bad:

```python
builder.add_binding({"kind": "curve", ...})
```

Good:

```python
section.add_curve("CBL", track="cbl", scale=(0, 100), color="black")
```

Acceptance:

- at least one existing example can be rebuilt entirely from Python

### Phase 7. Allow in-memory datasets in sections

Goal: remove the current dependency on `source_path` for programmatic work.

Needed behavior:

- sections can point to loaded file datasets
- or in-memory merged/processed datasets

Recommended API:

```python
builder.add_section("main", dataset=ds_main)
builder.add_section("repeat", dataset=ds_repeat)
```

Acceptance:

- full report render works without temporary input files

### Phase 8. Add render API

Goal: make rendering callable directly from Python.

New module:

- `src/well_log_os/api/render.py`

Public functions:

- `render_pdf(...)`
- `render_document(...)`
- `render_section(...)`
- `render_track(...)`
- `render_window(...)`

Output modes:

- file path
- bytes
- in-memory figure where practical

Acceptance:

- full-report rendering no longer requires CLI or logfile path

### Phase 9. Add partial renders

Goal: support research workflows and notebook previews.

Partial scopes:

- one section
- one track
- one depth/time window

Important rule:

- partial rendering should filter/view the same document model
- do not create a separate “mini document” architecture

Acceptance:

- users can inspect a processed interval without generating the full report

### Phase 10. Add notebook-friendly outputs

Goal: zero-friction inline use in Jupyter.

Functions:

- `render_png_bytes(...)`
- `render_svg_bytes(...)` if practical
- `render_matplotlib_figure(...)` where useful

Notebook example:

```python
from IPython.display import Image
from well_log_os.api import render_section_png

Image(data=render_section_png(document, datasets, section_id="main"))
```

Acceptance:

- notebook users can render inline without temporary files

### Phase 11. Add tests

New test groups:

- dataset ingestion from numpy
- dataset ingestion from pandas
- array/raster validation
- dataset merge/update behavior
- builder-generated document creation
- full and partial render API
- YAML round-trip for builder-generated documents

Suggested files:

- `tests/test_api_dataset.py`
- `tests/test_api_builder.py`
- `tests/test_api_render.py`
- `tests/test_api_serialize.py`

Acceptance:

- builder + ingestion + render pipeline is regression-tested

### Phase 12. Add examples

Recommended examples:

- `examples/api_dataset_ingest_demo.py`
- `examples/api_log_builder_demo.py`
- `examples/api_partial_render_demo.py`

Minimum example content:

1. load DLIS/LAS
2. compute derived curves in pandas/numpy
3. add them back into a dataset
4. build a report in Python
5. render a full PDF and an inline PNG

## Acceptance Criteria For The Whole Phase

The API phase is successful if a user can:

1. load LAS/DLIS into a dataset
2. compute new channels in pandas/numpy
3. add them as validated channels with explicit basis and units
4. build a log programmatically
5. render a full report
6. render one section or one depth/time window
7. display a PNG inline in Jupyter
8. optionally save the document as YAML

## Implementation Order

Recommended order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7
8. Phase 8
9. Phase 9
10. Phase 10
11. Phase 11
12. Phase 12

## References

Welly is a useful reference for the notebook/data-science side of the problem:

- `Curve` construction from `data + index`
- empty container creation
- pandas export/import workflows
- curve/data-matrix extraction

Useful references:

- <https://code.agilescientific.com/welly/userguide/Curves.html>
- <https://code.agilescientific.com/welly/userguide/Wells.html>
- <https://github.com/agilescientific/welly>

We should borrow the ergonomics, not the looser validation model. `well_log_os` still needs
stricter basis/unit semantics because rendering depends on physical layout correctness.
