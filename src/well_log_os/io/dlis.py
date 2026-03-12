from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..errors import DependencyUnavailableError
from ..model import RasterChannel, ScalarChannel, WellDataset
from ..units import DEFAULT_UNITS

_DEPTH_MNEMONICS = {"DEPT", "DEPTH", "MD", "TDEP"}
_LENGTH_UNITS = {"mm", "cm", "m", "in", "ft"}
_SCALED_UNIT_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)\s*$"
)


def _normalize_depth_unit(unit_raw: str | None) -> tuple[str, float]:
    text = str(unit_raw or "").strip()
    if not text:
        return "m", 1.0

    match = _SCALED_UNIT_PATTERN.match(text)
    if match is not None:
        factor = float(match.group(1))
        base_unit = DEFAULT_UNITS.normalize(match.group(2))
        if base_unit in _LENGTH_UNITS:
            return base_unit, factor

    normalized = DEFAULT_UNITS.normalize(text)
    if normalized in _LENGTH_UNITS:
        return normalized, 1.0
    return "m", 1.0


def _normalize_value_unit(unit_raw: str | None) -> str | None:
    text = str(unit_raw or "").strip()
    return text or None


def _extract_well_metadata(logical_file) -> dict[str, str]:
    metadata: dict[str, str] = {}
    origins = getattr(logical_file, "origins", []) or []
    if not origins:
        return metadata

    origin = origins[0]
    mapping = {
        "WELL": ("well_name",),
        "COMP": ("company",),
        "FIELD": ("field_name",),
        "WELL_ID": ("well_id",),
        "FILE_ID": ("file_id",),
    }
    for target, candidates in mapping.items():
        for candidate in candidates:
            value = getattr(origin, candidate, None)
            if value in (None, ""):
                continue
            metadata[target] = str(value)
            break
    return metadata


def _build_scalar_channel(
    *,
    channel_name: str,
    channel_obj,
    depth: np.ndarray,
    depth_unit: str,
    values: np.ndarray,
    source: str,
):
    return ScalarChannel(
        mnemonic=channel_name,
        depth=depth,
        depth_unit=depth_unit,
        values=np.asarray(values, dtype=float),
        value_unit=_normalize_value_unit(getattr(channel_obj, "units", None)),
        description=str(getattr(channel_obj, "long_name", "") or ""),
        source=source,
        metadata={
            "original_mnemonic": channel_name,
            "dimension": list(getattr(channel_obj, "dimension", []) or []),
            "reprc": getattr(channel_obj, "reprc", None),
            "properties": list(getattr(channel_obj, "properties", []) or []),
            "source_object": str(getattr(channel_obj, "source", "") or ""),
            "channel_type": "scalar",
        },
    )


def _build_raster_channel(
    *,
    channel_name: str,
    channel_obj,
    depth: np.ndarray,
    depth_unit: str,
    values: np.ndarray,
    source: str,
):
    values_2d = np.asarray(values, dtype=float)
    if values_2d.ndim > 2:
        values_2d = values_2d.reshape(values_2d.shape[0], -1)
    sample_axis = np.arange(values_2d.shape[1], dtype=float)
    return RasterChannel(
        mnemonic=channel_name,
        depth=depth,
        depth_unit=depth_unit,
        values=values_2d,
        value_unit=_normalize_value_unit(getattr(channel_obj, "units", None)),
        sample_axis=sample_axis,
        sample_unit=None,
        sample_label="sample",
        description=str(getattr(channel_obj, "long_name", "") or ""),
        source=source,
        metadata={
            "original_mnemonic": channel_name,
            "dimension": list(getattr(channel_obj, "dimension", []) or []),
            "reprc": getattr(channel_obj, "reprc", None),
            "properties": list(getattr(channel_obj, "properties", []) or []),
            "source_object": str(getattr(channel_obj, "source", "") or ""),
            "channel_type": "raster",
        },
    )


def _should_replace_channel(existing, candidate) -> bool:
    if existing is None:
        return True
    if candidate.depth.shape[0] > existing.depth.shape[0]:
        return True
    if candidate.depth.shape[0] < existing.depth.shape[0]:
        return False
    if isinstance(existing, ScalarChannel) and isinstance(candidate, RasterChannel):
        return True
    return False


def load_dlis(path: str | Path):
    try:
        from dlisio import dlis
    except ImportError as exc:
        raise DependencyUnavailableError(
            "DLIS ingestion requires dlisio. Install well-log-os[dlis]."
        ) from exc

    dlis_path = Path(path)
    logical_files = dlis.load(str(dlis_path))
    if not logical_files:
        raise ValueError(f"No logical files found in DLIS source: {dlis_path}")

    logical_file = logical_files[0]
    well_metadata = _extract_well_metadata(logical_file)
    dataset = WellDataset(
        name=str(well_metadata.get("WELL") or dlis_path.stem),
        well_metadata=well_metadata,
        provenance={
            "source_path": str(dlis_path),
            "format": "DLIS",
            "logical_files": len(logical_files),
        },
    )

    frame_count = 0
    loaded_channels = 0
    for frame in getattr(logical_file, "frames", []) or []:
        frame_count += 1
        curves = frame.curves()
        dtype_names = set(curves.dtype.names or ())
        if not dtype_names:
            continue

        index_name = str(getattr(frame, "index", ""))
        if not index_name or index_name not in dtype_names:
            continue

        frame_channels = list(getattr(frame, "channels", []) or [])
        frame_channel_map = {str(channel.name): channel for channel in frame_channels}
        index_channel = frame_channel_map.get(index_name)
        depth_unit, depth_factor = _normalize_depth_unit(
            getattr(index_channel, "units", None) if index_channel is not None else None
        )
        depth = np.asarray(curves[index_name], dtype=float) * depth_factor

        for channel in frame_channels:
            channel_name = str(channel.name)
            if channel_name not in dtype_names:
                continue
            if channel_name.upper() == "FRAMENO":
                continue
            if channel_name == index_name:
                continue

            values = np.asarray(curves[channel_name], dtype=float)
            if values.ndim == 1:
                if channel_name.upper() in _DEPTH_MNEMONICS and np.allclose(
                    values, depth, equal_nan=True
                ):
                    continue
                candidate = _build_scalar_channel(
                    channel_name=channel_name,
                    channel_obj=channel,
                    depth=depth,
                    depth_unit=depth_unit,
                    values=values,
                    source=str(dlis_path),
                )
            else:
                candidate = _build_raster_channel(
                    channel_name=channel_name,
                    channel_obj=channel,
                    depth=depth,
                    depth_unit=depth_unit,
                    values=values,
                    source=str(dlis_path),
                )

            existing = dataset.channels.get(channel_name)
            if _should_replace_channel(existing, candidate):
                dataset.add_channel(candidate)
                loaded_channels += 1

    dataset.provenance["frames_processed"] = frame_count
    dataset.provenance["channels_loaded"] = len(dataset.channels)
    if not dataset.channels:
        raise ValueError(f"No channels could be normalized from DLIS source: {dlis_path}")
    return dataset
