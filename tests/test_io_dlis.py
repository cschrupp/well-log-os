from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np

from well_log_os.io.dlis import load_dlis
from well_log_os.model import RasterChannel, ScalarChannel


class _FakeChannel:
    def __init__(
        self,
        name: str,
        *,
        units: str = "",
        dimension: list[int] | None = None,
        long_name: str = "",
    ) -> None:
        self.name = name
        self.units = units
        self.dimension = dimension or [1]
        self.long_name = long_name
        self.reprc = 2
        self.properties = []
        self.source = "tool"


class _FakeFrame:
    def __init__(self, name: str, channels: list[_FakeChannel], curves: np.ndarray) -> None:
        self.name = name
        self.index = "TDEP"
        self.channels = channels
        self._curves = curves

    def curves(self) -> np.ndarray:
        return self._curves


class _FakeOrigin:
    def __init__(self) -> None:
        self.well_name = "TEST-WELL"
        self.company = "TEST-COMP"
        self.field_name = "TEST-FIELD"
        self.well_id = "TEST-ID"
        self.file_id = "TEST-FILE"


class _FakeLogical:
    def __init__(self, frames: list[_FakeFrame]) -> None:
        self.frames = frames
        self.origins = [_FakeOrigin()]


def _frame_curves(depth: np.ndarray, cbl: np.ndarray, vdl: np.ndarray | None = None) -> np.ndarray:
    fields: list[tuple[str, str]] = [("FRAMENO", "<i4"), ("TDEP", "<f4"), ("CBL", "<f4")]
    if vdl is not None:
        fields.append(("VDL", "<f4", (vdl.shape[1],)))
    curves = np.zeros(depth.shape[0], dtype=fields)
    curves["FRAMENO"] = np.arange(depth.shape[0], dtype=np.int32)
    curves["TDEP"] = depth.astype(np.float32)
    curves["CBL"] = cbl.astype(np.float32)
    if vdl is not None:
        curves["VDL"] = vdl.astype(np.float32)
    return curves


class DLISIOTests(unittest.TestCase):
    def test_load_dlis_normalizes_scalar_and_raster_channels(self) -> None:
        depth_a = np.asarray([1200.0, 1080.0, 960.0], dtype=float)
        depth_b = np.asarray([1200.0, 1140.0, 1080.0, 1020.0, 960.0], dtype=float)
        cbl_a = np.asarray([10.0, 20.0, 30.0], dtype=float)
        cbl_b = np.asarray([11.0, 21.0, 31.0, 41.0, 51.0], dtype=float)
        vdl = np.linspace(-1.0, 1.0, depth_b.size * 4).reshape(depth_b.size, 4)

        index_ch = _FakeChannel("TDEP", units="0.1 in", dimension=[1], long_name="Depth")
        cbl_ch = _FakeChannel("CBL", units="mV", dimension=[1], long_name="CBL")
        vdl_ch = _FakeChannel("VDL", units="amplitude", dimension=[4], long_name="VDL")

        frame_a = _FakeFrame(
            "A",
            [index_ch, cbl_ch],
            _frame_curves(depth_a, cbl_a),
        )
        frame_b = _FakeFrame(
            "B",
            [index_ch, cbl_ch, vdl_ch],
            _frame_curves(depth_b, cbl_b, vdl=vdl),
        )
        logical = _FakeLogical([frame_a, frame_b])

        fake_dlis_module = types.SimpleNamespace(load=lambda _: [logical])
        fake_pkg = types.SimpleNamespace(dlis=fake_dlis_module)

        with patch.dict("sys.modules", {"dlisio": fake_pkg}):
            dataset = load_dlis("fake.dlis")

        self.assertEqual(dataset.name, "TEST-WELL")
        self.assertEqual(dataset.well_metadata["WELL"], "TEST-WELL")
        self.assertEqual(dataset.well_metadata["COMP"], "TEST-COMP")
        self.assertIn("CBL", dataset.channels)
        self.assertIn("VDL", dataset.channels)

        cbl = dataset.get_channel("CBL")
        self.assertIsInstance(cbl, ScalarChannel)
        self.assertEqual(cbl.depth_unit, "in")
        self.assertEqual(cbl.depth.shape[0], 5)
        self.assertAlmostEqual(float(cbl.depth[0]), 120.0)

        vdl_channel = dataset.get_channel("VDL")
        self.assertIsInstance(vdl_channel, RasterChannel)
        self.assertEqual(vdl_channel.values.shape, (5, 4))
        self.assertEqual(vdl_channel.sample_axis.shape[0], 4)
        self.assertEqual(dataset.provenance["format"], "DLIS")
        self.assertEqual(dataset.provenance["frames_processed"], 2)


if __name__ == "__main__":
    unittest.main()
