"""Regression tests for MPI HW dump assembly and existing single-file input."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset

from plot_unet_validation import read_history


class BoutHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def write_dumps(self, nxpe=2, nype=2):
        expected = np.arange(2 * 3 * (2 * nxpe) * (2 * nype) * 3).reshape(
            2, 3, 2 * nxpe, 2 * nype, 3
        ).astype(float)
        for rank in range(nxpe * nype):
            px, py = rank % nxpe, rank // nxpe
            with Dataset(self.path / f"BOUT.dmp.{rank}.nc", "w") as d:
                for name, size in (("t", 2), ("x", 4), ("y", 4), ("z", 3)):
                    d.createDimension(name, size)
                metadata = dict(NXPE=nxpe, NYPE=nype, MXSUB=2, MYSUB=2,
                                MXG=1, MYG=1, MZ=3, BOUT_VERSION=5,
                                nx=2 * nxpe + 2, ny=2 * nype)
                for name, value in metadata.items():
                    d.createVariable(name, "i4")[...] = value
                d.createVariable("t_array", "f8", ("t",))[:] = [2, 1]
                for channel, name in enumerate(("n", "phi", "vort")):
                    v = d.createVariable(name, "f8", ("t", "x", "y", "z"))
                    v[:] = -9999  # Guards must never appear in assembled fields.
                    v[:, 1:3, 1:3, :] = expected[
                        :, channel, 2 * px:2 * px + 2, 2 * py:2 * py + 2, :
                    ]
        return expected

    def test_xy_decomposition_and_nonzero_rank_input(self):
        expected = self.write_dumps()
        for source in (self.path, self.path / "BOUT.dmp.3.nc"):
            times, steps, fields = read_history(source, 0, "netcdf", 3)
            np.testing.assert_array_equal(times, [1, 2])
            np.testing.assert_array_equal(steps, [1, 0])
            np.testing.assert_array_equal(fields, expected[::-1, :, :, 3, :])

    def test_single_rank(self):
        expected = self.write_dumps(1, 1)
        _, _, fields = read_history(self.path, 0, "netcdf", 1)
        np.testing.assert_array_equal(fields, expected[::-1, :, :, 1, :])

    def test_missing_rank(self):
        self.write_dumps()
        (self.path / "BOUT.dmp.2.nc").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing MPI dump"):
            read_history(self.path, 0)

    def test_mismatched_times(self):
        self.write_dumps()
        with Dataset(self.path / "BOUT.dmp.2.nc", "a") as d:
            d["t_array"][:] = [2, 3]
        with self.assertRaisesRegex(ValueError, "inconsistent output times"):
            read_history(self.path, 0)

    def test_invalid_y(self):
        self.write_dumps()
        with self.assertRaises(IndexError):
            read_history(self.path, 0, "netcdf", 4)


if __name__ == "__main__":
    unittest.main()
