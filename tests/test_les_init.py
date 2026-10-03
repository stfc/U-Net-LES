import tempfile
import unittest
from pathlib import Path
import numpy as np
from les_init import initialize_snapshot


class NativeLESInitializationTests(unittest.TestCase):
    def test_filtered_snapshot_and_vorticity(self):
        fine, coarse = 16, 8
        x = 2*np.pi*np.arange(fine)/fine
        phi = np.cos(x[:, None]) + np.zeros((fine, fine)) + 4
        density = np.sin(x[None, :]) + np.zeros((fine, fine)) + 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dns.npz'
            np.savez(path, U=density, V=phi, P=np.full_like(phi, 999))
            for order in (2, 4):
                fields = initialize_snapshot(path, coarse, 2*np.pi/coarse,
                                             2*np.pi/coarse, 0.25, order)
                xc = 2*np.pi*np.arange(coarse)/coarse
                expected_density = np.broadcast_to(np.sin(xc[None, :]), (coarse, coarse))
                expected_phi = np.broadcast_to(np.cos(xc[:, None]), (coarse, coarse))
                expected = np.stack((expected_density, expected_phi)) / (1 + 0.25**2)
                np.testing.assert_allclose(fields[:2], expected, atol=1e-14)
                h = 2*np.pi/fine
                eigenvalue = ((2*np.cos(h)-2)/h**2 if order == 2 else
                              (-2*np.cos(2*h)+32*np.cos(h)-30)/(12*h**2))
                np.testing.assert_allclose(fields[2], eigenvalue*fields[1], atol=1e-13)
                self.assertEqual(fields.dtype, np.float64)
                self.assertTrue(fields.flags.c_contiguous)

    def test_invalid_parameters(self):
        for kwargs in ({'alpha': -1}, {'laplacian_order': 6}, {'dx': 0}):
            arguments = dict(path='unused', coarse_n=8, dx=1., dz=1.)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                initialize_snapshot(**arguments)
