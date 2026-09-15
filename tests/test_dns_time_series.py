import ast
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from germano_common import germano_filter, spectral_resize
from germano_common import periodic_laplacian_2nd_torch, periodic_laplacian_4th_torch
from dns_time_series import DNSTimeSeries
from visualization_common import arakawa_bracket


class DNSReplayTests(unittest.TestCase):
    def test_configuration_switches_are_independent(self):
        source = Path('pBOUT-unet.py').read_text()
        block = source[source.index('_dns_init_value ='):source.index('DNS_FIELD_DIR =')]
        for mode in range(4):
            for implicit in (0, 1):
                with self.subTest(mode=mode, implicit=implicit), patch.dict \
                (
                    os.environ, \
                    {
                        'PBOUT_UNET_DNS_INIT': str(mode),
                        'PBOUT_UNET_IMPLICIT': str(implicit)
                    }
                ):
                    namespace = dict(os=os, Path=Path)
                    exec(block, namespace)
                    self.assertEqual(namespace['DNS_REPLAY'], mode == 3)
                    self.assertEqual(namespace['IMPLICIT'], bool(implicit))
        for key, value in \
        [
            ('PBOUT_UNET_DNS_INIT', '4'),
            ('PBOUT_UNET_IMPLICIT', '3')
        ]:
            with patch.dict \
            (
                os.environ, \
                {
                    'PBOUT_UNET_DNS_INIT': '0',
                    'PBOUT_UNET_IMPLICIT': '1', key: value
                }
            ):
                with self.assertRaisesRegex(ValueError, key):
                    exec(block, dict(os=os, Path=Path))

    def test_interpolation_and_time_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, time, value in [('z', 2., 4.), ('a', 5., 10.)]:
                np.savez \
                (
                    Path(directory) / f'{name}.npz', simtime=time,
                    **{key: np.full((4, 4), value) for key in ('U', 'V', 'P')}
                )
            series = DNSTimeSeries(directory, 4)
            np.testing.assert_allclose(series.interpolate(3.), 6.)
            np.testing.assert_allclose(series.interpolate(5.), 10.)
            np.testing.assert_allclose(series.interpolate(2.), 4.)
            for time in (1., 6., float('nan')):
                with self.assertRaises(ValueError):
                    series.interpolate(time)

    def test_replay_initialization_matches_mode_zero(self):
        tree = ast.parse(Path('pBOUT-unet.py').read_text())
        init = next \
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'initFlow'
        )
        # Exercise initialization before plotting/reconstruction diagnostics.
        end = next(i for i, node in enumerate(init.body) if isinstance(node, ast.With))
        init.body = init.body[:end] + [ast.Return(value=ast.Name(id='fields', ctx=ast.Load()))]
        ast.fix_missing_locations(init)
        resize = next \
        (
            node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == '_spectral_resize_torch'
        )
        dns = np.random.default_rng(42).normal(size=(3, 8, 8))
        dns[:2] += 3.0
        dns[2] += 5.0
        vorticity = next \
        (
            node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == '_vorticity_from_potential'
        )
        centered = dns.copy()
        centered[:2] -= centered[:2].mean(axis=(-2, -1), keepdims=True)
        phi = centered[1]
        centered[2] = sum \
        (
            np.roll(phi, 1, axis=axis) - 2 * phi + np.roll(phi, -1, axis=axis)
            for axis in (0, 1)
        )
        with tempfile.TemporaryDirectory() as directory:
            class LocalPath:
                def __new__(cls, name):
                    return Path(directory) / name
            namespace = dict \
            (
                np=np, torch=torch, Path=LocalPath,
                DNS_PATH=Path(__file__), NUMPY_DTYPE=np.float64,
                DEVICE='cpu', TORCH_DTYPE=torch.float64,
                ALPHA=1., LX=8., LY=8., COARSE_N=4,
                FIELD_NAMES=('U', 'V', 'P'), LAPLACIAN_ORDER=2,
                periodic_laplacian_2nd_torch=periodic_laplacian_2nd_torch,
                periodic_laplacian_4th_torch=periodic_laplacian_4th_torch,
                _load_dns_uvp=lambda path: dns.copy(),
                germano_filter=germano_filter
            )
            exec \
            (
                compile \
                (
                    ast.Module(body=[resize, vorticity, init], type_ignores=[]),
                    '<init>', 'exec'
                ), namespace
            )
            initial = []
            for mode in (0, 1, 2, 3):
                namespace.update(DNS_INIT=mode, DNS_REPLAY=mode == 3, RECONSTRUCT_CHANNELS=())
                initial.append(namespace['initFlow']([]))
                reference = namespace['_INITIAL_DNS_REFERENCE']
                self.assertAlmostEqual(reference[2].mean(), 0.0, places=14)
                np.testing.assert_allclose(reference, centered, atol=1e-14)
        dns = centered
        expected = np.stack \
        (
             \
            [
                spectral_resize(germano_filter(plane, 1., 8., 8.), 4)
                for plane in dns
            ]
        )
        for fields in (initial[0], initial[3]):
            self.assertEqual(fields.shape, (3, 4, 4))
            np.testing.assert_allclose(fields, expected, atol=1e-14)
        np.testing.assert_allclose(initial[2], dns, atol=1e-14)
        np.testing.assert_allclose(initial[1][:2], np.stack([germano_filter(plane, 1., 8., 8.) for plane in dns[:2]]), atol=1e-14)
        for fields in initial:
            np.testing.assert_allclose(fields.mean(axis=(-2, -1)), 0.0, atol=1e-14)

    def test_replay_callback_uses_dns_brackets(self):
        self.check_replay_callback(False)

    def test_replay_callback_uses_implicit_correction(self):
        self.check_replay_callback(True)

    def check_replay_callback(self, implicit):
        # Load only the callback so this regression needs no trained checkpoint.
        tree = ast.parse(Path('pBOUT-unet.py').read_text())
        callback = next \
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'findLESTerms'
        )
        n = 8
        dns = torch.randn(2, 3, n, n, dtype=torch.float64)
        incoming = np.zeros((2, 5 if implicit else 3, n // 2, n // 2))
        if implicit:
            incoming[:, 3:] = 2.5
        requested = []
        def replay(time, batch):
            requested.append((time, batch))
            return dns
        namespace = dict \
        (
            np=np, torch=torch, DNS_INIT=3, DNS_REPLAY=True,
            RECONSTRUCT_CHANNELS=(), FINE_N=n, COARSE_N=n // 2,
            IMPLICIT=implicit, DEVICE='cpu', TORCH_DTYPE=torch.float64,
            LX=float(n), LY=float(n), ALPHA=1., WRITE_DIAGNOSTICS=0,
            _validate_les_spacing=lambda dx, dy, size: self.assertEqual(size, n // 2),
            _decode_fields=lambda *args: (incoming, None),
            _dns_at_time=replay, _arakawa_bracket=arakawa_bracket,
        )
        helpers = \
        [
            node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in ('_spectral_resize_torch', '_germano_filter_torch')
        ]
        exec(compile(ast.Module(body=helpers, type_ignores=[]), '<helpers>', 'exec'), namespace)
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<callback>', 'exec'), namespace)
        actual = namespace['findLESTerms']([7, 0, 2, 3.5])
        expected = [0.]
        for channel in (2, 0):
            bracket = arakawa_bracket(dns[:, 1:2], dns[:, channel:channel+1], 1., 1.)
            coarse = np.stack \
            (
                 \
                [
                    spectral_resize(germano_filter(plane, 1., n, n), n // 2)
                    for plane in bracket[:, 0].numpy()
                ]
            )
            if implicit:
                coarse = 2.5 - coarse
            expected.extend(coarse.transpose(1, 0, 2).reshape(-1))
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(requested, [(3.5, 2)])


if __name__ == '__main__':
    unittest.main()
