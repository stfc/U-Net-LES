import unittest

import torch

class RescalingOptionTests(unittest.TestCase):
    def wrapper(self, mode):
        import ast
        from pathlib import Path
        from germano_common import rescaling_amplitude_correction13

        source = Path(__file__).resolve().parents[1] / 'pBOUT-unet.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_apply_rescaling')
        namespace = \
        {
            'torch': torch,
            'RESCALING': mode,
            '_RESCALING_REFERENCE_MAXIMA': (torch.full((1, 2, 1, 1), 6.), torch.full((1, 2, 1, 1), 2.)),
            'rescaling_amplitude_correction13': rescaling_amplitude_correction13,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['_apply_rescaling']

    def test_disabled_returns_input(self):
        dns = torch.randn(1, 3, 2, 2)
        self.assertIs(self.wrapper(0)(dns, None), dns)

    def test_selects_formulas_and_preserves_vorticity(self):
        dns = torch.ones(1, 3, 2, 2)
        les = torch.tensor([1., 1., 1., 5.]).view(1, 1, 2, 2).expand(1, 3, 2, 2)
        for mode, expected in ((2, 15.), (3, 6.)):
            result = self.wrapper(mode)(dns, les)
            torch.testing.assert_close(result[:, :2], torch.full_like(dns[:, :2], expected))
            torch.testing.assert_close(result[:, 2:], dns[:, 2:])

    def test_formula13_matches_recurrence(self):
        from germano_common import rescaling_amplitude_correction13

        dns = torch.ones(2, 2, 4, 4)
        initial_dns = torch.tensor([6., 8.]).view(1, 2, 1, 1)
        initial_les = torch.tensor([2., 4.]).view(1, 2, 1, 1)
        previous_dns, previous_les = initial_dns, initial_les
        for amplitude in (4., 1., 7., 7., 3.):
            les = torch.full((2, 2, 2, 2), amplitude)
            anchored = rescaling_amplitude_correction13(dns, les, initial_dns, initial_les)
            recurrent = rescaling_amplitude_correction13(dns, les, previous_dns, previous_les)
            torch.testing.assert_close(anchored, recurrent)
            previous_dns = recurrent
            previous_les = les.abs().amax(dim=(-2, -1), keepdim=True)

    def test_invalid_modes_are_rejected(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / 'pBOUT-unet.py').read_text()
        block = source[source.index('_rescaling_value ='):source.index('_RESCALING_REFERENCE_MAXIMA =')]
        for mode in (0, 2, 3):
            namespace = {'_CONFIG': {'rescaling': str(mode)}}
            exec(block, namespace)
            self.assertEqual(namespace['RESCALING'], mode)
        dns = torch.ones(1, 3, 2, 2)
        for mode in (1, -1, 4):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'rescaling must be'):
                    exec(block, {'_CONFIG': {'rescaling': str(mode)}})
                with self.assertRaisesRegex(ValueError, 'rescaling must be'):
                    self.wrapper(mode)(dns, dns)

    def test_initial_maxima_preserved_across_calls(self):
        rescale = self.wrapper(3)
        dns = torch.tensor([-2., 1., 0., 1., -4., 2., 0., 2., 1., 2., 3., 4.])
        dns = dns.reshape(1, 3, 2, 2)
        for amplitude in (0.5, 3., 1.):
            current = dns * amplitude
            result = rescale(current, None)
            torch.testing.assert_close(result[:, :2].abs().amax(dim=(-2, -1)), torch.full((1, 2), 6.))
            torch.testing.assert_close(result[:, 2:], current[:, 2:])
            torch.testing.assert_close(rescale(result, None), result)

    def test_initial_maxima_zero_reconstruction(self):
        dns = torch.zeros(1, 3, 2, 2)
        torch.testing.assert_close(self.wrapper(3)(dns, None), dns)

    def test_previous_time_reference_and_rollback(self):
        import ast
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / 'pBOUT-unet.py'
        tree = ast.parse(source.read_text())
        functions = \
        [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ('_record_rescaling_maxima', '_prepare_rescaling_time')
        ]
        initial = (torch.ones(1, 2, 1, 1), torch.ones(1, 2, 1, 1))
        namespace = \
        {
            'torch': torch,
            'RESCALING': 2,
            '_RESCALING_HISTORY': {float('-inf'): initial},
            '_RESCALING_REFERENCE_MAXIMA': initial,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        record = namespace['_record_rescaling_maxima']
        prepare = namespace['_prepare_rescaling_time']
        field = torch.ones(1, 2, 2, 2)
        record(0., field * 6, field * 2)
        prepare(1.)
        record(1., field * 8, field * 4)
        prepare(1.)
        self.assertEqual(namespace['_RESCALING_REFERENCE_MAXIMA'][0][0, 0].item(), 6.)
        record(1., field * 10, field * 4)
        prepare(2.)
        self.assertEqual(namespace['_RESCALING_REFERENCE_MAXIMA'][0][0, 0].item(), 10.)
        record(2., field * 12, field * 5)
        prepare(0.5)
        self.assertEqual(namespace['_RESCALING_REFERENCE_MAXIMA'][0][0, 0].item(), 6.)
        self.assertEqual(set(namespace['_RESCALING_HISTORY']), {float('-inf'), 0.})

    def test_formula13_zero_fields(self):
        from germano_common import rescaling_amplitude_correction13

        zeros = torch.zeros(1, 1, 2, 2)
        ones = torch.ones_like(zeros)
        maximum = torch.ones(1, 1, 1, 1)
        self.assertEqual(rescaling_amplitude_correction13(ones, zeros, maximum, maximum).item(), 0.)
        self.assertEqual(rescaling_amplitude_correction13(zeros, ones, maximum, maximum).item(), 1.)
        self.assertEqual(rescaling_amplitude_correction13(ones, ones, maximum, maximum * 0).item(), 1.)
