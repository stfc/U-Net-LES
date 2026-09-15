import unittest

import torch

from germano_common import rescaling_amplitude_correction


class RescalingTests(unittest.TestCase):
    def test_independent_amplitudes_and_zero_crossings(self):
        filtered = torch.tensor([0., -2., 1., 3.], dtype=torch.float64)
        filtered = filtered.reshape(1, 1, 2, 2).expand(2, 2, 2, 2)
        expected = torch.tensor([2., 3., 4., 5.], dtype=torch.float64)
        expected = expected.reshape(2, 2, 1, 1)
        torch.testing.assert_close \
        (
            rescaling_amplitude_correction(filtered * expected, filtered),
            expected,
        )

    def test_median_rejects_outlier(self):
        filtered = torch.ones(1, 1, 2, 2, dtype=torch.float64)
        les = torch.tensor([2., 3., 3., 100.], dtype=torch.float64).reshape_as(filtered)
        self.assertEqual(rescaling_amplitude_correction(les, filtered).item(), 3.)

    def test_even_median_averages_middle_values(self):
        filtered = torch.ones(1, 1, 2, 2)
        les = torch.tensor([1., 2., 4., 100.]).reshape_as(filtered)
        self.assertEqual(rescaling_amplitude_correction(les, filtered).item(), 3.)

    def test_excludes_nonfinite_and_near_zero_denominators(self):
        filtered = torch.tensor([float('nan'), float('inf'), 1e-20, 2.])
        filtered = filtered.reshape(1, 1, 2, 2)
        les = torch.tensor([1., 1., 1., 6.]).reshape_as(filtered)
        self.assertEqual(rescaling_amplitude_correction(les, filtered).item(), 3.)

    def test_degenerate_fields(self):
        zeros = torch.zeros(1, 1, 2, 2)
        ones = torch.ones_like(zeros)
        self.assertEqual(rescaling_amplitude_correction(zeros, ones).item(), 0.)
        self.assertEqual(rescaling_amplitude_correction(ones, zeros).item(), 1.)
        self.assertEqual(rescaling_amplitude_correction(zeros, zeros).item(), 0.)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            rescaling_amplitude_correction(torch.ones(2, 2), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            rescaling_amplitude_correction(torch.ones(1, 2, 2, 2), torch.ones(1, 1, 2, 2))


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
            'rescaling_amplitude_correction': rescaling_amplitude_correction,
            'rescaling_amplitude_correction13': rescaling_amplitude_correction13,
            '_germano_filter_torch': lambda field: field,
            '_spectral_resize_torch': lambda field, size: field,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['_apply_rescaling']

    def test_disabled_returns_input(self):
        dns = torch.randn(1, 3, 2, 2)
        self.assertIs(self.wrapper(0)(dns, None), dns)

    def test_selects_formulas_and_preserves_vorticity(self):
        dns = torch.ones(1, 3, 2, 2)
        les = torch.tensor([1., 1., 1., 5.]).view(1, 1, 2, 2).expand(1, 3, 2, 2)
        for mode, expected in ((1, 1.), (2, 15.)):
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

    def test_formula11_filters_normalized_dns_and_restores_amplitude(self):
        rescale = self.wrapper(1)
        seen = []

        def filter_field(field):
            seen.append(field.clone())
            return field * 0.5

        rescale.__wrapped__.__globals__['_germano_filter_torch'] = filter_field
        dns = torch.tensor([-2., 1., 0., 2.]).reshape(1, 1, 2, 2).expand(1, 3, 2, 2)
        result = rescale(dns, dns * 3.)
        torch.testing.assert_close(seen[0].abs().amax(dim=(-2, -1)), torch.ones(1, 2))
        torch.testing.assert_close(result[:, :2], dns[:, :2] * 6.)
        torch.testing.assert_close(result[:, 2:], dns[:, 2:])

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
