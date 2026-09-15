import unittest
import math
import torch
from germano_common import inverse_input_filter_torch


class SpectrumThresholdTests(unittest.TestCase):
    def test_threshold_dispatch(self):
        n = 16
        x = torch.arange(n, dtype=torch.float64) * (2 * math.pi / n)
        field = (torch.cos(x) + torch.cos(3 * x))[None, None, None, :].expand(1, 1, n, n)
        target = torch.zeros(1, 32, dtype=field.dtype)
        def filtered(**kwargs):
            return inverse_input_filter_torch \
            (
                field, 2 * math.pi, 2 * math.pi,
                'filtered_dns_spectrum', target_shell_power=target, **kwargs,
            )
        low = torch.cos(x)[None, None, None, :].expand_as(field)
        torch.testing.assert_close(filtered(), low)
        torch.testing.assert_close(filtered(min_wavenumber=2), low)
        torch.testing.assert_close(filtered(min_wavenumber=4), field)
        torch.testing.assert_close(filtered(min_wavenumber=0), torch.zeros_like(field))
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                filtered(min_wavenumber=invalid)

    def test_analytic_cutoff_unchanged(self):
        field = torch.randn(1, 1, 16, 16)
        a = inverse_input_filter_torch(field, 10, 10, 'cutoff', 0.5)
        b = inverse_input_filter_torch(field, 10, 10, 'cutoff', 0.5, min_wavenumber=4)
        torch.testing.assert_close(a, b)
