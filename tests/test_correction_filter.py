import unittest

import torch

from germano_common import \
(
    BicubicUNetSR, periodic_interpolate, restore_model_correction_settings,
    smooth_correction_filter,
)


class CorrectionFilterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_constants_modes_and_gradients(self):
        x = torch.arange(32, dtype=torch.float64)
        wave = torch.cos(2 * torch.pi * 4 * x / 32)[None, None, :, None].expand(1, 1, 32, 32)
        field = (2 + wave).clone().requires_grad_()
        output = smooth_correction_filter(field, 4, 1.0)
        torch.testing.assert_close(output, 2 + wave * torch.exp(torch.tensor(-1.0, dtype=field.dtype)))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(field.grad).all())
        self.assertGreater(field.grad.abs().sum().item(), 0)
        self.assertIs(smooth_correction_filter(field, 4, None), field)

    def test_skip_untouched_and_settings_roundtrip(self):
        model = BicubicUNetSR(base_channels=2, scale_factor=2, variational=False).double().eval()
        settings = dict(correction_cutoff=0.7, correction_filter_order=6.0, global_input_skip=True)
        restore_model_correction_settings(model, settings)
        captured = []
        handle = model.out.register_forward_hook \
        (
            lambda module, args, result: captured.append(result)
        )
        x = torch.randn(1, 3, 8, 8, dtype=torch.float64)
        try:
            y = model(x)
        finally:
            handle.remove()
        expected = periodic_interpolate(x, 2, 'bicubic')[:, :2]
        expected = expected + smooth_correction_filter(captured[0], 2, 0.7, 6.0)
        torch.testing.assert_close(y, expected)
        restored = BicubicUNetSR(base_channels=2, scale_factor=2, variational=False).double().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        restore_model_correction_settings(restored, settings)
        torch.testing.assert_close(restored(x), y)
        restore_model_correction_settings(restored, {})
        self.assertIsNone(restored.correction_cutoff)
