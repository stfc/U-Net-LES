import unittest
import torch
from germano_common import (
    DirectUNetSR, FilteredDirectUNetSR, PeriodicBinomialLowPass2d,
    checkpoint_model_class,
)


class TestFilteredDecoder(unittest.TestCase):
    def test_dc_and_alternating_modes(self):
        blur = PeriodicBinomialLowPass2d()
        ones = torch.ones(1, 1, 8, 8, dtype=torch.float64)
        torch.testing.assert_close(blur(ones), ones)
        alternating = (-1.0) ** torch.arange(8, dtype=torch.float64)
        for pattern in (alternating[:, None], alternating[None, :],
                        alternating[:, None] * alternating[None, :]):
            torch.testing.assert_close(blur(ones * pattern), torch.zeros_like(ones))

    def test_periodicity_and_gradient(self):
        blur = PeriodicBinomialLowPass2d()
        x = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(blur(x.roll(3, -1)), blur(x).roll(3, -1))
        self.assertTrue(torch.autograd.gradcheck(blur, (x,)))

    def test_model_roundtrip_and_output_nyquist(self):
        model = FilteredDirectUNetSR(base_channels=2, scale_factor=2).double()
        x = torch.randn(1, 3, 8, 8, dtype=torch.float64)
        y, kl = model(x, return_kl=True)
        self.assertEqual(y.shape, (1, 2, 16, 16))
        spectrum = torch.fft.fft2(y)
        self.assertLess(spectrum[..., 8, :].abs().max().item(), 1e-12)
        self.assertLess(spectrum[..., :, 8].abs().max().item(), 1e-12)
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                            for p in model.parameters()))
        cls = checkpoint_model_class({'architecture': 'FilteredDirectUNetSR'})
        restored = cls(base_channels=2, scale_factor=2).double()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(x), y)
        self.assertIs(checkpoint_model_class({'architecture': 'DirectUNetSR'}),
                      DirectUNetSR)


if __name__ == '__main__':
    unittest.main()
