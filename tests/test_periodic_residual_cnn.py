import unittest

import torch

from germano_common import \
(
    PeriodicSubpixelCNN,
    ResidualUNetSR,
    checkpoint_model_class,
)
import test_variational_unet


class PeriodicResidualCNNTests(test_variational_unet.VariationalUNetTests):
    """Also run the existing sampling, gradient, and training-loss checks."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 8, 8)
        self.model = PeriodicSubpixelCNN(base_channels=2, scale_factor=2)

    def test_shift_equivariance_and_constant_resolution(self):
        for variational in (False, True):
            for scale in (1, 2, 4):
                model = PeriodicSubpixelCNN \
                (
                    base_channels=2, scale_factor=scale,
                    variational=variational,
                ).double().eval()
                x = torch.randn(1, 3, 8, 10, dtype=torch.float64)
                shapes = []
                handles = \
                [
                    block.register_forward_hook \
                    (
                        lambda module, inputs, output: shapes.append(output.shape[-2:])
                    )
                    for block in model.blocks
                ]
                with torch.no_grad():
                    y = model(x)
                    self.assertEqual(y.shape, (1, 2, 8 * scale, 10 * scale))
                    for shift in ((1, 0), (0, 1), (1, -1)):
                        shifted = model(torch.roll(x, shift, (-2, -1)))
                        torch.testing.assert_close \
                        (
                            shifted,
                            torch.roll \
                            (
                                y, tuple(s * scale for s in shift), (-2, -1)
                            ),
                            rtol=1e-10, atol=1e-10,
                        )
                for handle in handles:
                    handle.remove()
                self.assertTrue(all(s == (8, 10) for s in shapes))
                self.assertFalse \
                (
                    any \
                    (
                        isinstance \
                        (
                            m, (torch.nn.AvgPool2d, torch.nn.MaxPool2d, torch.nn.ConvTranspose2d)
                        )
                        for m in model.modules()
                    )
                )

    def test_no_interpolation_or_global_input_skip(self):
        from unittest.mock import patch

        self.model.eval()
        with torch.no_grad():
            self.model.out.weight.zero_()
            self.model.out.bias.zero_()
        with patch("germano_common.periodic_interpolate", side_effect=AssertionError):
            output = self.model(self.x)
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_legacy_and_variational_roundtrip(self):
        for architecture in ("ResidualUNetSR", "PeriodicResidualCNN", "PeriodicSubpixelCNN"):
            for variational in (False, True):
                checkpoint = {"architecture": architecture}
                cls = checkpoint_model_class(checkpoint)
                model = cls(base_channels=2, scale_factor=2, variational=variational).eval()
                restored = cls(base_channels=2, scale_factor=2, variational=variational).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(model(self.x), restored(self.x))
        self.assertIs(checkpoint_model_class({}), ResidualUNetSR)
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint architecture"):
            checkpoint_model_class({"architecture": "unknown"})


if __name__ == "__main__":
    unittest.main()
