import unittest

import torch

from germano_common import BicubicUNetSR, checkpoint_model_class, periodic_interpolate
import test_variational_unet


class BicubicUNetTests(test_variational_unet.VariationalUNetTests):
    def setUp(self):
        super().setUp()
        self.model = BicubicUNetSR(base_channels=2, scale_factor=2)

    def test_bicubic_input_without_global_skip(self):
        self.model.global_input_skip = False
        self.model.eval()
        inputs = []
        handle = self.model.e1.register_forward_pre_hook \
        (
            lambda module, args: inputs.append(args[0].detach())
        )
        try:
            with torch.no_grad():
                self.model.out.weight.zero_()
                self.model.out.bias.zero_()
                output = self.model(self.x)
        finally:
            handle.remove()
        torch.testing.assert_close \
        (
            inputs[0], periodic_interpolate(self.x, 2, "bicubic"),
        )
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_checkpoint_roundtrip(self):
        cls = checkpoint_model_class({"architecture": "BicubicUNetSR"})
        self.assertIs(cls, BicubicUNetSR)
        for variational in (False, True):
            model = cls(base_channels=2, scale_factor=2, variational=variational).eval()
            restored = cls(base_channels=2, scale_factor=2, variational=variational).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(model(self.x), restored(self.x))


if __name__ == "__main__":
    unittest.main()
