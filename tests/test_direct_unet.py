import unittest

import torch

from germano_common import DirectUNetSR, checkpoint_model_class


class TestDirectUNet(unittest.TestCase):
    def test_default_resolution_and_channels(self):
        model = DirectUNetSR(base_channels=2, scale_factor=4).eval()
        coarse = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            prediction, kl = model(coarse, return_kl=True)
        self.assertEqual(model.stem(coarse).shape, (1, 8, 8, 8))
        self.assertEqual(prediction.shape, (1, 2, 32, 32))
        self.assertEqual(kl.item(), 0.0)

    def test_checkpoint_dispatch(self):
        self.assertIs(
            checkpoint_model_class({"architecture": "DirectUNetSR"}),
            DirectUNetSR,
        )

    def test_scale_must_be_power_of_two(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            DirectUNetSR(scale_factor=3)


if __name__ == "__main__":
    unittest.main()
