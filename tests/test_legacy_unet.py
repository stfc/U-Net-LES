import unittest

import torch

from germano_common import \
(
    LegacyResidualUNetSR, ResidualUNetSR, checkpoint_model_class,
    periodic_interpolate, restore_checkpoint_model_settings,
)


class LegacyUNetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dispatch_and_strict_roundtrip(self):
        for variational in (False, True):
            model = LegacyResidualUNetSR \
            (
                base_channels=2, scale_factor=2, variational=variational,
            ).eval()
            for metadata in ({}, {"architecture": "ResidualUNetSR"}, {"architecture": "LegacyResidualUNetSR"}):
                checkpoint = dict(metadata, model=model.state_dict())
                cls = checkpoint_model_class(checkpoint)
                self.assertIs(cls, LegacyResidualUNetSR)
                restored = cls \
                (
                    base_channels=2, scale_factor=2, variational=variational,
                ).eval()
                restored.load_state_dict(checkpoint['model'], strict=True)
                x = torch.randn(1, 3, 8, 8)
                torch.testing.assert_close(model(x), restored(x))
        current = ResidualUNetSR(base_channels=2)
        self.assertIs(checkpoint_model_class({'model': current.state_dict()}), ResidualUNetSR)

    def test_checkpoint_skip_semantics(self):
        model = LegacyResidualUNetSR(base_channels=2)
        self.assertFalse(model.global_input_skip)
        restore_checkpoint_model_settings(model, {})
        self.assertTrue(model.global_input_skip)
        restore_checkpoint_model_settings(model, {"architecture": "LegacyResidualUNetSR"})
        self.assertFalse(model.global_input_skip)
        restore_checkpoint_model_settings(model, {"global_input_skip": True})
        self.assertTrue(model.global_input_skip)
        restore_checkpoint_model_settings(model, {"global_input_skip": False})
        self.assertFalse(model.global_input_skip)

    def test_archived_architecture_and_global_skip(self):
        model = LegacyResidualUNetSR().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1762394)
        self.assertFalse(any('latent_' in k or '.up.' in k for k in model.state_dict()))
        with torch.no_grad():
            model.out.weight.zero_()
            model.out.bias.zero_()
            x = torch.randn(1, 3, 4, 4)
            y, kl = model(x, return_kl=True)
        torch.testing.assert_close(y, torch.zeros_like(y))
        model.global_input_skip = True
        with torch.no_grad():
            restored = model(x)
        torch.testing.assert_close(restored, periodic_interpolate(x, 8, 'bicubic')[:, :2])
        self.assertEqual(kl.item(), 0)


if __name__ == '__main__':
    unittest.main()
