import unittest
from unittest.mock import patch

import torch

from germano_common import VariationalResidualUNetSR, checkpoint_model_class
from train_germano_unet import calculate_training_losses


class VariationalResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = VariationalResidualUNetSR(base_channels=2, scale_factor=2)
        self.x = torch.randn(2, 3, 8, 8)
        self.target = torch.randn(2, 3, 16, 16)

    def test_prior_samples_are_centered_on_deterministic_correction(self):
        self.model.eval()
        mean = self.model(self.x)
        self.assertTrue(torch.equal(mean, self.model(self.x)))
        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            positive = self.model(self.x, sample=True)
        with patch('torch.randn_like', side_effect=lambda x: -torch.ones_like(x)):
            negative = self.model(self.x, sample=True)
        torch.testing.assert_close((positive + negative) / 2, mean)
        self.assertFalse(torch.equal(positive, mean))

    def test_posterior_and_prior_receive_dns_loss_gradients(self):
        zeros = torch.zeros(1, 3, 1, 1)
        ones = torch.ones_like(zeros)
        total, values, pf, pd = calculate_training_losses \
        (
            self.model, self.x, self.target, self.target,
            zeros, ones, ones, 1., 0.1, 10., 10., 1., 0., 0., 0.2, True,
            None, 8., 'cutoff', None, None, 8., 'none', lambda_kl=0.01,
        )
        self.assertTrue(pd.requires_grad)
        self.assertGreaterEqual(values['kl'].item(), 0)
        total.backward()
        for name in ('prior', 'posterior_target', 'posterior', 'fluctuation', 'out', 'e1'):
            gradients = \
            [
                p.grad for p in getattr(self.model, name).parameters()
                if p.grad is not None
            ]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)

    def test_checkpoint_and_nonvariational_mode(self):
        for variational in (True, False):
            model = VariationalResidualUNetSR \
            (
                base_channels=2, scale_factor=2, variational=variational,
            ).eval()
            checkpoint = \
            {
                'architecture': type(model).__name__,
                'model': model.state_dict(),
            }
            restored = checkpoint_model_class(checkpoint) \
            (
                base_channels=2, scale_factor=2, variational=variational,
            ).eval()
            restored.load_state_dict(checkpoint['model'], strict=True)
            torch.testing.assert_close(model(self.x), restored(self.x))
            if not variational:
                torch.testing.assert_close(model(self.x), model(self.x, sample=True))

    def test_training_preserves_inverse_mean_removal(self):
        model = VariationalResidualUNetSR \
        (
            base_channels=2, scale_factor=2, output_channels=3,
            variational=False,
        ).eval()
        with torch.no_grad():
            model.out.weight.zero_()
            model.out.bias.copy_(torch.tensor([1., 2., 3.]))
        zeros = torch.zeros(1, 3, 1, 1)
        ones = torch.ones_like(zeros)
        _, _, _, pd = calculate_training_losses \
        (
            model, torch.zeros_like(self.x), self.target, self.target,
            zeros, ones, ones, 1., 0., 10., 10., 1., 0., 0., 0., False,
            None, 8., 'cutoff', None, None, 8., 'none',
        )
        torch.testing.assert_close \
        (
            pd.mean(dim=(-2, -1)), torch.zeros(2, 3),
        )

    def test_evaluation_never_uses_target_encoder(self):
        self.model.eval()
        zeros = torch.zeros(1, 3, 1, 1)
        ones = torch.ones_like(zeros)
        with patch.object(self.model.posterior, 'forward', side_effect=AssertionError):
            calculate_training_losses \
            (
                self.model, self.x, self.target, self.target,
                zeros, ones, ones, 1., 0.1, 10., 10., 1., 0., 0., 0., True,
                None, 8., 'cutoff', None, None, 8., 'none',
            )


if __name__ == '__main__':
    unittest.main()
