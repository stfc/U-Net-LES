import unittest
import torch
from germano_common import PeriodicLowPass2d, ResidualUNetSR, UpBlock
from train_germano_unet import calculate_training_losses


class VariationalUNetTests(unittest.TestCase):
    def test_low_pass_preserves_constants_and_removes_checkerboard(self):
        blur = PeriodicLowPass2d().double()
        constant = torch.ones(1, 2, 6, 10, dtype=torch.float64)
        constant[:, 1] *= 3
        torch.testing.assert_close(blur(constant), constant)
        checkerboard = constant.clone()
        checkerboard[..., ::2, :] *= -1
        checkerboard[..., :, ::2] *= -1
        torch.testing.assert_close \
        (
            blur(checkerboard), torch.zeros_like(checkerboard),
        )
        self.assertEqual(list(blur.parameters()), [])

    def test_transposed_upsampling_is_periodic(self):
        block = UpBlock(4, 2, 2).double()
        x = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
        skip = torch.randn(1, 2, 6, 10, dtype=torch.float64)
        y = block(x, skip)
        self.assertEqual(y.shape, (1, 2, 6, 10))
        shifted = block \
        (
            torch.roll(x, (1, -1), (-2, -1)),
            torch.roll(skip, (2, -2), (-2, -1)),
        )
        torch.testing.assert_close \
        (
            shifted, torch.roll(y, (2, -2), (-2, -1)),
        )
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(block.up.weight.grad).all())
        self.assertGreater(block.up.weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "twice"):
            block(x, skip[..., :-1, :])

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 8, 8)
        self.model = ResidualUNetSR(base_channels=2, scale_factor=2)

    def test_sampling_and_gradients(self):
        y, kl = self.model(self.x, return_kl=True)
        self.assertEqual(y.shape, (2, 2, 16, 16))
        self.assertGreaterEqual(kl.item(), 0)
        self.assertFalse(torch.equal(y, self.model(self.x)))
        (y.square().mean() + kl).backward()
        for head in (self.model.latent_mu, self.model.latent_logvar):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.model.eval()
        self.assertTrue(torch.equal(self.model(self.x), self.model(self.x)))
        self.assertFalse(torch.equal(self.model(self.x, sample=True), self.model(self.x)))

    def test_legacy_and_variational_roundtrip(self):
        for variational in (False, True):
            model = ResidualUNetSR(base_channels=2, scale_factor=2, variational=variational).eval()
            restored = ResidualUNetSR(base_channels=2, scale_factor=2, variational=variational).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertTrue(torch.equal(model(self.x), restored(self.x)))
            if not variational:
                self.assertEqual(model(self.x, return_kl=True)[1].item(), 0)
                self.assertFalse(any('latent_' in k for k in model.state_dict()))

    def test_training_objective(self):
        self.model.eval()
        zeros = torch.zeros(1, 3, 1, 1)
        ones = torch.ones_like(zeros)
        total, vals, _, _ = calculate_training_losses \
        (
            self.model, self.x, torch.zeros(2, 3, 16, 16),
            torch.zeros(2, 3, 16, 16), zeros, ones, ones, 1.,
            0.1, 10., 10., 1., 0., 0., 0.2, True,
            None, 8., 'cutoff', None, None, 8., 'none', lambda_kl=0.3,
        )
        torch.testing.assert_close \
        (
            total,
            vals['dns'] + 0.1 * vals['dns_mse']
            + 0.2 * (vals['dns_P'] + 0.1 * vals['dns_P_mse'])
            + 0.3 * vals['kl'],
        )
        total.backward()
        self.assertTrue(torch.isfinite(self.model.latent_logvar.weight.grad).all())

    def test_loss_and_gradients_depend_on_dns_and_inversion(self):
        self.model.eval()
        mean = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1)
        std = torch.tensor([2., 3., 4.]).reshape(1, 3, 1, 1)
        target = torch.randn(2, 3, 16, 16)
        for derived in (False, True):
            model = ResidualUNetSR \
            (
                base_channels=2, scale_factor=2,
                output_channels=2 if derived else 3,
            ).eval()
            results = []
            for alpha, dns in \
            (
                (0.1, torch.zeros_like(target)),
                (0.9, torch.ones_like(target) * 100)
            ):
                model.zero_grad(set_to_none=True)
                total, vals, pf, pd = calculate_training_losses \
                (
                    model, self.x, target, dns, mean, std, std, 4.,
                    alpha, 10., 10., 1., 0., 0., 0.2, derived,
                    None, 8., 'cutoff', None, None, 8., 'none',
                    lambda_kl=0.3,
                )
                expected = \
                (
                    (pd[:, :2] - dns[:, :2]) / std[:, :2]
                ).abs().mean()
                torch.testing.assert_close(vals['dns'], expected)
                torch.testing.assert_close \
                (
                    vals['dns_mse'],
                    ((pd[:, :2] - dns[:, :2]) / std[:, :2]).square().mean(),
                )
                torch.testing.assert_close \
                (
                    vals['dns_P_mse'],
                    ((pd[:, 2] - dns[:, 2]) / 4.).square().mean(),
                )
                self.assertTrue(pd.requires_grad)
                total.backward()
                gradients = torch.cat \
                (
                     \
                    [
                        p.grad.flatten() for p in model.parameters()
                        if p.grad is not None
                    ]
                )
                results.append((total.detach(), gradients, pd))
            self.assertFalse(torch.equal(results[0][0], results[1][0]))
            self.assertFalse(torch.equal(results[0][1], results[1][1]))
            self.assertFalse(torch.equal(results[0][2], results[1][2]))


if __name__ == '__main__':
    unittest.main()
