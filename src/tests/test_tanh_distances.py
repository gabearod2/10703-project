"""Numerical checks for squared Wasserstein penalties in environment units."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep
import train
from algos.networks import GaussianActorCritic
from algos.ppo import PPO, adapt_penalty_coef, tanh_distance_for_update


class TestTanhDistances(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        torch.set_num_threads(1)
        self.policy = GaussianActorCritic(3, 4, np.zeros(5), np.ones(5), (8,))
        self.mean = torch.tensor([[0., -.4, .3, .9], [3., 3., 4., 4.]])
        self.std = torch.tensor([[.6, .9, 1.2, .4], [.5, .4, .7, .9]])
        self.new = torch.distributions.Normal(self.mean + .5, self.std * 1.1)

    def test_reference_integrals(self):
        # Dense integration against the standard normal density, independent of Hermite nodes.
        z = np.linspace(-10, 10, 20001)[:, None, None]
        def actions(mean, std, noise):
            return (1 + np.tanh(mean.numpy() + std.numpy() * noise)) / 2
        old = actions(self.mean, self.std, z)
        new = actions(self.new.mean, self.new.stddev, z)
        density = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        ref = ((new - old) ** 2 * density).sum(axis=(0, 2)) * (z[1, 0, 0] - z[0, 0, 0])
        action = tanh_distance_for_update("tanh-action", self.policy, 10)
        actual = action(self.mean, self.std, self.new)
        np.testing.assert_allclose(actual, ref, atol=2e-5)
        self.assertLess(actual[1], actual[0])  # Saturation attenuates the same mean shift.

        rng = np.random.default_rng(123)
        def movement(mean, std):
            a = actions(mean, std, rng.standard_normal((131072, 1, 4)))
            return np.sort(a[..., 1::2] - a[..., ::2], axis=0)
        ref = ((movement(self.mean, self.std) - movement(self.new.mean, self.new.stddev))
               ** 2).mean(axis=0).sum(-1)
        distance = tanh_distance_for_update("tanh-movement", self.policy, 10)
        np.testing.assert_allclose(distance(self.mean, self.std, self.new), ref, atol=.001)
        # Reflect and swap each opposing pair: T(-z) = 1-T(z) preserves the net-force law.
        swap = [1, 0, 3, 2]
        equivalent = torch.distributions.Normal(-self.mean[:, swap], self.std[:, swap])
        self.assertLess(distance(self.mean, self.std, equivalent).max().item(), .002)

    def test_layouts_gradients_and_reproducibility(self):
        for mode in ("tanh-action", "tanh-movement"):
            distance = tanh_distance_for_update(mode, self.policy, 9)
            old = torch.cat((self.mean[:, :1], self.mean), -1).requires_grad_()
            std = torch.ones_like(old, requires_grad=True)
            new = (old.detach() + .4).requires_grad_()
            scale = torch.full_like(std, .9, requires_grad=True)
            costs = distance(old, std, torch.distributions.Normal(new, scale))
            self.assertTrue(torch.equal(distance(old, std, torch.distributions.Normal(old, std)),
                                        torch.zeros_like(costs)))
            again = tanh_distance_for_update(mode, self.policy, 9)
            torch.testing.assert_close(costs, again(old[:, 1:], std[:, 1:],
                                                    torch.distributions.Normal(new[:, 1:], scale[:, 1:])))
            costs.sum().backward()
            self.assertIsNone(old.grad)
            self.assertIsNone(std.grad)
            for grad in (new.grad, scale.grad):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertEqual(grad[:, 0].abs().sum().item(), 0.)
                self.assertGreater(grad[:, 1:].abs().sum().item(), 0.)

    def test_sweep_and_rollout_adaptation(self):
        modes = ["tanh-action", "tanh-movement"]
        cfg = dict(grid={"penalty": modes}, fixed={}, seeds=[0])
        self.assertEqual([c["penalty"] for c in sweep.expand(cfg)], modes)
        with self.assertRaisesRegex(ValueError, "requires --action_transform tanh"):
            sweep.expand(dict(cfg, fixed={"action_transform": "clip"}))
        for mode in modes:
            args = train.get_args(["--penalty", mode, "--update_epochs", "1",
                                   "--num_minibatches", "2"], create_dir=False)
            ppo = PPO(self.policy, args)
            obs = torch.randn(514, 3)  # Unequal final measurement chunks.
            for update in range(2):
                with torch.no_grad():
                    old = self.policy.dist(obs)
                    old_std = old.stddev.clone()
                    act, logp, val = self.policy.act(obs)
                beta = ppo.beta
                result = ppo.update(dict(obs=obs, act=act, logp=logp, val=val,
                                         adv=torch.randn_like(val), ret=torch.randn_like(val)))
                fn = tanh_distance_for_update(mode, self.policy, args.seed + update)
                expected = fn(old.mean, old_std, self.policy.dist(obs)).mean().item()
                self.assertAlmostEqual(result["penalty_distance_final"], expected, delta=1e-9)
                self.assertEqual(ppo.beta, adapt_penalty_coef(beta, expected, args.penalty_target))


if __name__ == "__main__":
    unittest.main()
