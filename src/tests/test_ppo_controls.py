"""Run from src: python tests/test_ppo_controls.py (stdlib unittest; no extra dependencies)."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import numpy as np
import torch

from algos.ppo import PPO, adapt_penalty_coef, gaussian_kl, gaussian_wasserstein2, ppo_loss
import train


class TestLoss(unittest.TestCase):
    def test_wasserstein_values_and_gradients(self):
        for mean, std, expected in [([0., 0.], [1., 1.], 0.),
                                    ([3., 4.], [1., 1.], 5.),
                                    ([0., 0.], [4., 5.], 5.),
                                    ([3., 0.], [1., 5.], 5.)]:
            old_mean = torch.zeros(1, 2, requires_grad=True)
            old_std = torch.ones(1, 2, requires_grad=True)
            new_mean = torch.tensor([mean], requires_grad=True)
            new_std = torch.tensor([std], requires_grad=True)
            distance = gaussian_wasserstein2(old_mean, old_std, torch.distributions.Normal(new_mean, new_std))
            self.assertAlmostEqual(distance.item(), expected)
            distance.sum().backward()
            self.assertIsNone(old_mean.grad)
            self.assertIsNone(old_std.grad)
            for grad in (new_mean.grad, new_std.grad):
                self.assertTrue(torch.isfinite(grad).all())
                if expected == 0:
                    self.assertTrue((grad == 0).all())
            if expected:
                self.assertGreater(new_mean.grad.abs().sum() + new_std.grad.abs().sum(), 0)

    def test_analytical_gaussian_kl(self):
        old_mean = torch.tensor([[0., 1.], [2., -1.]], dtype=torch.float64)
        old_std = torch.tensor([[1., 2.], [.5, 1.5]], dtype=torch.float64)
        new_mean = torch.tensor([[1., -.5], [-1., .5]], dtype=torch.float64)
        new_std = torch.tensor([[2., 1.], [1.5, .75]], dtype=torch.float64)
        identical = gaussian_kl(old_mean, old_std, torch.distributions.Normal(old_mean, old_std))
        torch.testing.assert_close(identical, torch.zeros(2, dtype=torch.float64))
        expected = (np.log(new_std.numpy() / old_std.numpy())
                    + (old_std.numpy() ** 2 + (old_mean.numpy() - new_mean.numpy()) ** 2)
                    / (2 * new_std.numpy() ** 2) - .5).sum(axis=-1)
        actual = gaussian_kl(old_mean, old_std, torch.distributions.Normal(new_mean, new_std))
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)

    def test_penalty_gradients_only_reach_new_distribution(self):
        old_mean = torch.zeros(3, 2, requires_grad=True)
        old_std = torch.ones(3, 2, requires_grad=True)
        new_mean = torch.full((3, 2), .5, requires_grad=True)
        new_log_std = torch.full((2,), .2, requires_grad=True)
        kl = gaussian_kl(old_mean, old_std, torch.distributions.Normal(new_mean, new_log_std.exp()))
        zero = torch.zeros(3)
        loss, _ = ppo_loss(zero, zero, zero, zero, zero, zero, zero, .1, .5, .05,
                           distance=kl, penalty_coef=2.)
        loss.backward()
        self.assertIsNone(old_mean.grad)
        self.assertIsNone(old_std.grad)
        self.assertTrue(torch.isfinite(new_mean.grad).all())
        self.assertTrue(torch.isfinite(new_log_std.grad).all())
        self.assertTrue((new_mean.grad.abs() > 0).all())
        self.assertTrue((new_log_std.grad.abs() > 0).all())

    def test_objective_combinations_and_shared_value_clip(self):
        zero = torch.zeros(2, dtype=torch.float64)
        inputs = dict(new_logp=torch.tensor([1.5, .5], dtype=torch.float64).log(), old_logp=zero,
                      adv=torch.tensor([2., -2.], dtype=torch.float64),
                      new_v=torch.tensor([1., -1.], dtype=torch.float64), old_v=zero,
                      ret=torch.tensor([2., -2.], dtype=torch.float64),
                      entropy=torch.tensor([2., 4.], dtype=torch.float64),
                      clip_coef=.1, vf_coef=.5, ent_coef=.05)
        for clip in (False, True):
            for penalize in (False, True):
                for clip_value in (False, True):
                    with self.subTest(clip=clip, penalize=penalize, clip_value=clip_value):
                        loss, stats = ppo_loss(**inputs, clip_policy=clip, clip_vloss=clip_value,
                                              distance=torch.tensor([.2, .4], dtype=torch.float64) if penalize else None,
                                              penalty_coef=.7 if penalize else 0.)
                        expected_pg = -.2 if clip else -1.
                        expected_value = 1.805 if clip_value else .5
                        expected_penalty = .21 if penalize else 0.
                        self.assertAlmostEqual(stats["policy_loss"], expected_pg)
                        self.assertAlmostEqual(stats["value_loss"], expected_value)
                        self.assertAlmostEqual(stats["penalty_loss"], expected_penalty)
                        self.assertAlmostEqual(loss.item(), expected_pg + .5 * expected_value - .05 * 3 + expected_penalty)
        default_loss, default_stats = ppo_loss(**inputs)
        self.assertAlmostEqual(default_loss.item(), -.2 + .5 * 1.805 - .05 * 3)
        self.assertEqual(default_stats["penalty_loss"], 0.)

    def test_coefficient_thresholds(self):
        target, coef = .01, 2.
        for kl, expected in [(0., 1.), (target / 1.5 - 1e-8, 1.), (target / 1.5, 2.),
                             (target, 2.), (target * 1.5, 2.), (target * 1.5 + 1e-8, 4.)]:
            with self.subTest(kl=kl):
                self.assertEqual(adapt_penalty_coef(coef, kl, target), expected)


class _ToyPolicy(torch.nn.Module):
    """Observation-dependent means make full-batch and minibatch KL distinguishable."""
    def __init__(self):
        super().__init__()
        self.mean_weight = torch.nn.Parameter(torch.zeros(2))
        self.log_std = torch.nn.Parameter(torch.zeros(2))
        self.value_bias = torch.nn.Parameter(torch.zeros(()))

    def dist(self, obs):
        return torch.distributions.Normal(obs * self.mean_weight, self.log_std.exp())

    def value(self, obs):
        return self.value_bias.expand(len(obs))


def make_batch(policy):
    obs = torch.arange(1, 17, dtype=torch.float32).reshape(8, 2) / 10
    act = torch.zeros_like(obs)
    with torch.no_grad():
        logp = policy.dist(obs).log_prob(act).sum(-1)
        val = policy.value(obs).clone()
    return dict(obs=obs, act=act, logp=logp, val=val, adv=torch.ones(8), ret=torch.ones(8))


class TestUpdate(unittest.TestCase):
    def test_adaptation_uses_final_policy_once_and_carries_to_next_update(self):
        for mode, early_stop in [(m, e) for m in ("kl", "wasserstein-2") for e in (False, True)]:
            with self.subTest(early_stop=early_stop):
                policy = _ToyPolicy()
                args = SimpleNamespace(lr=.01, penalty_coef_init=1., penalty=mode, penalty_target=.001,
                                       num_minibatches=2, update_epochs=3, norm_adv=False,
                                       clip_coef=.1, vf_coef=.5, ent_coef=.05, clip_vloss=True,
                                       clip_policy=True, max_grad_norm=.5,
                                       target_kl=1e-12 if early_stop else None)
                ppo = PPO(policy, args)
                for expected_coef in (1., 2.):
                    batch = make_batch(policy)
                    with torch.no_grad():
                        old = policy.dist(batch["obs"])
                        old_mean, old_std = old.mean.clone(), old.stddev.clone()
                    step_coefficients = []

                    def controlled_step():
                        # Predictable policy movement isolates update ordering from optimizer behavior.
                        step_coefficients.append(ppo.beta)
                        with torch.no_grad():
                            policy.mean_weight.add_(.2)
                            policy.log_std.add_(.02)

                    with patch.object(ppo.opt, "step", side_effect=controlled_step), \
                         patch("algos.ppo.adapt_penalty_coef", wraps=adapt_penalty_coef) as adapt:
                        stats = ppo.update(batch)
                    with torch.no_grad():
                        expected_kl = torch.distributions.kl_divergence(
                            torch.distributions.Normal(old_mean, old_std), policy.dist(batch["obs"])
                        ).sum(-1).mean().item()
                    self.assertEqual(step_coefficients, [expected_coef] * (2 if early_stop else 6))
                    self.assertAlmostEqual(stats["kl_exact_final"], expected_kl)
                    expected_distance = (expected_kl if mode == "kl" else
                                         gaussian_wasserstein2(old_mean, old_std, policy.dist(batch["obs"])).mean().item())
                    self.assertAlmostEqual(stats["penalty_distance_final"], expected_distance)
                    adapt.assert_called_once_with(expected_coef, expected_distance, args.penalty_target)
                    self.assertEqual(stats["penalty_coef_used"], expected_coef)
                    self.assertEqual(stats["penalty_coef_next"], expected_coef * 2)
                    self.assertEqual(ppo.beta, expected_coef * 2)
                    self.assertGreater(stats["penalty_loss"], 0.)

    def test_disabled_penalty_does_not_adapt(self):
        policy = _ToyPolicy()
        args = SimpleNamespace(lr=.01, penalty_coef_init=3., penalty="none", penalty_target=.01,
                               num_minibatches=2, update_epochs=2, norm_adv=False,
                               clip_coef=.1, vf_coef=.5, ent_coef=.05, clip_vloss=True,
                               clip_policy=True, max_grad_norm=.5, target_kl=None)
        ppo = PPO(policy, args)
        with patch("algos.ppo.adapt_penalty_coef") as adapt:
            stats = ppo.update(make_batch(policy))
        adapt.assert_not_called()
        self.assertEqual(ppo.beta, 3.)
        for key in ("penalty_coef_used", "penalty_coef_next", "penalty_loss", "penalty_distance_final"):
            self.assertEqual(stats[key], 0.)
        self.assertGreater(stats["kl_exact_final"], 0.)


class TestRunConfiguration(unittest.TestCase):
    def test_concurrent_runs_get_distinct_directories(self):
        with tempfile.TemporaryDirectory() as root, patch("train.datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 6, 14, 30)
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(train.create_run_dir, [root] * 24))
            self.assertEqual(len(set(paths)), 24)
            for path in paths:
                self.assertTrue(Path(path).is_dir())
                self.assertTrue(Path(path).name.startswith("20261106-143000-"))
                UUID(Path(path).name.split("-")[-1])

    def test_collision_retries_without_overwriting(self):
        with tempfile.TemporaryDirectory() as root, patch("train.datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 6, 14, 30)
            with patch("train.uuid4", return_value=UUID(int=1)):
                existing = Path(train.create_run_dir(root))
            marker = existing / "config.json"
            marker.write_text('{"seed": 17}')
            with patch("train.uuid4", side_effect=[UUID(int=1), UUID(int=2)]) as make_id:
                created = Path(train.create_run_dir(root))
            self.assertEqual(make_id.call_count, 2)
            self.assertNotEqual(created, existing)
            self.assertEqual(marker.read_text(), '{"seed": 17}')
            self.assertEqual(list(created.iterdir()), [])

    def test_invalid_kl_parameters_fail_before_directory_creation(self):
        for flag in ("--penalty_target", "--penalty_coef_init"):
            for value in ("0", "-1", "nan", "inf", "-inf"):
                with self.subTest(flag=flag, value=value), patch("train.create_run_dir") as create, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        train.get_args([f"{flag}={value}"])
                    self.assertEqual(error.exception.code, 2)
                    create.assert_not_called()

    def test_invalid_selector(self):
        with patch("train.create_run_dir") as create, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train.get_args(["--penalty", "wasserstein"])
            create.assert_not_called()

    def test_default_and_independent_switches(self):
        with tempfile.TemporaryDirectory() as root:
            defaults = train.get_args(["--out_dir", root])
            self.assertTrue(defaults.clip_policy)
            self.assertEqual(defaults.penalty, "none")
            self.assertEqual(defaults.penalty_target, .01)
            self.assertEqual(defaults.penalty_coef_init, 1.)
            self.assertIsNone(defaults.target_kl)
            for clip, kl in [(c, m) for c in ("false", "true") for m in ("none", "kl", "wasserstein-2")]:
                args = train.get_args(["--out_dir", root, "--clip_policy", clip, "--penalty", kl])
                self.assertEqual(args.clip_policy, clip == "true")
                self.assertEqual(args.penalty, kl)
                self.assertTrue(args.clip_vloss)
                self.assertEqual(args.clip_coef, .1)


class TestTrainingModes(unittest.TestCase):
    def test_short_training_runs_save_config_metrics_and_loadable_policy(self):
        from algos.networks import GaussianActorCritic

        with tempfile.TemporaryDirectory() as root:
            for clip, kl in [(c, m) for c in ("false", "true") for m in ("none", "kl", "wasserstein-2")]:
                with self.subTest(clip=clip, kl=kl):
                    base = Path(root) / f"clip_{clip}_kl_{kl}"
                    result = subprocess.run(
                        [sys.executable, "-B", str(SRC / "train.py"), "--out_dir", str(base),
                         "--total_steps", "512", "--num_envs", "2", "--num_steps", "32",
                         "--hidden", "16", "16", "--threads", "1", "--update_epochs", "2",
                         "--num_minibatches", "2", "--eval_every", "8", "--eval_episodes", "2",
                         "--clip_policy", clip, "--penalty", kl],
                        cwd=SRC, capture_output=True, text=True, timeout=120,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    run_dirs = list(base.iterdir())
                    self.assertEqual(len(run_dirs), 1)
                    run = run_dirs[0]
                    cfg = json.loads((run / "config.json").read_text())
                    self.assertEqual(cfg["out_dir"], str(run))
                    self.assertEqual(cfg["clip_policy"], clip == "true")
                    self.assertEqual(cfg["penalty"], kl)
                    log = json.loads((run / "training_log.json").read_text())
                    for key in ("loss", "kl_exact_final", "penalty_coef_used", "penalty_coef_next", "penalty_loss", "penalty_distance_final"):
                        self.assertEqual(len(log[key]), 8)
                        self.assertTrue(np.isfinite(log[key]).all(), key)
                    if kl != "none":
                        self.assertEqual(log["penalty_coef_used"][0], cfg["penalty_coef_init"])
                        self.assertEqual(log["penalty_coef_used"][1:], log["penalty_coef_next"][:-1])
                        self.assertTrue(np.all(np.array(log["penalty_coef_next"]) > 0))
                    else:
                        self.assertEqual(log["penalty_loss"], [0.] * 8)
                        self.assertEqual(log["penalty_coef_used"], [0.] * 8)
                    self.assertEqual(len(log["eval"]), 1)
                    for name in ("final.pt", "latest.pt", "training_curves.png"):
                        self.assertTrue((run / name).is_file(), name)
                    policy = GaussianActorCritic(14, 5, np.zeros(5), np.ones(5), hidden=(16, 16))
                    policy.load_state_dict(torch.load(run / "final.pt", map_location="cpu", weights_only=True))
                    actions, _, _ = policy.act(torch.zeros(4, 14))
                    env_actions = policy.to_env(actions)
                    self.assertTrue(torch.isfinite(env_actions).all())
                    self.assertTrue(((env_actions >= 0) & (env_actions <= 1)).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
