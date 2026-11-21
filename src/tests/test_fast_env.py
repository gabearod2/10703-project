"""Parity: envs/fast_tag.FastTagVecEnv must reproduce MPE2 Simple Tag (envs/tag.TagVecEnv) step for step."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import numpy as np

from envs.tag import TagVecEnv, ADV, GOOD
from envs.fast_tag import FastTagVecEnv


def mpe_state(env):
    pos = np.array([[a.state.p_pos for a in TagVecEnv.world(e).agents] for e in env.envs])
    vel = np.array([[a.state.p_vel for a in TagVecEnv.world(e).agents] for e in env.envs])
    lm = np.array([[l.state.p_pos for l in TagVecEnv.world(e).landmarks] for e in env.envs])
    return pos, vel, lm


def set_mpe_state(env, pos, vel, lm):
    for i, e in enumerate(env.envs):
        w = TagVecEnv.world(e)
        for k, a in enumerate(w.agents):
            a.state.p_pos, a.state.p_vel = pos[i, k].copy(), vel[i, k].copy()
        for j, l in enumerate(w.landmarks):
            l.state.p_pos = lm[i, j].copy()


class FastEnvParity(unittest.TestCase):
    def check(self, n=64, crowd=False, **kw):
        rng = np.random.default_rng(0)
        ref, fast = TagVecEnv(n, seed=3, **kw), FastTagVecEnv(n, seed=3, **kw)
        self.assertEqual((ref.obs_dim, ref.act_dim, ref.pad_dim), (fast.obs_dim, fast.act_dim, fast.pad_dim))
        ref.reset()
        pos, vel, lm = mpe_state(ref)
        if crowd:      # start overlapping, on top of landmarks and out of bounds so every force/penalty branch fires
            pos[:, GOOD] = pos[:, ADV] + rng.normal(0, 0.05, (n, 2))
            lm[:, 0] = pos[:, ADV] + rng.normal(0, 0.1, (n, 2))
            pos[: n // 4] *= 1.6
            set_mpe_state(ref, pos, vel, lm)
        obs = fast.set_state(pos, vel, lm)
        hits = 0
        for t in range(ref.max_cycles):
            act = rng.uniform(-0.3, 1.3, (n, 2, 5))             # also exercises action clipping
            r, f = ref.step(act), fast.step(act)
            for key in ("final_obs", "reward", "truncated", "terminated"):
                np.testing.assert_allclose(f[key], r[key], rtol=1e-4, atol=1e-4, err_msg=f"{key} at t={t}")
            hits += int((r["reward"][:, GOOD] <= -10).sum())
        np.testing.assert_allclose([e["ret"] for e in f["episodes"]], [e["ret"] for e in r["episodes"]], atol=1e-3)
        ref.close()
        return hits

    def test_random_states(self):
        self.check()

    def test_contacts_and_bounds(self):
        self.assertGreater(self.check(crowd=True), 0)

    def test_asymmetric_non_zero_sum_no_share_vel(self):
        self.check(crowd=True, symmetric=False, zero_sum=False, share_vel=False)

    def test_reset_and_crn(self):
        a, b = FastTagVecEnv(5, seed=7), FastTagVecEnv(5, seed=7)
        np.testing.assert_array_equal(a.reset(), b.reset())   # same seed -> same initial states (common random numbers)
        o = a.reset()
        self.assertEqual(o.shape, (5, 2, 14))
        self.assertTrue(np.all(o[:, ADV, 12] == 1) and np.all(o[:, GOOD, 13] == 1))
        for t in range(25):
            res = a.step(np.random.uniform(0, 1, (5, 2, 5)))
            self.assertEqual(res["truncated"].all(), t == 24)
        self.assertEqual(len(res["episodes"]), 5)
        self.assertFalse(np.allclose(res["obs"], res["final_obs"]))   # auto-reset obs differs from the true final obs
        self.assertTrue(np.all(a._t == 0))


if __name__ == "__main__":
    unittest.main()
