# tests/test_sanity.py  --  run from project/src:   python tests/test_sanity.py
"""
1. GAE and PPO losses match the (verified) HW3 PPOAgent numerically.
2. Env wrapper: shapes, role ids, time-limit truncation, auto-reset, identical dynamics.
3. Gaussian policy: log-prob matches torch Normal, to_env stays in bounds.
4. Opponents return valid actions.
5. Tiny end-to-end training run.
"""
import os, sys
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)                                    # our packages: envs, algos, utils
sys.path.append(os.path.join(SRC, "hw3_reference"))        # HW3 code (flat imports); appended so it can't shadow ours

import numpy as np, torch
from ppo_agent import PPOAgent                             # HW3 implementation (reference)
from algos.ppo import compute_gae, ppo_loss
from algos.networks import GaussianActorCritic
from envs.tag import TagVecEnv, ADV, GOOD
from opponents import make_opponent
import train as T

torch.manual_seed(0); np.random.seed(0)


class _StubActor:
    """Stands in for HW's actor so the reference agent can be driven with our numbers."""
    def __init__(self, fn): self.fn = fn
    def __call__(self, obs): return self.fn(obs)
    def parameters(self): return []


def make_ref_agent(**kw):
    info = dict(obs_dim=14, act_dim=5, act_low=np.zeros(5), act_high=np.ones(5))
    return PPOAgent(info, **kw)


def test_gae_matches_hw3():
    T, K, gamma, lam = 40, 6, 0.99, 0.95
    r = torch.randn(T, K); v = torch.randn(T, K)
    d = (torch.rand(T, K) < 0.1).float(); last = torch.randn(K)
    adv, ret = compute_gae(r, v, d, last, gamma, lam)
    ag = make_ref_agent(gamma=gamma, gae_lambda=lam)
    for k in range(K):
        ag.actor = _StubActor(lambda o, fv=float(last[k]): (None, torch.tensor([fv])))
        rollout = [dict(reward=float(r[t, k]), value=float(v[t, k]), done=bool(d[t, k]), next_obs=np.zeros(14)) for t in range(T)]
        ref_adv, ref_ret = ag._compute_gae(rollout)
        assert np.allclose(ref_adv, adv[:, k].numpy(), atol=1e-5), f"adv mismatch col {k}"
        assert np.allclose(ref_ret, ret[:, k].numpy(), atol=1e-5), f"ret mismatch col {k}"
    print("PASS gae == HW3 _compute_gae (incl. mid-rollout dones)")


def test_loss_matches_hw3():
    B, clip = 256, 0.1
    old_lp = torch.randn(B) * 0.3; new_lp = old_lp + torch.randn(B) * 0.2
    adv = torch.randn(B); ret = torch.randn(B); v = torch.randn(B); ent = torch.rand(B) + 1
    ag = make_ref_agent(clip_coef=clip, vf_coef=0.5, ent_coef=0.05)

    class _Dist:
        def log_prob(self, a): return new_lp
        def entropy(self): return ent
    ag.actor = _StubActor(lambda o: (_Dist(), v))
    batch = dict(obs=torch.zeros(B, 14), actions=torch.zeros(B, 5), log_probs=old_lp, advantages=adv, returns=ret)
    ref_total, ref = ag._ppo_loss(batch)
    total, mine = ppo_loss(new_lp, old_lp, adv, v, v.detach(), ret, ent, clip, 0.5, 0.05, clip_vloss=False)
    assert abs(ref["policy_loss"] - mine["policy_loss"]) < 1e-6, "policy loss differs"
    assert abs(ref["entropy"] - mine["entropy"]) < 1e-6
    assert abs(ref["clipfrac"] - mine["clipfrac"]) < 1e-9
    assert abs(ref["value_loss"] * 0.5 - mine["value_loss"]) < 1e-6      # ours carries the CleanRL/OpenSpiel 0.5 factor
    print("PASS policy loss / entropy / clipfrac == HW3 _ppo_loss; value loss == 0.5 * HW3 (documented factor)")


def test_env():
    env = TagVecEnv(num_envs=3, max_cycles=25, seed=1)
    o = env.reset()
    assert o.shape == (3, 2, 14) and env.act_dim == 5
    assert np.all(o[:, 0, 12] == 1) and np.all(o[:, 1, 13] == 1) and np.all(o[:, 1, 10:12] == 0)  # roles + padding
    for t in range(25):
        res = env.step(np.random.uniform(0, 1, (3, 2, 5)))
        assert res["truncated"].all() == (t == 24) and not res["terminated"].any()
    assert len(res["episodes"]) == 3 and res["episodes"][0]["len"] == 25
    res = env.step(np.random.uniform(0, 1, (3, 2, 5)))                 # auto-reset worked
    assert not res["truncated"].any()
    ws = env.world(env.envs[0]).agents
    assert (ws[0].size, ws[0].accel, ws[0].max_speed) == (ws[1].size, ws[1].accel, ws[1].max_speed)   # symmetric default
    oe = TagVecEnv(num_envs=1, symmetric=False)
    orig = oe.world(oe.envs[0]).agents
    assert orig[0].max_speed == 1.0 and orig[1].max_speed == 1.3                                  # original MPE2 roles
    print("PASS env wrapper (shapes, roles, truncation @25, auto-reset, identical dynamics by default)")


def test_policy():
    pol = GaussianActorCritic(14, 5, np.zeros(5), np.ones(5), hidden=(32, 32))
    x = torch.randn(7, 14); a, lp, v = pol.act(x)
    ref = torch.distributions.Normal(pol.actor_mean(x), pol.log_std.exp()).log_prob(a).sum(-1)
    assert torch.allclose(lp, ref, atol=1e-5)
    lp2, ent, v2 = pol.evaluate(x, a)
    assert torch.allclose(lp, lp2, atol=1e-5) and torch.allclose(v, v2, atol=1e-5)
    e = pol.to_env(torch.randn(1000, 5) * 5)
    assert e.min() >= 0 and e.max() <= 1
    assert torch.allclose(pol.to_env(torch.zeros(1, 5)), torch.full((1, 5), 0.5))
    print("PASS gaussian policy (log-prob, evaluate consistency, action bounds)")


def test_opponents():
    obs = np.random.randn(6, 14).astype(np.float32)
    for name in ("random", "scripted"):
        for role in (ADV, GOOD):
            a = make_opponent(name, role, 5, seed=0)(obs)
            assert a.shape == (6, 5) and a.min() >= 0 and a.max() <= 1
    print("PASS opponents (random / scripted give valid [N,5] actions in [0,1])")


def test_train_smoke():
    import tempfile, os
    args = T.get_args(["--total_steps", "4096", "--num_envs", "4", "--num_steps", "64", "--hidden", "32", "32",
                        "--eval_every", "1", "--eval_episodes", "4", "--out_dir", tempfile.mkdtemp(), "--log_every", "1"])
    pol, log = T.train(args)
    assert len(log["loss"]) == 4096 // (64 * 4) and np.isfinite(log["loss"]).all()
    assert os.path.exists(os.path.join(args.out_dir, "final.pt"))
    print("PASS end-to-end smoke run")


if __name__ == "__main__":
    test_gae_matches_hw3(); test_loss_matches_hw3(); test_env(); test_policy(); test_opponents(); test_train_smoke()
    print("\nALL OK")
