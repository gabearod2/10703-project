# tests/test_eval.py  --  run from project/src:   python tests/test_eval.py
"""
1. load_policy rebuilds a checkpoint exactly (dims/hidden inferred); make_opponent("ckpt") gives valid actions.
2. play_match agrees with the existing evaluate() "self" setting (same seeds, deterministic play).
3. A frozen opponent is handed the right role's observation inside collect_rollout.
4. --opponent ckpt trains end to end and never touches the frozen network.
5. nashconv smoke run: output structure, gaps >= 0, NashConv = sum of the two gaps.
6. crossplay: matrix shape, and identical deterministic checkpoints give identical entries.
7. zero_sum (default) makes u_adv + u_good == 0 at every step (incl. out-of-bounds); share_vel (default) gives the runner the chaser's
   velocity; with --no_zero_sum/--no_share_vel the original env is recovered; the settings flow through train, evaluate, nashconv and crossplay.
"""
import os, sys, tempfile
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

import numpy as np, torch
from algos.networks import GaussianActorCritic
from algos.rollout import collect_rollout
from envs.tag import TagVecEnv, ADV, GOOD
from opponents import PolicyOpponent, load_policy, make_opponent
from utils.evaluate import evaluate, play_match
import exploit
import train as T

torch.manual_seed(0); np.random.seed(0)
HID = (16, 16)


def save_tiny(seed=0, path=None):
    torch.manual_seed(seed)
    env = TagVecEnv(1)
    pol = GaussianActorCritic(env.obs_dim, env.act_dim, env.act_low, env.act_high, HID)
    pol.actor_mean[-1].weight.data.normal_(0, 1.0)      # default init outputs ~0 (a no-op): make seeds behave differently
    path = path or os.path.join(tempfile.mkdtemp(), "final.pt")
    torch.save(pol.state_dict(), path)
    return pol, path


def test_load_and_make_opponent():
    pol, path = save_tiny()
    loaded = load_policy(path)
    x = torch.randn(5, pol.obs_dim)
    assert exploit.hidden_sizes(loaded) == list(HID)
    for (_, a), (_, b) in zip(pol.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b)
    assert torch.equal(loaded.dist(x).mean, pol.dist(x).mean)
    assert load_policy(os.path.dirname(path)).obs_dim == pol.obs_dim                     # run-dir spelling
    obs = np.random.randn(6, pol.obs_dim).astype(np.float32)
    for det in (True, False):
        a = make_opponent("ckpt", ADV, 5, ckpt=path, deterministic=det)(obs)
        assert a.shape == (6, 5) and a.min() >= 0 and a.max() <= 1
    d = make_opponent("ckpt", GOOD, 5, ckpt=path, deterministic=True)
    assert np.array_equal(d(obs), d(obs))                                                  # deterministic really is
    print("PASS load_policy / PolicyOpponent (dims inferred, valid actions, deterministic flag)")


def test_play_match_matches_evaluate():
    pol, _ = save_tiny(1)
    ev = evaluate(pol, num_episodes=8, seed=4242)
    opp = PolicyOpponent(pol, deterministic=True)
    m = play_match(opp, opp, num_episodes=8, seed=4242)
    assert np.isclose(m["adv"], ev["self_adv"], atol=1e-5) and np.isclose(m["good"], ev["self_good"], atol=1e-5)
    print("PASS play_match == evaluate()['self_*'] for the same seeds")


def test_opponent_gets_its_own_role_obs():
    seen = []

    class Rec:
        def __call__(self, obs):
            seen.append(obs.copy())
            return np.random.uniform(0, 1, (obs.shape[0], 5)).astype(np.float32)

    env = TagVecEnv(2, seed=0)
    pol = GaussianActorCritic(env.obs_dim, env.act_dim, env.act_low, env.act_high, HID)
    for learner, opp_role in ((ADV, GOOD), (GOOD, ADV)):
        seen.clear()
        collect_rollout(env, pol, env.reset(), [learner], Rec(), 4, 0.99, "cpu")
        assert len(seen) == 4
        for o in seen:                                              # role one-hot sits in the last two columns
            assert np.all(o[:, env.pad_dim + opp_role] == 1) and np.all(o[:, env.pad_dim + learner] == 0)
    print("PASS frozen opponent receives its own role's observation (both directions)")


def test_train_vs_frozen_ckpt():
    pol, path = save_tiny(2)
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    for learner in ("adv", "good"):
        args = T.get_args(["--learner", learner, "--opponent", "ckpt", "--opponent_ckpt", path, "--total_steps", "512",
                           "--num_envs", "4", "--num_steps", "32", "--hidden", *map(str, HID), "--eval_every", "0",
                           "--out_dir", tempfile.mkdtemp()])
        _, log = T.train(args)
        assert len(log["loss"]) == 512 // (32 * 4) and np.isfinite(log["loss"]).all()
    args = T.get_args(["--learner", "adv", "--opponent", "ckpt", "--opponent_ckpt", path, "--init_ckpt", path,
                       "--total_steps", "128", "--num_envs", "4", "--num_steps", "32", "--hidden", *map(str, HID),
                       "--eval_every", "0", "--lr", "1e-12", "--anneal_lr", "false", "--out_dir", tempfile.mkdtemp()])
    warm, _ = T.train(args)                                           # lr ~ 0, so it must still equal the checkpoint
    assert all(torch.allclose(warm.state_dict()[k], before[k], atol=1e-5) for k in before)
    after = torch.load(path)
    assert all(torch.equal(before[k], after[k]) for k in before)    # checkpoint file untouched
    try:
        T.get_args(["--learner", "adv", "--opponent", "ckpt"]); raise AssertionError("expected usage error")
    except SystemExit:
        pass
    print("PASS train.py --opponent ckpt (both roles; --init_ckpt warm start; checkpoint untouched; missing path rejected)")


def test_nashconv_smoke():
    _, path = save_tiny(3)
    out = tempfile.mkdtemp()
    r = exploit.nashconv(path, br_steps=256, br_seeds=2, episodes=6, out_dir=out,
                         br_args="--num_envs 2 --num_steps 32")
    assert os.path.exists(os.path.join(out, "exploitability.json"))
    assert r["adv"]["gap"] >= 0 and r["good"]["gap"] >= 0
    assert np.isclose(r["nashconv"], r["adv"]["gap"] + r["good"]["gap"])
    assert len(r["adv"]["candidates"]) == 2 * 2                    # 2 seeds x {deterministic, sampled}
    assert np.isclose(r["adv"]["gap_raw"], r["adv"]["br_value"] - r["adv"]["policy_value"])
    print("PASS nashconv smoke (structure, gaps >= 0, NashConv = sum of gaps)")


def test_crossplay():
    paths = [save_tiny(s)[1] for s in (4, 5)] + [save_tiny(4)[1]]   # third == first (same seed -> same weights)
    M = exploit.crossplay(paths, episodes=6, deterministic=True, out=os.path.join(tempfile.mkdtemp(), "new_dir", "cp"))   # parent dir does not exist yet
    assert M.shape == (3, 3, 2)
    assert np.allclose(M[0], M[2]) and np.allclose(M[:, 0], M[:, 2])
    assert not np.allclose(M[0], M[1])                               # different checkpoints really play differently
    print("PASS crossplay (shape; identical checkpoints are interchangeable)")


def _roll(env, steps, big=True):
    obs, rews = env.reset(), []
    rng = np.random.RandomState(0)
    for _ in range(steps):
        a = rng.uniform(0, 1, (env.num_envs, 2, env.act_dim)).astype(np.float32)
        if big:
            a[:, GOOD, 1], a[:, GOOD, 2] = 0.0, 1.0                  # runner pushes right every step -> leaves the arena
        res = env.step(a)
        rews.append(res["reward"].copy()); obs = res["obs"]
    return np.array(rews), obs


def test_zero_sum_and_share_vel():
    base, _ = _roll(TagVecEnv(3, seed=1, zero_sum=False), 60)
    assert (base[:, :, GOOD] < -1).any() and not np.allclose(base[:, :, ADV], -base[:, :, GOOD])   # default: boundary tax breaks it
    zs, _ = _roll(TagVecEnv(3, seed=1), 60)                              # zero_sum is the default
    assert np.allclose(zs[:, :, ADV], -zs[:, :, GOOD])                                              # exactly zero-sum
    assert np.allclose(zs[:, :, GOOD], base[:, :, GOOD])                                            # runner's reward unchanged
    env = TagVecEnv(2, seed=1)                                          # share_vel is the default
    obs = env.reset()
    for _ in range(5):
        res = env.step(np.random.uniform(0, 1, (2, 2, env.act_dim)).astype(np.float32)); obs = res["obs"]
    for i in range(2):
        v = TagVecEnv.world(env.envs[i]).agents[ADV].state.p_vel
        assert np.allclose(obs[i, GOOD, 10:12], v) and np.abs(v).sum() > 0
        assert np.allclose(obs[i, ADV, 10:12], TagVecEnv.world(env.envs[i]).agents[GOOD].state.p_vel)   # chaser's own view unchanged
    off = TagVecEnv(2, seed=1, share_vel=False).reset()
    assert np.all(off[:, GOOD, 10:12] == 0) and off.shape == obs.shape                              # off: zero padding, same obs_dim
    print("PASS zero_sum (sum == 0 incl. out-of-bounds, runner reward unchanged) and share_vel (runner sees chaser velocity)")


def test_env_flags_flow_through():
    d = tempfile.mkdtemp()
    assert T.get_args([]).zero_sum and T.get_args([]).share_vel                                     # defaults on
    off = T.get_args(["--no_zero_sum", "--no_share_vel"]); assert not off.zero_sum and not off.share_vel
    args = T.get_args(["--total_steps", "256", "--num_envs", "4", "--num_steps", "32",
                       "--hidden", *map(str, HID), "--eval_every", "1", "--eval_episodes", "4", "--out_dir", d])
    pol, log = T.train(args)
    assert np.isfinite(log["loss"]).all()
    e = log["eval"][-1]
    assert abs(e["self_adv"] + e["self_good"]) < 1e-4                                                # evaluate() is zero-sum too
    path = os.path.join(d, "ckpt.pt"); torch.save(pol.state_dict(), path)
    cfg_dir = os.path.join(d, "run"); os.makedirs(cfg_dir)
    torch.save(pol.state_dict(), os.path.join(cfg_dir, "final.pt"))
    import json
    json.dump(vars(args), open(os.path.join(cfg_dir, "config.json"), "w"))
    r = exploit.nashconv(cfg_dir, br_steps=128, br_seeds=1, episodes=4, out_dir=os.path.join(d, "x"),
                         br_args="--num_envs 2 --num_steps 32")
    assert r["zero_sum"] and r["share_vel"]
    old = {k: v for k, v in vars(args).items() if k not in ("zero_sum", "share_vel")}                # run from before the flags existed
    assert exploit.env_options(old) == {"zero_sum": False, "share_vel": False}
    fz = PolicyOpponent(load_policy(cfg_dir), False)
    m = play_match(fz, fz, 6, 25, 5, True)
    assert abs(m["adv"] + m["good"]) < 1e-4                                                          # one match: returns sum to 0
    for role in ("adv", "good"):
        cfg = json.load(open(os.path.join(r[role]["selected"]["run_dir"], "config.json")))
        assert cfg["zero_sum"] and cfg["share_vel"]                                                  # BR trained in the same game
    exploit.crossplay([cfg_dir, cfg_dir], episodes=4, out=os.path.join(d, "cp"))
    print("PASS zero_sum/share_vel reach train, evaluate, nashconv (BR runs too) and crossplay")


if __name__ == "__main__":
    test_load_and_make_opponent(); test_play_match_matches_evaluate(); test_opponent_gets_its_own_role_obs()
    test_train_vs_frozen_ckpt(); test_nashconv_smoke(); test_crossplay()
    test_zero_sum_and_share_vel(); test_env_flags_flow_through()
    print("\nALL OK")
