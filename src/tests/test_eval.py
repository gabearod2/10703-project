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
8. PoolOpponent: every env plays the member it drew, keeps it for the episode, re-draws at episode end, uniformly.
9. Per-env learner roles in collect_rollout: learner / opponent each see their own role, rewards are the learner's.
11. no_op masking (default): 4-dim policy, env dim 0 always 0, other dims unchanged; old 5-dim checkpoints still
    load and get a 5-dim best response; mismatched --init_ckpt is rejected.
10. --algo fsp: trains, snapshots to <run>/pool, the pool grows; nashconv --average and crossplay accept the pool.
"""
import os, sys, tempfile
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

import numpy as np, torch
from algos.networks import GaussianActorCritic
from algos.rollout import collect_rollout
from envs.tag import TagVecEnv, ADV, GOOD
from opponents import PolicyOpponent, PoolOpponent, load_policy, make_opponent, pool_files
from utils.evaluate import evaluate, play_match
import exploit
import train as T

torch.manual_seed(0); np.random.seed(0)
HID = (16, 16)


def save_tiny(seed=0, path=None):
    torch.manual_seed(seed)
    env = TagVecEnv(1)
    pol = GaussianActorCritic(env.obs_dim, env.act_dim - 1, env.act_low, env.act_high, HID)      # no_op masked (default)
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


def test_pool_opponent():
    env = TagVecEnv(1)
    pool = PoolOpponent(seed=0, deterministic=True)
    members = []
    for s in range(3):
        torch.manual_seed(s)
        m = GaussianActorCritic(env.obs_dim, env.act_dim, env.act_low, env.act_high, HID)
        with torch.no_grad():
            m.actor_mean[-1].bias.copy_(torch.randn(env.act_dim))    # make the members clearly different
        members.append(m); pool.add(m)
    with torch.no_grad():
        members[0].actor_mean[-1].bias.add_(5.0)                     # the pool holds frozen COPIES
    obs = np.random.randn(64, env.obs_dim).astype(np.float32)
    act = pool(obs)
    first = pool.assign.copy()
    for i in range(64):                                              # each env is played by the member it drew
        m = pool.members[first[i]]
        want = m.to_env(m.act(torch.as_tensor(obs[i:i + 1]), True)[0]).numpy()[0]
        assert np.allclose(act[i], want, atol=1e-6)
    assert not np.allclose(pool.members[0].actor_mean[-1].bias.numpy(), members[0].actor_mean[-1].bias.detach().numpy())
    pool(obs); assert np.array_equal(pool.assign, first)             # kept within an episode
    done = np.zeros(64, bool); done[:32] = True
    pool.episode_end(done)
    assert np.array_equal(pool.assign[32:], first[32:]) and not np.array_equal(pool.assign[:32], first[:32])
    counts = np.zeros(3)
    for _ in range(300):
        pool.episode_end(np.ones(64, bool)); counts += np.bincount(pool.assign, minlength=3)
    assert np.all(np.abs(counts / counts.sum() - 1 / 3) < 0.02)      # uniform over members
    pool.reseed(5); pool(obs); a5 = pool.assign.copy(); pool.reseed(5); pool(obs)
    assert np.array_equal(a5, pool.assign)                           # reproducible
    print("PASS PoolOpponent (frozen copies, per-env member, kept per episode, uniform re-draw, reseed)")


def test_per_env_roles_rollout():
    seen = []

    class Rec:
        ended = 0
        def __call__(self, obs):
            seen.append(obs.copy())
            return np.random.uniform(0, 1, (obs.shape[0], 5)).astype(np.float32)
        def episode_end(self, done):
            Rec.ended += int(done.sum())

    env = TagVecEnv(4, seed=0, zero_sum=False)
    pol = GaussianActorCritic(env.obs_dim, env.act_dim, env.act_low, env.act_high, HID)
    roles = np.arange(4) % 2
    roll = collect_rollout(env, pol, env.reset(), roles, Rec(), 30, 0.99, "cpu")
    assert roll["obs"].shape == (30, 4, env.obs_dim)
    for i, r in enumerate(roles):                                    # learner sees its role, opponent the other one
        assert torch.all(roll["obs"][:, i, env.pad_dim + r] == 1) and torch.all(roll["obs"][:, i, env.pad_dim + 1 - r] == 0)
        assert all(o[i, env.pad_dim + 1 - r] == 1 for o in seen)
    assert Rec.ended == 4 and len(roll["episodes"]) == 4 and sorted(roll["episode_env"]) == [0, 1, 2, 3]
    rew = roll["rew"][:24]                                           # before the time-limit bootstrap at step 25
    assert (rew[:, roles == ADV] >= 0).all() and (rew[:, roles == GOOD] <= 0).all()     # chaser >= 0, runner <= 0
    env2 = TagVecEnv(4, seed=0)
    one = collect_rollout(env2, pol, env2.reset(), [GOOD], Rec(), 3, 0.99, "cpu")
    assert one["obs"].shape == (3, 4, env.obs_dim) and torch.all(one["obs"][:, :, env.pad_dim + GOOD] == 1)
    print("PASS per-env learner roles (own-role obs for learner and opponent, learner's reward, episode_end called)")


def test_fsp_end_to_end():
    import json
    d = tempfile.mkdtemp()
    args = T.get_args(["--algo", "fsp", "--total_steps", "2048", "--num_envs", "4", "--num_steps", "32", "--snapshot_every", "4",
                       "--hidden", *map(str, HID), "--eval_every", "0", "--out_dir", d])
    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"))
    pol, log = T.train(args)
    snaps = pool_files(os.path.join(args.out_dir, "pool"))
    assert [os.path.basename(s) for s in snaps] == [f"snap_{i:06d}.pt" for i in (0, 4, 8, 12)]      # 16 iterations
    assert np.isfinite(log["loss"]).all() and len(log["loss"]) == 16
    first, last = load_policy(snaps[0]), load_policy(snaps[-1])
    assert not torch.allclose(first.actor_mean[0].weight, last.actor_mean[0].weight)                 # snapshots differ over time
    assert T.get_args(["--out_dir", d]).snapshot_every == 0                                          # plain self-play: off by default
    for bad in (["--algo", "fsp", "--learner", "adv"], ["--algo", "fsp", "--snapshot_every", "0"],
                ["--learner", "adv", "--opponent", "pool"]):
        try:
            T.get_args(bad + ["--out_dir", d]); raise AssertionError("expected usage error")
        except SystemExit:
            pass
    r = exploit.nashconv(args.out_dir, br_steps=128, br_seeds=1, episodes=6, average=True,
                         br_args="--num_envs 2 --num_steps 32")
    assert "average of 4 snapshots" in r["policy"] and r["nashconv"] >= 0
    assert abs(r["nashconv"] - (r["adv"]["gap"] + r["good"]["gap"])) < 1e-9
    assert os.path.exists(os.path.join(args.out_dir, "exploit_avg", "exploitability.json"))
    cfg = json.load(open(os.path.join(r["adv"]["selected"]["run_dir"], "config.json")))
    assert cfg["opponent"] == "pool" and cfg["opponent_ckpt"].endswith("pool")                       # BR trained vs the mixture
    from unittest.mock import patch
    pool = os.path.join(args.out_dir, "pool")
    assert all(exploit.checkpoint_label(i, path).endswith(os.path.basename(path)[:-3]) for i, path in enumerate(snaps))
    assert exploit.checkpoint_label(0, pool).endswith(" (avg)")
    with patch.object(exploit, "load_policy", wraps=load_policy) as load, patch.object(
            PoolOpponent, "from_dir", wraps=PoolOpponent.from_dir) as load_pool:
        M = exploit.crossplay([snaps[0], pool], episodes=6, out=os.path.join(d, "cp"), labels=["initial", "mixture"])
    load.assert_called_once()
    load_pool.assert_called_once()
    with open(os.path.join(d, "cp.json")) as f:
        assert json.load(f)["labels"] == ["initial", "mixture"]
    reference = play_match(PoolOpponent.from_dir(pool, 0), PoolOpponent.from_dir(pool, 0), 6,
                           args.max_cycles, 12345, args.symmetric, zero_sum=args.zero_sum, share_vel=args.share_vel)
    np.testing.assert_array_equal(M[1, 1], [reference["adv"], reference["good"]])
    assert M.shape == (2, 2, 2) and np.allclose(M[:, :, 0], -M[:, :, 1])                             # zero-sum game
    print("PASS --algo fsp (snapshots, pool grows), nashconv --average (BR vs mixture), crossplay with a pool")


def test_mask_noop():
    env = TagVecEnv(1)
    torch.manual_seed(0)
    m4 = GaussianActorCritic(env.obs_dim, env.act_dim - 1, env.act_low, env.act_high, HID)
    m5 = GaussianActorCritic(env.obs_dim, env.act_dim, env.act_low, env.act_high, HID)
    assert m4.mask_noop and not m5.mask_noop and m4.log_std.shape == (4,)
    a = torch.randn(7, 4) * 3
    e4, e5 = m4.to_env(a), m5.to_env(torch.cat((torch.randn(7, 1), a), -1))
    assert e4.shape == (7, 5) and torch.all(e4[:, 0] == 0) and torch.allclose(e4[:, 1:], e5[:, 1:])   # live dims as before
    assert e4.min() >= 0 and e4.max() <= 1
    x = torch.randn(3, env.obs_dim)
    lp, ent, _ = m4.evaluate(x, a[:3])
    assert torch.allclose(ent, torch.distributions.Normal(0, m4.log_std.exp()).entropy().sum().expand(3))   # 4 dims only
    d = tempfile.mkdtemp()
    args = T.get_args(["--total_steps", "256", "--num_envs", "2", "--num_steps", "32", "--hidden", *map(str, HID),
                       "--eval_every", "1", "--eval_episodes", "2", "--out_dir", d])
    pol, log = T.train(args)
    assert args.mask_noop and pol.act_dim == 4 and np.isfinite(log["loss"]).all()
    path = os.path.join(args.out_dir, "final.pt")
    re = load_policy(path)
    assert re.act_dim == 4 and re.mask_noop and PolicyOpponent(re)(np.zeros((3, env.obs_dim), np.float32)).shape == (3, 5)
    old = T.get_args(["--no_mask_noop", "--total_steps", "128", "--num_envs", "2", "--num_steps", "32",
                      "--hidden", *map(str, HID), "--eval_every", "0", "--out_dir", d])
    pol5, _ = T.train(old)
    assert pol5.act_dim == 5
    try:                                                              # 5-dim checkpoint into a 4-dim run
        T.train(T.get_args(["--init_ckpt", os.path.join(old.out_dir, "final.pt"), "--total_steps", "64", "--num_envs", "2",
                            "--num_steps", "32", "--hidden", *map(str, HID), "--eval_every", "0", "--out_dir", d]))
        raise AssertionError("expected a dimension error")
    except ValueError as e:
        assert "--no_mask_noop" in str(e)
    r = exploit.nashconv(path, br_steps=64, br_seeds=1, episodes=4, br_args="--num_envs 2 --num_steps 32")
    assert load_policy(os.path.join(r["adv"]["selected"]["run_dir"], "final.pt")).act_dim == 4          # masked BR for masked ckpt
    r5 = exploit.nashconv(os.path.join(old.out_dir, "final.pt"), br_steps=64, br_seeds=1, episodes=4,
                          br_args="--num_envs 2 --num_steps 32")
    assert load_policy(os.path.join(r5["good"]["selected"]["run_dir"], "final.pt")).act_dim == 5       # old ckpt -> 5-dim BR
    import json, subprocess
    json.dump(vars(args), open(os.path.join(args.out_dir, "config.json"), "w"))
    out = subprocess.run([sys.executable, os.path.join(SRC, "visualize.py"), "--run_dir", args.out_dir, "--episodes", "1"],
                         capture_output=True, text=True, cwd=SRC)
    assert out.returncode == 0 and os.path.exists(os.path.join(args.out_dir, "rollout.gif")), out.stderr[-400:]
    print("PASS no_op masking (4-dim policy, env dim 0 == 0, entropy over 4 dims, old checkpoints, init mismatch rejected, visualize.py runs)")


if __name__ == "__main__":
    test_load_and_make_opponent(); test_play_match_matches_evaluate(); test_opponent_gets_its_own_role_obs()
    test_train_vs_frozen_ckpt(); test_nashconv_smoke(); test_crossplay()
    test_zero_sum_and_share_vel(); test_env_flags_flow_through()
    test_pool_opponent(); test_per_env_roles_rollout(); test_fsp_end_to_end(); test_mask_noop()
    print("\nALL OK")
