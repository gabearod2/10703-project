# utils/evaluate.py
"""Evaluate a policy against itself and against fixed random / scripted opponents."""
import numpy as np
import torch

from envs.tag import TagVecEnv, ADV, GOOD
from opponents import make_opponent

# (controller of the chaser, controller of the runner)
SETTINGS = {"self": ("policy", "policy"),
            "adv_vs_random": ("policy", "random"), "good_vs_random": ("random", "policy"),
            "adv_vs_flee": ("policy", "scripted"), "good_vs_chase": ("scripted", "policy")}


@torch.no_grad()
def evaluate(policy, num_episodes=50, max_cycles=25, seed=12345, device="cpu", symmetric=True, **env_kw):
    """Mean return of each role per setting (deterministic policy actions). Keys: <setting>_adv / <setting>_good."""
    out = {}
    for name, controllers in SETTINGS.items():
        n = num_episodes
        env = TagVecEnv(num_envs=n, max_cycles=max_cycles, seed=seed, symmetric=symmetric, **env_kw)
        try:
            opps = {role: make_opponent(c, role, env.act_dim, seed) for role, c in zip((ADV, GOOD), controllers) if c != "policy"}
            obs = env.reset()
            total = np.zeros((n, 2))
            for _ in range(max_cycles):
                a_env = np.zeros((n, 2, env.act_dim), dtype=np.float32)
                for role in (ADV, GOOD):
                    if role in opps:
                        a_env[:, role] = opps[role](obs[:, role])
                    else:
                        a, _, _ = policy.act(torch.as_tensor(obs[:, role], device=device), deterministic=True)
                        a_env[:, role] = policy.to_env(a).cpu().numpy()
                res = env.step(a_env)
                total += res["reward"]
                obs = res["obs"]
        finally:
            env.close()
        out[name + "_adv"], out[name + "_good"] = float(total[:, ADV].mean()), float(total[:, GOOD].mean())
    return out


@torch.no_grad()
def play_match(adv_ctrl, good_ctrl, num_episodes=100, max_cycles=25, seed=12345,
               symmetric=True, return_episodes=False, **env_kw):
    """Mean return when `adv_ctrl` plays the chaser and `good_ctrl` the runner (seeded stochastic matchup).

    Controllers are callables: encoded obs [N, D] -> env actions [N, A] (PolicyOpponent, PoolOpponent, scripted, random...).
    Env seeds are fixed by `seed`, so different matchups see the same initial states (common random numbers). Each seat
    plays a copy of its controller with its own action seed, so results do not depend on the global RNG state and one
    object may sit in both seats (a PoolOpponent then draws its members independently per seat).
    `env_kw`: extra TagVecEnv options (zero_sum, share_vel), which must match how the controllers were trained.
    Returns {"adv", "good", "adv_se", "good_se"}: mean and standard error over episodes of the episode return;
    with `return_episodes`, also "returns" ([episode][role]) for paired deviation estimates.
    """
    import copy
    if num_episodes < 1 or max_cycles < 1:
        raise ValueError("Episode count and horizon must be positive")
    controllers = [copy.copy(adv_ctrl), copy.copy(good_ctrl)]
    for role, ctrl in enumerate(controllers):
        action_seed = int(np.random.SeedSequence([seed, role]).generate_state(1)[0])
        if hasattr(ctrl, "reset_rng"):
            ctrl.reset_rng(action_seed)
        elif hasattr(ctrl, "rng"):
            ctrl.rng = np.random.default_rng(action_seed)
    env = TagVecEnv(num_episodes, max_cycles, seed=seed, symmetric=symmetric, **env_kw)
    try:
        obs, total = env.reset(), np.zeros((num_episodes, 2))
        active = np.ones(num_episodes, dtype=bool)
        for _ in range(max_cycles):
            actions = np.stack([controllers[r](obs[:, r]) for r in (ADV, GOOD)], axis=1)
            if not np.isfinite(actions).all():
                raise FloatingPointError("Non-finite evaluation actions")
            res = env.step(actions)
            total += res["reward"] * active[:, None]
            active &= ~(res["terminated"] | res["truncated"])
            obs = res["obs"]
        if not np.isfinite(total).all():
            raise FloatingPointError("Non-finite evaluation returns")
    finally:
        env.close()
    se = total.std(axis=0, ddof=1) / np.sqrt(num_episodes) if num_episodes > 1 else np.zeros(2)
    result = dict(adv=float(total[:, ADV].mean()), good=float(total[:, GOOD].mean()),
                  adv_se=float(se[ADV]), good_se=float(se[GOOD]))
    if return_episodes:
        result["returns"] = total.tolist()
    return result
