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
        env.close()
        out[name + "_adv"], out[name + "_good"] = float(total[:, ADV].mean()), float(total[:, GOOD].mean())
    return out


@torch.no_grad()
def play_match(adv_ctrl, good_ctrl, num_episodes=100, max_cycles=25, seed=12345, symmetric=True, **env_kw):
    """Mean return when `adv_ctrl` plays the chaser and `good_ctrl` the runner.

    Controllers are callables: encoded obs [N, D] -> env actions [N, A] (PolicyOpponent, scripted, random...).
    Env seeds are fixed by `seed`, so different matchups see the same initial states (common random numbers).
    `env_kw`: extra TagVecEnv options (zero_sum, share_vel), which must match how the controllers were trained.
    Returns {"adv", "good", "adv_se", "good_se"}: mean and standard error over episodes of the episode return.
    """
    n = num_episodes
    env = TagVecEnv(num_envs=n, max_cycles=max_cycles, seed=seed, symmetric=symmetric, **env_kw)
    obs, total = env.reset(), np.zeros((n, 2))
    for _ in range(max_cycles):
        a_env = np.zeros((n, 2, env.act_dim), dtype=np.float32)
        a_env[:, ADV], a_env[:, GOOD] = adv_ctrl(obs[:, ADV]), good_ctrl(obs[:, GOOD])
        res = env.step(a_env)
        total += res["reward"]
        obs = res["obs"]
    env.close()
    se = total.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(2)
    return {"adv": float(total[:, ADV].mean()), "good": float(total[:, GOOD].mean()),
            "adv_se": float(se[ADV]), "good_se": float(se[GOOD])}
