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
def evaluate(policy, num_episodes=50, max_cycles=25, seed=12345, device="cpu", symmetric=True):
    """Mean return of each role per setting (deterministic policy actions). Keys: <setting>_adv / <setting>_good."""
    out = {}
    for name, controllers in SETTINGS.items():
        n = num_episodes
        env = TagVecEnv(num_envs=n, max_cycles=max_cycles, seed=seed, symmetric=symmetric)
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
