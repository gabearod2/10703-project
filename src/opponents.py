# opponents.py
"""Fixed opponents: callables  obs [N, D] -> env actions [N, 5]. A frozen checkpoint is PolicyOpponent;
a PSRO mixture would be one more class with the same __call__."""
import os
import numpy as np
import torch

from envs.tag import ADV, GOOD
from algos.networks import GaussianActorCritic


def load_policy(path, device="cpu"):
    """Rebuild a GaussianActorCritic from a checkpoint (.pt file, or a run dir holding final.pt).
    obs/act dims, hidden sizes and action bounds are all read off the state dict, so no config is needed."""
    if os.path.isdir(path):
        path = os.path.join(path, "final.pt")
    sd = torch.load(path, map_location="cpu")
    layers = sorted((int(k.split(".")[1]), k) for k in sd if k.startswith("actor_mean.") and k.endswith(".weight"))
    ws = [sd[k] for _, k in layers]
    policy = GaussianActorCritic(ws[0].shape[1], ws[-1].shape[0], sd["low"], sd["high"],
                                 tuple(w.shape[0] for w in ws[:-1]))
    policy.load_state_dict(sd)
    return policy.to(device).eval()


def scripted_action(role, obs_raw):
    """Greedy heuristic in env units. obs layout (both roles): [vel2, pos2, landmarks4, other_rel_pos2, ...]."""
    rel = obs_raw[:, 8:10]                                    # other agent's position relative to me
    d = rel if role == ADV else -rel                          # chase / flee
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    if role == GOOD:                                          # runner also steers back in-bounds (else it eats the exit penalty)
        pos = obs_raw[:, 2:4]
        d = d - 4.0 * pos * np.clip(np.abs(pos) - 0.6, 0, None)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    a = np.zeros((obs_raw.shape[0], 5), dtype=np.float32)
    a[:, 1], a[:, 2] = np.maximum(-d[:, 0], 0), np.maximum(d[:, 0], 0)   # left, right
    a[:, 3], a[:, 4] = np.maximum(-d[:, 1], 0), np.maximum(d[:, 1], 0)   # down, up
    return a


class RandomOpponent:
    def __init__(self, act_dim, seed=0):
        self.act_dim, self.rng = act_dim, np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(0, 1, (obs.shape[0], self.act_dim))


class ScriptedOpponent:
    """Greedy chase (if it plays ADV) or flee-and-stay-in-bounds (if it plays GOOD)."""
    def __init__(self, role):
        self.role = role

    def __call__(self, obs):
        return scripted_action(self.role, obs)


class PolicyOpponent:
    """A frozen GaussianActorCritic. The observation it is given already carries the role one-hot,
    so the same object can play either role. deterministic=False samples from the Gaussian (the policy
    as trained, which is what exploitability is defined for); True plays the mean action."""
    def __init__(self, policy, deterministic=False, device="cpu"):
        self.policy, self.deterministic, self.device = policy.to(device).eval(), deterministic, device
        for p in self.policy.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, obs):
        a, _, _ = self.policy.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device), self.deterministic)
        return self.policy.to_env(a).cpu().numpy()


def make_opponent(name, role, act_dim, seed=0, ckpt=None, deterministic=False, device="cpu"):
    if name == "ckpt":
        return PolicyOpponent(load_policy(ckpt, device), deterministic, device)
    return RandomOpponent(act_dim, seed) if name == "random" else ScriptedOpponent(role)
