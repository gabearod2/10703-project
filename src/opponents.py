# opponents.py
"""Fixed opponents: callables  obs [N, D] -> env actions [N, 5]. (A frozen checkpoint or a PSRO mixture
would be one more class with the same __call__.)"""
import numpy as np

from envs.tag import ADV, GOOD


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


def make_opponent(name, role, act_dim, seed=0):
    return RandomOpponent(act_dim, seed) if name == "random" else ScriptedOpponent(role)
