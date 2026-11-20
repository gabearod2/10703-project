# opponents.py
"""Fixed opponents: callables  obs [N, D] -> env actions [N, 5]. A frozen checkpoint is PolicyOpponent;
a uniform mixture over a pool of checkpoints (fictitious self-play's average policy) is PoolOpponent."""
import copy, glob, os
import numpy as np
import torch

from envs.tag import ADV, GOOD
from algos.networks import GaussianActorCritic


def load_policy(path, device="cpu"):
    """Rebuild a GaussianActorCritic from a checkpoint (.pt file, or a run dir holding final.pt).
    obs/act dims (incl. whether no_op is masked), hidden sizes and action bounds are all read off the state dict."""
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

    def reset_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def __call__(self, obs):
        dist = self.policy.dist(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        a = dist.mean
        if not self.deterministic:
            if not hasattr(self, "rng"):
                self.reset_rng(0)
            noise = torch.as_tensor(self.rng.standard_normal(tuple(a.shape)), dtype=a.dtype, device=a.device)
            a = a + dist.stddev * noise
        return self.policy.to_env(a).cpu().numpy()


class PoolOpponent:
    """Uniform mixture over a pool of frozen checkpoints: each env draws one member per EPISODE and keeps it until
    that episode ends. Drawing a member uniformly at the start of every episode is exactly the time-average of the
    members, i.e. the average policy of fictitious play, without having to distil it into one network.

    Training (fictitious self-play): `add(policy)` appends a frozen copy; collect_rollout calls `episode_end(done)`.
    Evaluation: `PoolOpponent.from_dir(<run>/pool)`; play_match gives each seat a copy and calls `reset_rng(seed)`, so a
    match is reproducible and the two seats of a mixture-vs-mixture match draw their members independently.
    """
    def __init__(self, seed=0, deterministic=False, device="cpu"):
        self.members, self.deterministic, self.device = [], deterministic, device
        self.rng, self.assign = np.random.default_rng(seed), None

    @classmethod
    def from_dir(cls, path, seed=0, deterministic=False, device="cpu"):
        files = pool_files(path)
        if not files:
            raise FileNotFoundError(f"no snap_*.pt checkpoints in {path}")
        pool = cls(seed, deterministic, device)
        pool.members = [load_policy(f, device) for f in files]
        return pool

    def add(self, policy):
        m = copy.deepcopy(policy).to(self.device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.members.append(m)

    def reset_rng(self, seed):
        """New member draw and action noise from `seed` (same protocol as PolicyOpponent.reset_rng)."""
        self.rng, self.assign = np.random.default_rng(seed), None

    reseed = reset_rng

    def episode_end(self, done):
        """Envs flagged in `done` start a new episode: draw a new member for each of them."""
        if self.assign is not None and done.any():
            self.assign[done] = self.rng.integers(len(self.members), size=int(done.sum()))

    @torch.no_grad()
    def __call__(self, obs):
        n = obs.shape[0]
        if self.assign is None or len(self.assign) != n:
            self.assign = self.rng.integers(len(self.members), size=n)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        out = np.zeros((n, self.members[0].low.numel()), dtype=np.float32)
        for k in np.unique(self.assign):                      # one batched forward pass per member in use
            idx = np.flatnonzero(self.assign == k)
            dist = self.members[k].dist(x[idx])
            a = dist.mean
            if not self.deterministic:                        # own generator, like PolicyOpponent: reproducible matches
                noise = torch.as_tensor(self.rng.standard_normal(tuple(a.shape)), dtype=a.dtype, device=a.device)
                a = a + dist.stddev * noise
            out[idx] = self.members[k].to_env(a).cpu().numpy()
        return out


def pool_files(path):
    """Snapshot files of a pool directory, oldest first."""
    return sorted(glob.glob(os.path.join(path, "snap_*.pt")))


def is_pool(path):
    return os.path.isdir(path) and bool(pool_files(path))


def make_opponent(name, role, act_dim, seed=0, ckpt=None, deterministic=False, device="cpu"):
    if name == "pool":
        return PoolOpponent.from_dir(ckpt, seed, deterministic, device)
    if name == "ckpt":
        return PolicyOpponent(load_policy(ckpt, device), deterministic, device)
    return RandomOpponent(act_dim, seed) if name == "random" else ScriptedOpponent(role)
