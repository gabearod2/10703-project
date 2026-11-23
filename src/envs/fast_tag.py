# envs/fast_tag.py
"""
Batched NumPy re-implementation of the 1v1 MPE2 Simple Tag used in envs/tag.py.

Same API, observation layout, rewards and physics as TagVecEnv, but all N games are stepped with array ops instead of
a Python loop over PettingZoo envs/agents/entity pairs, so it is ~100x faster per env-step and the speed barely depends
on N. tests/test_fast_env.py checks step-by-step parity with MPE2 (starting from identical states).

Physics (mpe2 _mpe_utils/core.py World.step, dt 0.1, damping 0.25, mass 1, contact_force 100, contact_margin 1e-3):
    u      = accel * (right - left, up - down)
    F      = u + sum over colliding pairs of soft contact forces (agent-agent, agent-landmark; landmarks don't move)
    pos   += vel * dt;  vel = vel * (1 - damping) + F * dt;  |vel| clamped to max_speed
Rewards (simple_tag.py): chaser +10 / runner -10 while the bodies overlap; runner bound penalty per coordinate.

Initial states come from one Generator(seed) for all envs, so they differ from TagVecEnv's per-env PettingZoo seeds
(same distribution), but a given (seed, num_envs) always yields the same states, so matchups in play_match still share
initial states (common random numbers).
"""
import numpy as np

from envs.tag import ADV, GOOD, SYM_SIZE, SYM_ACCEL, SYM_MAX_SPEED

DT, DAMPING, CONTACT_FORCE, CONTACT_MARGIN, LANDMARK_SIZE = 0.1, 0.25, 1e2, 1e-3, 0.2


class FastTagVecEnv:
    def __init__(self, num_envs=8, max_cycles=25, num_obstacles=2, seed=0,
                 symmetric=True, size=SYM_SIZE, accel=SYM_ACCEL, max_speed=SYM_MAX_SPEED,
                 zero_sum=True, share_vel=True):
        self.symmetric, self.zero_sum, self.share_vel = symmetric, zero_sum, share_vel
        self.num_envs, self.max_cycles, self.num_obstacles = num_envs, max_cycles, num_obstacles
        if symmetric:
            self.size, self.accel, self.max_speed = (np.full(2, v, dtype=np.float64) for v in (size, accel, max_speed))
        else:                                                   # original MPE2: chaser .075/3/1, runner .05/4/1.3
            self.size, self.accel, self.max_speed = np.array([.075, .05]), np.array([3., 4.]), np.array([1., 1.3])
        L = num_obstacles
        self._raw_dims = [4 + 2 * L + 4, 4 + 2 * L + 2]          # chaser also sees the runner's velocity
        self.pad_dim = max(self._raw_dims)
        if share_vel:
            assert self._raw_dims[GOOD] + 2 <= self.pad_dim, "no padding slots left for the chaser velocity"
            self._vel_slot = slice(self._raw_dims[GOOD], self._raw_dims[GOOD] + 2)
        self.obs_dim = self.pad_dim + 2
        self.act_dim = 5
        self.act_low, self.act_high = np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((num_envs, 2, 2)); self.vel = np.zeros((num_envs, 2, 2))
        self.lm = np.zeros((num_envs, L, 2))
        self._t = np.zeros(num_envs, dtype=np.int64)
        self._ep_ret = np.zeros((num_envs, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    def set_state(self, pos, vel, landmarks):
        """Overwrite the physical state ([N,2,2], [N,2,2], [N,L,2]); used by the parity test."""
        self.pos[:], self.vel[:], self.lm[:] = pos, vel, landmarks
        return self._obs()

    def get_state(self, i=0):
        return dict(agents=[(self.pos[i, k].copy(), float(self.size[k])) for k in (ADV, GOOD)],
                    landmarks=[(self.lm[i, j].copy(), LANDMARK_SIZE) for j in range(self.num_obstacles)])

    def _reset_envs(self, idx):
        n = len(idx)
        self.pos[idx] = self.rng.uniform(-1, 1, (n, 2, 2))
        self.vel[idx] = 0.0
        self.lm[idx] = self.rng.uniform(-0.9, 0.9, (n, self.num_obstacles, 2))
        self._t[idx] = 0
        self._ep_ret[idx] = 0

    def _obs(self):
        N, L = self.num_envs, self.num_obstacles
        out = np.zeros((N, 2, self.obs_dim), dtype=np.float32)
        for r in (ADV, GOOD):
            me, other = self.pos[:, r], self.pos[:, 1 - r]
            parts = [self.vel[:, r], me, (self.lm - me[:, None]).reshape(N, 2 * L), other - me]
            if r == ADV:
                parts.append(self.vel[:, GOOD])                 # MPE2: only good agents' velocities are observed
            raw = np.concatenate(parts, axis=1)
            out[:, r, :raw.shape[1]] = raw
            out[:, r, self.pad_dim + r] = 1.0
        if self.share_vel:
            out[:, GOOD, self._vel_slot] = self.vel[:, ADV]
        return out

    def reset(self):
        self._reset_envs(np.arange(self.num_envs))
        return self._obs()

    @staticmethod
    def _contact(delta, dist_min):
        """MPE2 soft contact force on the first body of each pair (delta = p_a - p_b)."""
        dist = np.sqrt(np.sum(np.square(delta), axis=-1, keepdims=True))
        penetration = np.logaddexp(0, -(dist - dist_min) / CONTACT_MARGIN) * CONTACT_MARGIN
        return CONTACT_FORCE * delta / dist * penetration

    def step(self, actions):
        """Same contract as TagVecEnv.step: actions [N, 2, 5] in env units."""
        a = np.clip(actions, self.act_low, self.act_high).astype(np.float64)
        u = np.stack((a[..., 2] - a[..., 1], a[..., 4] - a[..., 3]), axis=-1) * self.accel[None, :, None]
        f_aa = self._contact(self.pos[:, ADV] - self.pos[:, GOOD], self.size[ADV] + self.size[GOOD])
        f_al = self._contact(self.pos[:, :, None] - self.lm[:, None], (self.size[:, None] + LANDMARK_SIZE)[None, :, :, None])
        force = u + f_al.sum(axis=2)
        force[:, ADV] += f_aa
        force[:, GOOD] -= f_aa

        self.pos += self.vel * DT
        self.vel = self.vel * (1 - DAMPING) + force * DT        # mass = 1
        speed = np.sqrt(np.sum(np.square(self.vel), axis=-1, keepdims=True))
        cap = self.max_speed[None, :, None]
        self.vel = np.where(speed > cap, self.vel / np.where(speed > 0, speed, 1) * cap, self.vel)

        caught = np.sqrt(np.sum(np.square(self.pos[:, ADV] - self.pos[:, GOOD]), axis=-1)) < self.size.sum()
        x = np.abs(self.pos[:, GOOD])
        bound = np.where(x < 0.9, 0.0, np.where(x < 1.0, (x - 0.9) * 10, np.minimum(np.exp(2 * x - 2), 10)))
        rew = np.zeros((self.num_envs, 2), dtype=np.float32)
        rew[:, GOOD] = -10.0 * caught - bound.sum(axis=1)
        rew[:, ADV] = -rew[:, GOOD] if self.zero_sum else 10.0 * caught

        self._t += 1
        self._ep_ret += rew
        final_obs = self._obs()
        trunc = self._t >= self.max_cycles
        term = np.zeros(self.num_envs, dtype=bool)
        episodes = []
        obs = final_obs
        if trunc.any():
            idx = np.flatnonzero(trunc)
            episodes = [{"ret": self._ep_ret[i].copy(), "len": int(self._t[i]), "env": int(i)} for i in idx]
            self._reset_envs(idx)
            obs = final_obs.copy()
            obs[idx] = self._obs()[idx]
        return {"obs": obs, "final_obs": final_obs, "reward": rew,
                "terminated": term, "truncated": trunc, "episodes": episodes}

    def close(self):
        pass


def make_vec_env(num_envs=8, max_cycles=25, seed=0, backend="fast", **kw):
    """backend="fast": FastTagVecEnv (default). backend="mpe": the reference MPE2/PettingZoo wrapper (envs/tag.py)."""
    if backend not in ("fast", "mpe"):
        raise ValueError(f"unknown env backend {backend!r}")
    from envs.tag import TagVecEnv
    return (FastTagVecEnv if backend == "fast" else TagVecEnv)(num_envs, max_cycles, seed=seed, **kw)
