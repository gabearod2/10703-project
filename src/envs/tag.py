# envs/tag.py
"""
Vectorised 1v1 Simple Tag (MPE2) with continuous actions, for two-player self-play.

Roles (fixed order everywhere in the code base):
    ADV  = 0  -> "adversary_0"  (chaser,  +10 per collision with the runner)
    GOOD = 1  -> "agent_0"      (runner,  -10 per collision, plus out-of-bounds penalty)

Observations of the two roles have different sizes in MPE2 (adversary 12, good 10 with
1 landmark pair), so each is zero-padded to a common size and a 2-d one-hot role id is
appended. This lets ONE network be shared by both players (as in the Rudolph et al. PPO
self-play setup, "we learn a single model for both players").

MPE2 continuous action = Box(0, 1, (5,)) = [no_op, left, right, down, up]; the physical
force is (right - left, up - down), so component 0 is ignored by the dynamics.
"""
import numpy as np
from mpe2 import simple_tag_v3

ADV, GOOD = 0, 1
SYM_SIZE, SYM_ACCEL, SYM_MAX_SPEED = 0.0625, 3.5, 1.15     # midpoints of adversary (.075/3/1) and good (.05/4/1.3)
ROLE_NAMES = ["adversary_0", "agent_0"]


class TagVecEnv:
    """N independent 1v1 simple_tag games stepped in a Python loop (CPU)."""

    def __init__(self, num_envs=8, max_cycles=25, num_obstacles=2, seed=0,
                 symmetric=True, size=SYM_SIZE, accel=SYM_ACCEL, max_speed=SYM_MAX_SPEED,
                 zero_sum=True, share_vel=True):
        """
        symmetric=True : both roles get identical body/dynamics (size, accel, max_speed) -> the only asymmetry
                         left is the reward (chaser +10 on contact, runner -10 and an out-of-bounds penalty).
        symmetric=False: original MPE2 Simple Tag (chaser: size .075/accel 3/speed 1; runner: .05/4/1.3).
        Defaults are the midpoints of the two original roles.
        zero_sum=True  (default): chaser reward := -(runner reward), so the runner's out-of-bounds penalty is also the chaser's gain.
                         The game is then exactly zero-sum (u_adv + u_good = 0); the runner's own reward is unchanged.
        share_vel=True (default): the runner also observes the chaser's (absolute) velocity, as the chaser already observes the
                         runner's. Occupies the zero-padded slots, so obs_dim is unchanged and old checkpoints still load.
        """
        self.symmetric = symmetric
        self.zero_sum = zero_sum
        self.share_vel = share_vel
        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self.envs = [
            simple_tag_v3.parallel_env(
                num_good=1, num_adversaries=1, num_obstacles=num_obstacles,
                max_cycles=max_cycles, continuous_actions=True,
            )
            for _ in range(num_envs)
        ]
        if symmetric:
            for env in self.envs:
                for ag in self.world(env).agents:
                    ag.size, ag.accel, ag.max_speed = size, accel, max_speed
        e = self.envs[0]
        e.reset(seed=0)
        self._raw_dims = [int(np.prod(e.observation_space(n).shape)) for n in ROLE_NAMES]
        self.pad_dim = max(self._raw_dims)
        if share_vel:
            assert self._raw_dims[GOOD] + 2 <= self.pad_dim, "no padding slots left for the chaser velocity"
            self._vel_slot = slice(self._raw_dims[GOOD], self._raw_dims[GOOD] + 2)
        self.obs_dim = self.pad_dim + 2                    # + one-hot role
        self.act_dim = int(np.prod(e.action_space(ROLE_NAMES[0]).shape))
        self.act_low = e.action_space(ROLE_NAMES[0]).low.astype(np.float32)
        self.act_high = e.action_space(ROLE_NAMES[0]).high.astype(np.float32)
        self.seed = seed
        self._started = False
        self._ep_ret = np.zeros((num_envs, 2), dtype=np.float64)
        self._ep_len = np.zeros(num_envs, dtype=np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def world(env):
        """underlying MPE2 World of a parallel env (agents[0]=adversary, agents[1]=good, plus landmarks)."""
        return env.aec_env.unwrapped.world

    def get_state(self, i=0):
        """Positions/sizes of env i for plotting: dict(agents=[(pos, size)]*2, landmarks=[(pos, size)])."""
        w = self.world(self.envs[i])
        return dict(agents=[(a.state.p_pos.copy(), a.size) for a in w.agents],
                    landmarks=[(l.state.p_pos.copy(), l.size) for l in w.landmarks])

    def _encode(self, obs_dict, i):
        out = np.zeros((2, self.obs_dim), dtype=np.float32)
        for r, name in enumerate(ROLE_NAMES):
            o = np.asarray(obs_dict[name], dtype=np.float32).reshape(-1)
            out[r, : o.shape[0]] = o
            out[r, self.pad_dim + r] = 1.0
        if self.share_vel:                                  # chaser's velocity, same convention as the chaser's view of the runner
            out[GOOD, self._vel_slot] = self.world(self.envs[i]).agents[ADV].state.p_vel
        return out

    def reset(self):
        obs = np.zeros((self.num_envs, 2, self.obs_dim), dtype=np.float32)
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=self.seed + i) if not self._started else e.reset()
            obs[i] = self._encode(o, i)
        self._started = True
        self._ep_ret[:] = 0
        self._ep_len[:] = 0
        return obs

    def step(self, actions):
        """
        actions: [N, 2, A] in env units (will be clipped to [low, high]).
        Returns dict:
          obs        [N,2,D]  observation to act on next (already reset where an episode ended)
          final_obs  [N,2,D]  true successor obs (== obs unless the episode ended)
          reward     [N,2]
          terminated [N] bool, truncated [N] bool
          episodes   list of dicts {ret: [adv, good], len} for episodes finishing this step
        """
        actions = np.clip(actions, self.act_low, self.act_high).astype(np.float32)
        N = self.num_envs
        obs = np.zeros((N, 2, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros_like(obs)
        rew = np.zeros((N, 2), dtype=np.float32)
        term = np.zeros(N, dtype=bool)
        trunc = np.zeros(N, dtype=bool)
        episodes = []
        for i, e in enumerate(self.envs):
            o, r, te, tr, _ = e.step({n: actions[i, k] for k, n in enumerate(ROLE_NAMES)})
            rew[i] = [r[ROLE_NAMES[0]], r[ROLE_NAMES[1]]]
            if self.zero_sum:
                rew[i, ADV] = -rew[i, GOOD]
            term[i] = any(te.values())
            trunc[i] = any(tr.values()) and not term[i]
            final_obs[i] = self._encode(o, i)
            self._ep_ret[i] += rew[i]
            self._ep_len[i] += 1
            if term[i] or trunc[i]:
                episodes.append({"ret": self._ep_ret[i].copy(), "len": int(self._ep_len[i])})
                self._ep_ret[i] = 0
                self._ep_len[i] = 0
                o2, _ = e.reset()
                obs[i] = self._encode(o2, i)
            else:
                obs[i] = final_obs[i]
        return {"obs": obs, "final_obs": final_obs, "reward": rew,
                "terminated": term, "truncated": trunc, "episodes": episodes}

    def close(self):
        for e in self.envs:
            e.close()
