# algos/rollout.py
"""Collect T steps from every env. The learner role(s) act with `policy`; when only one role per env learns,
the other role is played by `opponent` (a callable: obs [N, D] -> env actions [N, A])."""
import numpy as np
import torch

from envs.tag import ADV, GOOD


def learner_obs(obs_np, roles):
    """Observations of the learning agents, [K, D]. `roles` is one of
         [ADV, GOOD]        self-play: both roles of every env learn (K = 2N, env-major order)
         [role]             that role learns in every env (K = N)
         int array of len N the learner's role in each env (K = N), used by fictitious self-play."""
    N = obs_np.shape[0]
    if isinstance(roles, np.ndarray):
        return obs_np[np.arange(N), roles]
    return obs_np[:, roles].reshape(N * len(roles), -1)


def collect_rollout(env, policy, obs_np, roles, opponent, T, gamma, device):
    """
    roles: see `learner_obs`. With one learner per env, `opponent` plays the other role; if it has an
    `episode_end(done)` method it is told which envs just finished (a checkpoint pool re-draws its member there).
    Returns dict of [T, K, ...] tensors plus next_obs_np, finished-episode returns and their env indices.
    Time-limit truncations are handled by adding gamma * V(final_obs) to the reward and cutting the GAE chain.
    """
    N, A, D = env.num_envs, env.act_dim, env.obs_dim
    both = not isinstance(roles, np.ndarray) and len(roles) == 2
    ar = np.arange(N)
    lr = None if both else (roles if isinstance(roles, np.ndarray) else np.full(N, roles[0]))   # learner role per env
    R = 2 if both else 1
    K = N * R
    buf = {k: torch.zeros(shape, device=device) for k, shape in
           dict(obs=(T, K, D), act=(T, K, policy.act_dim), logp=(T, K), val=(T, K), rew=(T, K), done=(T, K)).items()}
    episodes, episode_env = [], []
    pick = (lambda x: x[:, roles].reshape(K, *x.shape[2:])) if both else (lambda x: x[ar, lr])
    obs = torch.as_tensor(pick(obs_np), device=device)

    for t in range(T):
        a, logp, v = policy.act(obs)
        buf["obs"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = obs, a, logp, v

        a_env = np.zeros((N, 2, A), dtype=np.float32)
        a_learner = policy.to_env(a).cpu().numpy()
        if both:
            a_env[:, roles] = a_learner.reshape(N, R, A)
        else:
            a_env[ar, lr] = a_learner
            a_env[ar, 1 - lr] = opponent(obs_np[ar, 1 - lr])
        res = env.step(a_env)

        r = torch.as_tensor(pick(res["reward"]), device=device)
        done = res["terminated"] | res["truncated"]
        if res["truncated"].any():                       # time limit: bootstrap V(s_final) into the reward
            final = torch.as_tensor(pick(res["final_obs"]), device=device)
            with torch.no_grad():
                v_final = policy.value(final)
            trunc = torch.as_tensor(np.repeat(res["truncated"], R), device=device)
            r = r + gamma * v_final * trunc.float()
        buf["rew"][t] = r
        buf["done"][t] = torch.as_tensor(np.repeat(done, R).astype(np.float32), device=device)
        if done.any() and hasattr(opponent, "episode_end"):
            opponent.episode_end(done)

        obs_np = res["obs"]
        obs = torch.as_tensor(pick(obs_np), device=device)
        episodes += [e["ret"] for e in res["episodes"]]
        episode_env += [e["env"] for e in res["episodes"]]

    buf["next_obs_np"], buf["episodes"], buf["episode_env"] = obs_np, episodes, episode_env
    return buf
