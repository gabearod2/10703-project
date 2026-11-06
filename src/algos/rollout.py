# algos/rollout.py
"""Collect T steps from every env. The learner role(s) act with `policy`; when only one role learns,
the other role is played by `opponent` (a callable: obs [N, D] -> env actions [N, A])."""
import numpy as np
import torch

from envs.tag import ADV, GOOD


def collect_rollout(env, policy, obs_np, roles, opponent, T, gamma, device):
    """
    roles: [ADV, GOOD] for self-play (both share `policy`), or a single role that learns against `opponent`.
    Returns dict of [T, K, ...] tensors (K = num_envs * len(roles)) plus next_obs_np and finished-episode returns.
    Time-limit truncations are handled by adding gamma * V(final_obs) to the reward and cutting the GAE chain.
    """
    N, A, D = env.num_envs, env.act_dim, env.obs_dim
    R = len(roles)
    K = N * R
    opp_role = None if R == 2 else (GOOD if roles[0] == ADV else ADV)
    buf = {k: torch.zeros(shape, device=device) for k, shape in
           dict(obs=(T, K, D), act=(T, K, A), logp=(T, K), val=(T, K), rew=(T, K), done=(T, K)).items()}
    episodes = []
    obs = torch.as_tensor(obs_np[:, roles].reshape(K, D), device=device)

    for t in range(T):
        a, logp, v = policy.act(obs)
        buf["obs"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = obs, a, logp, v

        a_env = np.zeros((N, 2, A), dtype=np.float32)
        a_env[:, roles] = policy.to_env(a).cpu().numpy().reshape(N, R, A)
        if opponent is not None:
            a_env[:, opp_role] = opponent(obs_np[:, opp_role])
        res = env.step(a_env)

        r = torch.as_tensor(res["reward"][:, roles].reshape(K), device=device)
        done = res["terminated"] | res["truncated"]
        if res["truncated"].any():                       # time limit: bootstrap V(s_final) into the reward
            final = torch.as_tensor(res["final_obs"][:, roles].reshape(K, D), device=device)
            with torch.no_grad():
                v_final = policy.value(final)
            trunc = torch.as_tensor(np.repeat(res["truncated"], R), device=device)
            r = r + gamma * v_final * trunc.float()
        buf["rew"][t] = r
        buf["done"][t] = torch.as_tensor(np.repeat(done, R).astype(np.float32), device=device)

        obs_np = res["obs"]
        obs = torch.as_tensor(obs_np[:, roles].reshape(K, D), device=device)
        episodes += [e["ret"] for e in res["episodes"]]

    buf["next_obs_np"], buf["episodes"] = obs_np, episodes
    return buf
