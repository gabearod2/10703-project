# algos/ppo.py
"""PPO: GAE + clipped loss (same maths as HW3 ppo_agent.py) and the minibatch update on one batch."""
import numpy as np
import torch
import torch.nn as nn


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    rewards, values, dones: [T, K] (done[t]=1 means transition t ended its episode);
    last_value: [K] = V(s_T). Same recursion as HW3 `_compute_gae`, vectorised over K columns.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    next_value, next_adv = last_value, torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nt = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nt - values[t]
        next_adv = delta + gamma * lam * nt * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values


def ppo_loss(new_logp, old_logp, adv, new_v, old_v, ret, entropy,
             clip_coef, vf_coef, ent_coef, clip_vloss=True):
    ratio = (new_logp - old_logp).exp()
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
    if clip_vloss:
        v_clip = old_v + torch.clamp(new_v - old_v, -clip_coef, clip_coef)
        v_loss = 0.5 * torch.max((new_v - ret) ** 2, (v_clip - ret) ** 2).mean()
    else:
        v_loss = 0.5 * ((new_v - ret) ** 2).mean()
    ent = entropy.mean()
    total = pg_loss + vf_coef * v_loss - ent_coef * ent
    with torch.no_grad():
        approx_kl = ((ratio - 1) - (ratio.log())).mean()      # k3 estimator of KL(old||new)
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return total, dict(loss=total.item(), policy_loss=pg_loss.item(), value_loss=v_loss.item(),
                       entropy=ent.item(), kl=approx_kl.item(), clipfrac=clipfrac.item())


class PPO:
    """Owns the optimizer; update() runs the PPO epochs/minibatches on one flat batch."""

    def __init__(self, policy, args):
        self.policy, self.args = policy, args
        self.opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def update(self, batch):
        """batch: flat tensors obs, act, logp, val, adv, ret. Returns stats averaged over minibatches."""
        a = self.args
        n = batch["obs"].shape[0]
        mb_size = n // a.num_minibatches
        stats = []
        for _ in range(a.update_epochs):
            perm = torch.randperm(n, device=batch["obs"].device)
            for s in range(0, n, mb_size):
                idx = perm[s:s + mb_size]
                adv = batch["adv"][idx]
                if a.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                new_logp, ent, new_v = self.policy.evaluate(batch["obs"][idx], batch["act"][idx])
                loss, st = ppo_loss(new_logp, batch["logp"][idx], adv, new_v, batch["val"][idx], batch["ret"][idx],
                                    ent, a.clip_coef, a.vf_coef, a.ent_coef, a.clip_vloss)
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), a.max_grad_norm)
                self.opt.step()
                stats.append(st)
            if a.target_kl is not None and np.mean([s_["kl"] for s_ in stats[-a.num_minibatches:]]) > a.target_kl:
                break
        out = {k: float(np.mean([s_[k] for s_ in stats])) for k in stats[0]}
        var_y = batch["ret"].var()
        out["explained_var"] = float(1 - (batch["ret"] - batch["val"]).var() / (var_y + 1e-8)) if var_y > 0 else float("nan")
        return out
