# algos/ppo.py
"""PPO: GAE, optional policy clipping / adaptive KL penalty, and minibatch updates."""
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


def gaussian_kl(old_mean, old_std, new_dist):
    """KL(old || new) per observation, summed over the unclipped Gaussian action dimensions.

    Only the new distribution receives gradients; old parameters must describe the
    fixed rollout policy throughout an update.
    """
    old_dist = torch.distributions.Normal(old_mean.detach(), old_std.detach())
    return torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)


def adapt_kl_coef(coef, kl, target):
    """Schulman et al. (2017), section 4: coefficient for the next rollout update."""
    if kl < target / 1.5:
        return coef / 2.0
    if kl > target * 1.5:
        return coef * 2.0
    return coef


def ppo_loss(new_logp, old_logp, adv, new_v, old_v, ret, entropy,
             clip_coef, vf_coef, ent_coef, clip_vloss=True, *,
             clip_policy=True, kl=None, kl_coef=0.0):
    """Optional per-sample analytical KL adds a differentiable penalty to the loss."""
    ratio = (new_logp - old_logp).exp()
    if clip_policy:
        pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
    else:
        pg_loss = (-adv * ratio).mean()
    if clip_vloss:
        v_clip = old_v + torch.clamp(new_v - old_v, -clip_coef, clip_coef)
        v_loss = 0.5 * torch.max((new_v - ret) ** 2, (v_clip - ret) ** 2).mean()
    else:
        v_loss = 0.5 * ((new_v - ret) ** 2).mean()
    ent = entropy.mean()
    kl_penalty = kl_coef * kl.mean() if kl is not None else new_logp.new_zeros(())
    total = pg_loss + vf_coef * v_loss - ent_coef * ent + kl_penalty
    with torch.no_grad():
        approx_kl = ((ratio - 1) - (ratio.log())).mean()      # k3 estimator of KL(old||new)
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return total, dict(loss=total.item(), policy_loss=pg_loss.item(), value_loss=v_loss.item(),
                       entropy=ent.item(), kl=approx_kl.item(), clipfrac=clipfrac.item(),
                       kl_penalty=kl_penalty.item())


class PPO:
    """Owns the optimizer; update() runs the PPO epochs/minibatches on one flat batch."""

    def __init__(self, policy, args):
        self.policy, self.args = policy, args
        self.opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
        self.beta = args.kl_coef_init

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def update(self, batch):
        """Update on a fresh on-policy batch of obs, act, logp, val, adv, ret tensors.

        Optimization stats are minibatch means; kl_exact_final is measured under
        the final policy over the entire batch, and controls next update's beta.
        """
        a = self.args
        n = batch["obs"].shape[0]
        mb_size = n // a.num_minibatches
        # No optimizer step has occurred since collection. Freeze this distribution
        # for all epochs, including its learned standard deviation.
        with torch.no_grad():
            old_dist = self.policy.dist(batch["obs"])
            old_mean = old_dist.mean.detach().clone()
            old_std = old_dist.stddev.detach().clone()
        coef_used = self.beta if a.adaptive_kl else 0.0
        stats = []
        for _ in range(a.update_epochs):
            perm = torch.randperm(n, device=batch["obs"].device)
            for s in range(0, n, mb_size):
                idx = perm[s:s + mb_size]
                adv = batch["adv"][idx]
                if a.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                new_dist = self.policy.dist(batch["obs"][idx])
                new_logp = new_dist.log_prob(batch["act"][idx]).sum(-1)
                ent = new_dist.entropy().sum(-1)
                new_v = self.policy.value(batch["obs"][idx])
                kl = gaussian_kl(old_mean[idx], old_std[idx], new_dist) if a.adaptive_kl else None
                loss, st = ppo_loss(new_logp, batch["logp"][idx], adv, new_v, batch["val"][idx], batch["ret"][idx],
                                    ent, a.clip_coef, a.vf_coef, a.ent_coef, a.clip_vloss,
                                    clip_policy=a.clip_policy, kl=kl, kl_coef=coef_used)
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), a.max_grad_norm)
                self.opt.step()
                stats.append(st)
            if a.target_kl is not None and np.mean([s_["kl"] for s_ in stats[-a.num_minibatches:]]) > a.target_kl:
                break
        out = {k: float(np.mean([s_[k] for s_ in stats])) for k in stats[0]}
        with torch.no_grad():
            final_kl = gaussian_kl(old_mean, old_std, self.policy.dist(batch["obs"])).mean().item()
        if a.adaptive_kl:
            self.beta = adapt_kl_coef(self.beta, final_kl, a.kl_target)
        out.update(kl_exact_final=final_kl, kl_coef_used=coef_used,
                   kl_coef_next=self.beta if a.adaptive_kl else 0.0)
        var_y = batch["ret"].var()
        out["explained_var"] = float(1 - (batch["ret"] - batch["val"]).var() / (var_y + 1e-8)) if var_y > 0 else float("nan")
        return out
