# algos/networks.py
"""Gaussian actor-critic: separate actor / critic MLPs (3x512 tanh), state-independent log-std."""
import numpy as np
import torch
import torch.nn as nn


def _layer(i, o, std=np.sqrt(2.0)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


def _mlp(i, hidden, o, out_std):
    layers, d = [], i
    for h in hidden:
        layers += [_layer(d, h), nn.Tanh()]
        d = h
    layers.append(_layer(d, o, out_std))
    return nn.Sequential(*layers)


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(512, 512, 512), init_log_std=-0.5):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.actor_mean = _mlp(obs_dim, hidden, act_dim, 0.01)
        self.critic = _mlp(obs_dim, hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        low, high = torch.as_tensor(act_low, dtype=torch.float32), torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def dist(self, obs):
        mu = self.actor_mean(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def to_env(self, a):
        """normalised action ([-1,1], unclipped Gaussian sample) -> env action in [low, high]."""
        return torch.clamp(self.low + 0.5 * (a + 1.0) * (self.high - self.low), self.low, self.high)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def evaluate(self, obs, a):
        d = self.dist(obs)
        return d.log_prob(a).sum(-1), d.entropy().sum(-1), self.value(obs)
