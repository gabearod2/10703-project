# policies.py
import torch
import torch.nn as nn
from typing import Sequence, Tuple

# --------------------------- helpers ---------------------------

def mlp(sizes: Sequence[int], activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    """
    Simple MLP: sizes like [in, h1, h2, ..., out].
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.zeros_(m.bias)
    return seq


# -------------------- shared tanh-squashed distribution --------------------

class _TanhDiagGaussian:
    """
    Lightweight distribution wrapper for a diagonal Gaussian in pre-tanh space.

    Exposes:
      - sample()    : non-reparameterized sample (good for PPO env stepping)
      - rsample()   : reparameterized sample (required for SAC actor update)
      - log_prob(a) : log π(a|s) with tanh + scale correction; returns [B]
      - entropy()   : entropy of base Gaussian (sum over dims); returns [B]
      - mean_action : deterministic action tanh(μ) mapped to env bounds
    """
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor,
                 scale: torch.Tensor, bias: torch.Tensor):
        """
        Args:
            mu:      [B, A]
            log_std: [B, A] or [A]
            scale:   [A]  (maps [-1,1] to env bounds)
            bias:    [A]
        """
        self.mu = mu
        self.log_std = log_std
        self.std = log_std.exp()
        self.scale = scale
        self.bias = bias
        self._base = torch.distributions.Normal(self.mu, self.std)  # broadcast over [B, A]

    # --- sampling ---

    def sample(self) -> torch.Tensor:
        """Non-reparameterized sample (no grads through noise). Returns [B, A]."""
        u = self._base.sample()
        return torch.tanh(u) * self.scale + self.bias

    def rsample(self) -> torch.Tensor:
        """Reparameterized sample (pathwise grads). Returns [B, A]."""
        u = self._base.rsample()
        return torch.tanh(u) * self.scale + self.bias

    # --- densities ---

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        """
        Log π(a|s) with tanh change-of-variables and affine logdet.
        Accepts actions in env bounds; returns [B].
        """
        if a.ndim == 1:  # [A] -> [1, A]
            a = a.unsqueeze(0)

        # Map action back to pre-tanh space: y in (-1,1), u = atanh(y)
        y = (a - self.bias) / (self.scale + 1e-8)
        y = torch.clamp(y, -0.999999, 0.999999)
        u = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

        logp_u = self._base.log_prob(u).sum(dim=-1)   # sum over action dims

        # Jacobian of tanh: diag(1 - tanh(u)^2)
        correction = torch.log(1 - torch.tanh(u).pow(2) + 1e-8).sum(dim=-1)

        # Affine scale (bias adds no volume)
        scale_logdet = torch.log(self.scale + 1e-8).sum(dim=-1)

        return logp_u - correction - scale_logdet

    def entropy(self) -> torch.Tensor:
        """
        Proxy entropy: entropy of the base Gaussian (sum over dims). Returns [B].
        (True tanh entropy has no simple closed form; this proxy is standard in PPO.)
        """
        return self._base.entropy().sum(dim=-1)

    # --- eval convenience ---

    @property
    def mean_action(self) -> torch.Tensor:
        """Deterministic action for evaluation, [B, A]."""
        return torch.tanh(self.mu) * self.scale + self.bias


# ------------------------------- Actor -------------------------------

class Actor(nn.Module):
    """
    Unified tanh-squashed Gaussian policy. Works for both SAC and TD3.

    Args:
      obs_dim, act_dim: dimensions
      act_low, act_high: per-dim bounds (array-like)
      hidden: MLP hidden sizes
      state_independent_std: if True, use a learned global log_std vector;
                             if False, predict log_std with a head
      log_std_bounds: clamp range for numerical stability
      body_activation: nonlinearity for MLP trunk
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 act_low,
                 act_high,
                 hidden: Tuple[int, ...] = (256, 256),
                 state_independent_std: bool = False,
                 log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
                 body_activation = nn.ReLU):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds
        self.state_independent_std = state_independent_std

        # shared trunk
        self.body = mlp([obs_dim, *hidden], activation=body_activation)

        # mean head
        self.mu = nn.Linear(hidden[-1], act_dim)

        # std parameterization
        if state_independent_std:
            self.log_std_param = nn.Parameter(torch.zeros(act_dim))
        else:
            self.log_std_head = nn.Linear(hidden[-1], act_dim)

        # action bounds
        low_t  = torch.as_tensor(act_low, dtype=torch.float32)
        high_t = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (high_t - low_t) / 2.0)
        self.register_buffer("bias",  (high_t + low_t) / 2.0)

        # init heads
        nn.init.orthogonal_(self.mu.weight, 1.0); nn.init.zeros_(self.mu.bias)
        if not state_independent_std:
            nn.init.orthogonal_(self.log_std_head.weight, 1.0); nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs: torch.Tensor) -> _TanhDiagGaussian:
        """
        Returns a distribution-like object: _TanhDiagGaussian
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        feats = self.body(obs)
        mu = self.mu(feats)
        if self.state_independent_std:
            log_std = self.log_std_param.expand_as(mu)
        else:
            log_std = self.log_std_head(feats)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return _TanhDiagGaussian(mu, log_std, self.scale, self.bias)


# ------------------------------- Critic -------------------------------

class Critic(nn.Module):
    """
    Q-network: predicts Q(s, a). Used by off-policy algorithms like SAC/TD3.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1], activation=nn.ReLU)
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1: obs = obs.unsqueeze(0)
        if act.ndim == 1: act = act.unsqueeze(0)
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)


# ----------------------------- ActorCritic -----------------------------

class ActorCritic(nn.Module):
    """
    PPO-style module with an actor head and a value head

    Forward:
      obs -> (dist, value)
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 act_low,
                 act_high,
                 hidden: Tuple[int, ...] = (64, 64),
                 state_independent_std: bool = True,
                 body_activation = nn.Tanh):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # shared trunk
        self.body = mlp([obs_dim, *hidden], activation=body_activation)

        # actor heads (same as `Actor`)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.state_independent_std = state_independent_std
        if state_independent_std:
            self.log_std_param = nn.Parameter(torch.full((act_dim,), -0.5))
        else:
            self.log_std_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_min, self.log_std_max = (-20.0, 2.0)

        # value head
        self.v = nn.Linear(hidden[-1], 1)

        # action bounds
        low_t  = torch.as_tensor(act_low, dtype=torch.float32)
        high_t = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("scale", (high_t - low_t) / 2.0)
        self.register_buffer("bias",  (high_t + low_t) / 2.0)

        # # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0); nn.init.zeros_(m.bias)
                
    def forward(self, obs: torch.Tensor):
        """
        Returns:
            dist: _TanhDiagGaussian
            value: Tensor[B]
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        feats = self.body(obs)

        # actor bits
        mu = self.mu(feats)
        if self.state_independent_std:
            log_std = self.log_std_param.expand_as(mu)
        else:
            log_std = self.log_std_head(feats)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = _TanhDiagGaussian(mu, log_std, self.scale, self.bias)

        # value
        value = self.v(feats).squeeze(-1)
        return dist, value