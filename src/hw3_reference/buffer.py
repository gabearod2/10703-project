# buffer.py
import torch

class Buffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.capacity = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        self.ptr = 0
        self.size_ = 0
        # preallocate tensors on device
        self.obs        = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs   = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions    = torch.zeros((self.capacity, self.act_dim), dtype=torch.float32, device=self.device)
        self.log_probs   = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.rewards    = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.dones      = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.values     = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.returns    = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.iteration  = torch.zeros((self.capacity,), dtype=torch.int32,   device=self.device)

    @property
    def size(self):
        return self.size_

    def add(self, obs, next_obs, action, log_probs=0.0, reward=0.0, done=0.0, value=0.0,
            advantage=0.0, curr_return=0.0, iteration=0):
        i = self.ptr
        self.obs[i]        = obs
        self.next_obs[i]   = next_obs
        self.actions[i]    = action
        self.log_probs[i]   = log_probs
        self.rewards[i]    = reward
        self.dones[i]      = done
        self.values[i]     = value
        self.advantages[i] = advantage
        self.returns[i]    = curr_return
        self.iteration[i]  = iteration

        # advance pointer in ring fashion
        self.ptr = (self.ptr + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)

    def add_batch(self, batch: dict):
        n = batch["obs"].shape[0]
        for j in range(n):
            self.add(
                obs=batch["obs"][j],
                next_obs=batch["next_obs"][j],
                action=batch["actions"][j],
                log_probs=batch["log_probs"][j],
                reward=batch["rewards"][j],
                done=batch["dones"][j],
                value=batch["values"][j],
                advantage=batch["advantages"][j],
                curr_return=batch["returns"][j],
                iteration=batch["iteration"][j]
            )

    def sample(self, num_samples: int | None = None, filter: dict | None = None):
        if self.size == 0:
            raise ValueError("The buffer is empty")

        # --- build mask if filter provided ---
        if filter is not None:
            mask = torch.ones(self.size, dtype=torch.bool, device=self.device)
            for key, val_list in filter.items():
                if not hasattr(self, key):
                    raise KeyError(f"Buffer has no field '{key}'")
                attr = getattr(self, key)[:self.size]
                val_tensor = torch.as_tensor(val_list, device=self.device, dtype=attr.dtype)
                mask &= torch.isin(attr, val_tensor)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        else:
            idx = torch.arange(self.size, device=self.device)

        if idx.numel() == 0:
            raise ValueError("No samples match the filter conditions")

        # --- subsample if requested ---
        if num_samples is not None and num_samples < idx.numel():
            perm = torch.randperm(idx.numel(), device=self.device)[:num_samples]
            idx = idx[perm]

        return dict(
            obs=self.obs[idx],
            next_obs=self.next_obs[idx],
            actions=self.actions[idx],
            log_probs=self.log_probs[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            values=self.values[idx],
            advantages=self.advantages[idx],
            returns=self.returns[idx],
            iteration=self.iteration[idx],
        )