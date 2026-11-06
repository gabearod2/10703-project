# sac_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from buffer import Buffer
from policies import Actor, Critic


class SACAgent:
    """
    Soft Actor-Critic agent that matches PPO's interface pattern.
    Simplified to remove unnecessary abstraction layers.
    """
    
    def __init__(self, env_info, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, 
                 batch_size=128, update_every=1, buffer_size=100000, 
                 warmup_steps=5000, utd_ratio=1, device="cpu"):
        self.device = torch.device(device)
        
        # Environment info
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = torch.as_tensor(env_info["act_low"], dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(env_info["act_high"], dtype=torch.float32, device=self.device)
        
        # SAC hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.utd_ratio = utd_ratio
        
        # ================== Problem 3.1.1: SAC initialization ==================
        ### BEGIN STUDENT SOLUTION - 3.1.1 ###
        self.actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_low=self.act_low,
            act_high=self.act_high,
            hidden =(64, 64),
        ).to(self.device)

        # critic networks
        self.critic1 = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=(64,64),
        ).to(self.device)
        self.critic2 = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=(64,64),
        ).to(self.device)
        self.critic1_target = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=(64,64),
        )
        self.critic2_target = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=(64,64),
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_target.to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_target.to(self.device)
        ### END STUDENT SOLUTION  -  3.1.1 ###
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        self._buffer = Buffer(
            size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=device
        )
        
        # Training state
        self.total_steps = 0
    
    def act(self, obs):
        """Return action info dict matching PPO's interface"""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.actor(obs_t)
            action = dist.sample()
            
            # ---------------- Problem 3.5: Deterministic Action ----------------
            ### BEGIN STUDENT SOLUTION - 3.5 ###
            # action = dist.mean_action
            ### END STUDENT SOLUTION  -  3.5 ###
            # Clamp to environment bounds
            action = torch.clamp(action, self.act_low, self.act_high)
            
            return {
                "action": action.squeeze(0).cpu().numpy()
            }
    
    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        Add transition to buffer and perform updates when ready.
        Matches PPO's step interface.
        """
        # Add to buffer using existing Buffer.add method
        obs_t = torch.as_tensor(transition["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(transition["next_obs"], dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(transition["action"], dtype=torch.float32, device=self.device)
        
        self._buffer.add(
            obs=obs_t,
            next_obs=next_obs_t,
            action=action_t,
            log_probs=0.0,  # Not used in SAC
            reward=float(transition["reward"]),
            done=float(transition["done"] and not transition["truncated"]),
            value=0.0,  # Not used in SAC
            advantage=0.0,  # Not used in SAC
            curr_return=0.0,  # Not used in SAC
            iteration=0  # Not used in SAC
        )
        
        self.total_steps += 1
        
        # Check if we should update
        # ---------------- Problem 3.2: Environment Step ----------------
        ### BEGIN STUDENT SOLUTION - 3.2 ###
        if self.warmup_steps > self._buffer.size or self.batch_size > self._buffer.size:
            return {}
        if self.total_steps % self.update_every != 0:
            return {}
        ### END STUDENT SOLUTION  -  3.2 ###
        
        # Perform SAC updates
        return self._perform_update()
    
    def _perform_update(self) -> Dict[str, float]:
        """Perform SAC updates and return stats"""
        all_stats = []
        
        # Perform multiple updates based on UTD ratio
        num_updates = max(1, self.utd_ratio)
        
        for _ in range(num_updates):
            # Sample batch from buffer
            batch = self._buffer.sample(self.batch_size)
            
            # Perform one SAC update step
            stats = self._sac_update_step(batch)
            all_stats.append(stats)
        
        # Average stats across updates
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
    
    def _sac_update_step(self, batch) -> Dict[str, float]:
        """Single SAC update step"""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        
        entropy = 0.0 # placeholder
        current_q1 = torch.zeros(actions.shape[0],) # placeholder
        current_q2 = torch.zeros((actions.shape[0],)) # placeholder
        target_q = torch.zeros(actions.shape[0],) # placeholder
        actor_loss = torch.tensor(0.0) # placeholder
        
        # ---------------- Problem 3.1.2: Soft Bellman target ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.2 ###
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_actions = dist.rsample()
            next_log_probs = torch.clamp(dist.log_prob(next_actions), -20, 20)
            min_q_next = torch.minimum(
                self.critic1_target(next_obs, next_actions),
                self.critic2_target(next_obs, next_actions),
            )
            target_q = rewards + self.gamma * (1 - dones) * (min_q_next - self.alpha * next_log_probs)
        ### END STUDENT SOLUTION  -  3.1.2 ###
        
        # ---------------- Problem 3.1.3: Critic update ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.3 ###
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = torch.mean((current_q1 - target_q)**2)
        critic2_loss = torch.mean((current_q2 - target_q)**2)
        critic_loss = critic1_loss + critic2_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_opt.step()
        ### END STUDENT SOLUTION  -  3.1.3 ###
        
        
        
        # ---------------- Problem 3.1.4: Actor update ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.4 ###
        dist = self.actor(obs)
        new_actions = dist.rsample()
        log_probs = torch.clamp(dist.log_prob(new_actions), -20, 20)
        q_new = torch.minimum(self.critic1(obs, new_actions), self.critic2(obs, new_actions))
        actor_loss = torch.mean(self.alpha * log_probs - q_new)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        entropy = float(-log_probs.mean().item())
        ### END STUDENT SOLUTION  -  3.1.4 ###
        
        # ---------------- Problem 3.1.5: Target soft-updates ---------------
        ### BEGIN STUDENT SOLUTION - 3.1.5 ###
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        ### END STUDENT SOLUTION  -  3.1.5 ###
        
        # Return stats in format expected by runner
        return {
            "actor_loss": float(actor_loss.item()),
            "critic1_loss": float(nn.functional.mse_loss(current_q1, target_q).item()),
            "critic2_loss": float(nn.functional.mse_loss(current_q2, target_q).item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
            "entropy": entropy
        }
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters"""
        # ---------------- Problem 3.1.5 Helper: Soft update implementation ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.5 HELPER ###
        with torch.no_grad():
            for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
                target_param.copy_((1 - self.tau) * target_param + self.tau * local_param)
        ### END STUDENT SOLUTION  -  3.1.5 HELPER ###