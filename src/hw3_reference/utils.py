import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(env_id="LunarLanderContinuous-v3", render=False, seed=None):
    render_mode = "rgb_array" if render else None
    env = gym.make(env_id, render_mode=render_mode)#, continuous=True)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env

@torch.no_grad()
def greedy_action(policy, obs_t):
    """Get deterministic action from policy/actor"""
    if hasattr(policy, 'forward'):
        # This is an ActorCritic (PPO) - returns (dist, value)
        result = policy(obs_t)
        if isinstance(result, tuple):
            dist, _ = result
        else:
            dist = result
    else:
        # This is just an Actor (SAC) - returns dist
        dist = policy(obs_t)
    
    if hasattr(dist, "mean_action"):        # continuous
        a = dist.mean_action
    else:                                   # discrete
        a = torch.argmax(dist.logits, dim=-1)
    return a

def _to_env_action(env, action_tensor):
    if isinstance(env.action_space, gym.spaces.Box):
        a = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        return np.clip(a, env.action_space.low, env.action_space.high)
    else:
        return int(action_tensor.item())

def evaluate_policy(agent, env_id="LunarLanderContinuous-v3", episodes=10, seed=42):
    """Evaluate agent performance - works for both PPO and SAC"""
    env = make_env(env_id, render=False, seed=seed)
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = truncated = False
        ep_r = 0.0
        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Get the policy/actor from agent
            if hasattr(agent, 'actor'):
                policy = agent.actor  # SAC
            elif hasattr(agent, 'policy'):
                policy = agent.policy  # PPO
            else:
                raise ValueError(f"Agent {type(agent)} has no 'actor' or 'policy' attribute")
                
            a = greedy_action(policy, obs_t)
            a_env = _to_env_action(env, a)
            obs, r, done, truncated, _ = env.step(a_env)
            ep_r += r
        scores.append(ep_r)
    env.close()
    return float(np.mean(scores)), float(np.std(scores))

def record_eval_video(agent, video_dir="videos", video_name="eval",
                      env_id="LunarLanderContinuous-v3", seed=None, episodes=1):
    """Record evaluation video - works for both PPO and SAC"""
    os.makedirs(video_dir, exist_ok=True)
    if seed is None:
        env = make_env(env_id, render=True)
    else:
        env = make_env(env_id, render=True)
    env = RecordVideo(env, video_folder=video_dir, name_prefix=video_name)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Get the policy/actor from agent
            if hasattr(agent, 'actor'):
                policy = agent.actor  # SAC
            elif hasattr(agent, 'policy'):
                policy = agent.policy  # PPO
            else:
                raise ValueError(f"Agent {type(agent)} has no 'actor' or 'policy' attribute")
                
            a = greedy_action(policy, obs_t)
            a_env = _to_env_action(env, a)
            obs, _, done, truncated, _ = env.step(a_env)
    
    env.close()
    mp4s = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4") and f.startswith(video_name)])
    return os.path.join(video_dir, mp4s[-1]) if mp4s else ""

def detect_agent_type(log):
    """Detect whether this is PPO, SAC, or TD3 based on log keys"""
    ppo_keys = {"policy_loss", "value_loss", "kl", "clipfrac"}
    sac_keys = {"actor_loss", "critic1_loss", "critic2_loss", "alpha", "entropy"}
    td3_keys = {"actor_loss", "critic1_loss", "critic2_loss", "q1", "q2"}
    
    log_keys = set(log.keys())
    
    if ppo_keys.intersection(log_keys):
        return "ppo"
    elif "alpha" in log_keys or "entropy" in log_keys:
        # If we have alpha or entropy, it's SAC
        return "sac"
    elif td3_keys.intersection(log_keys):
        # If we have actor/critic losses but no alpha/entropy, it's TD3
        return "td3"
    else:
        return "unknown"

def plot_curves(log, out_path="training_curves.png"):
    """
    Universal plotting function that adapts to PPO, SAC, or TD3 based on log contents
    """
    agent_type = detect_agent_type(log)
    
    if agent_type == "ppo":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPO Training Progress', fontsize=16)
        plot_ppo_metrics(log, axes)
    elif agent_type == "sac":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SAC Training Progress', fontsize=16)
        plot_sac_metrics(log, axes)
    elif agent_type == "td3":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TD3 Training Progress', fontsize=16)
        plot_td3_metrics(log, axes)
    else:
        # Fallback: just plot what we can
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Training Progress', fontsize=16)
        plot_basic_metrics(log, axes)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ppo_metrics(log, axes):
    """Plot PPO-specific metrics"""
    
    # Helper function to get x-axis values
    def get_x_values(data_key):
        if data_key == "episodic_return":
            return log.get("steps", list(range(len(log.get(data_key, [])))))[:len(log.get(data_key, []))]
        else:
            # For loss metrics, estimate step values
            steps = log.get("steps", [])
            loss_data = log.get(data_key, [])
            if len(loss_data) == 0:
                return []
            total_steps = steps[-1] if steps else len(loss_data)
            return np.linspace(0, total_steps, len(loss_data))
    
    # Plot episodic returns
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        x_vals = get_x_values("episodic_return")
        axes[0, 0].plot(x_vals, log["episodic_return"], 'b-', alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title("Episode Returns")
        axes[0, 0].set_xlabel("Environment Steps")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(log["episodic_return"]) > 10:
            window = min(50, len(log["episodic_return"]) // 10)
            moving_avg = np.convolve(log["episodic_return"], 
                                   np.ones(window)/window, mode='valid')
            ma_x = x_vals[window-1:len(moving_avg)+window-1]
            axes[0, 0].plot(ma_x, moving_avg, 'r-', linewidth=2, alpha=0.8, label=f'MA({window})')
            axes[0, 0].legend()
    
    # Plot PPO-specific losses
    ppo_metrics = [
        ("loss", "Total Loss", (0, 1)),
        ("policy_loss", "Policy Loss", (0, 2)),
        ("value_loss", "Value Loss", (1, 0)),
        ("entropy", "Entropy", (1, 1)),
        ("kl", "KL Divergence", (1, 2))
    ]
    
    for key, title, (i, j) in ppo_metrics:
        if key in log and len(log[key]) > 0:
            x_vals = get_x_values(key)
            if len(x_vals) > 0:
                axes[i, j].plot(x_vals, log[key], linewidth=1.5, alpha=0.8)
                axes[i, j].set_title(title)
                axes[i, j].set_xlabel("Environment Steps")
                axes[i, j].set_ylabel(title)
                axes[i, j].grid(True, alpha=0.3)
                
                # Add KL divergence reference line
                if key == "kl":
                    axes[i, j].axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Target KL')
                    axes[i, j].legend()

def plot_td3_metrics(log, axes):
    """Plot TD3-specific metrics (similar to SAC but without entropy/alpha)"""
    
    def get_update_steps(data_key):
        """Get x-axis values for update-based metrics"""
        data = log.get(data_key, [])
        if len(data) == 0:
            return []
        return list(range(len(data)))
    
    # Plot episodic returns
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        steps = log.get("steps", list(range(len(log["episodic_return"]))))[:len(log["episodic_return"])]
        axes[0, 0].plot(steps, log["episodic_return"], 'b-', alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title("Episode Returns")
        axes[0, 0].set_xlabel("Environment Steps")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(log["episodic_return"]) > 10:
            window = min(50, len(log["episodic_return"]) // 10)
            moving_avg = np.convolve(log["episodic_return"], 
                                   np.ones(window)/window, mode='valid')
            ma_x = steps[window-1:len(moving_avg)+window-1]
            axes[0, 0].plot(ma_x, moving_avg, 'r-', linewidth=2, alpha=0.8, label=f'MA({window})')
            axes[0, 0].legend()
    
    # Actor loss with smoothing
    if "actor_loss" in log and len(log["actor_loss"]) > 0:
        x_vals = get_update_steps("actor_loss")
        # Plot raw data lightly
        axes[0, 1].plot(x_vals, log["actor_loss"], 'g-', linewidth=0.5, alpha=0.3)
        # Plot moving average prominently
        if len(log["actor_loss"]) > 20:
            window = min(100, len(log["actor_loss"]) // 20)
            moving_avg = np.convolve(log["actor_loss"], np.ones(window)/window, mode='valid')
            ma_x = x_vals[window-1:len(moving_avg)+window-1]
            axes[0, 1].plot(ma_x, moving_avg, 'g-', linewidth=2, alpha=0.9, label=f'MA({window})')
            axes[0, 1].legend()
        axes[0, 1].set_title("Actor Loss")
        axes[0, 1].set_xlabel("Updates")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Critic losses with smoothing
    if "critic1_loss" in log and "critic2_loss" in log:
        if len(log["critic1_loss"]) > 0 and len(log["critic2_loss"]) > 0:
            x_vals = get_update_steps("critic1_loss")
            # Plot raw data lightly
            axes[0, 2].plot(x_vals, log["critic1_loss"], 'r-', linewidth=0.5, alpha=0.2)
            axes[0, 2].plot(x_vals, log["critic2_loss"], 'orange', linewidth=0.5, alpha=0.2)
            
            # Plot moving averages prominently
            if len(log["critic1_loss"]) > 20:
                window = min(100, len(log["critic1_loss"]) // 20)
                c1_ma = np.convolve(log["critic1_loss"], np.ones(window)/window, mode='valid')
                c2_ma = np.convolve(log["critic2_loss"], np.ones(window)/window, mode='valid')
                ma_x = x_vals[window-1:len(c1_ma)+window-1]
                axes[0, 2].plot(ma_x, c1_ma, 'r-', linewidth=2, alpha=0.9, label="Critic 1")
                axes[0, 2].plot(ma_x, c2_ma, 'orange', linewidth=2, alpha=0.9, label="Critic 2")
            axes[0, 2].set_title("Critic Losses")
            axes[0, 2].set_xlabel("Updates")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
    
    # Q-values with smoothing
    if "q1" in log and "q2" in log:
        if len(log["q1"]) > 0 and len(log["q2"]) > 0:
            x_vals = get_update_steps("q1")
            # Plot raw data lightly
            axes[1, 0].plot(x_vals, log["q1"], 'b-', linewidth=0.5, alpha=0.3)
            axes[1, 0].plot(x_vals, log["q2"], 'purple', linewidth=0.5, alpha=0.3)
            
            # Plot moving averages prominently
            if len(log["q1"]) > 20:
                window = min(100, len(log["q1"]) // 20)
                q1_ma = np.convolve(log["q1"], np.ones(window)/window, mode='valid')
                q2_ma = np.convolve(log["q2"], np.ones(window)/window, mode='valid')
                ma_x = x_vals[window-1:len(q1_ma)+window-1]
                axes[1, 0].plot(ma_x, q1_ma, 'b-', linewidth=2, alpha=0.9, label="Q1")
                axes[1, 0].plot(ma_x, q2_ma, 'purple', linewidth=2, alpha=0.9, label="Q2")
            axes[1, 0].set_title("Q-Values")
            axes[1, 0].set_xlabel("Updates")
            axes[1, 0].set_ylabel("Q-Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # Evaluation results
    if "eval_mean" in log and "eval_steps" in log and len(log["eval_mean"]) > 0:
        eval_std = log.get("eval_std", [0] * len(log["eval_mean"]))
        axes[1, 1].errorbar(log["eval_steps"], log["eval_mean"], 
                          yerr=eval_std, marker='o', linewidth=2, alpha=0.8)
        axes[1, 1].set_title("Evaluation Returns")
        axes[1, 1].set_xlabel("Environment Steps")
        axes[1, 1].set_ylabel("Mean Return")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Hide unused subplot if no evaluation data
        axes[1, 1].axis('off')
    
    # Hide the third subplot in bottom row (no entropy for TD3)
    axes[1, 2].axis('off')

def plot_sac_metrics(log, axes):
    """Plot SAC-specific metrics"""
    
    def get_update_steps(data_key):
        """Get x-axis values for update-based metrics"""
        data = log.get(data_key, [])
        if len(data) == 0:
            return []
        return list(range(len(data)))
    
    # Plot episodic returns
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        steps = log.get("steps", list(range(len(log["episodic_return"]))))[:len(log["episodic_return"])]
        axes[0, 0].plot(steps, log["episodic_return"], 'b-', alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title("Episode Returns")
        axes[0, 0].set_xlabel("Environment Steps")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(log["episodic_return"]) > 10:
            window = min(50, len(log["episodic_return"]) // 10)
            moving_avg = np.convolve(log["episodic_return"], 
                                   np.ones(window)/window, mode='valid')
            ma_x = steps[window-1:len(moving_avg)+window-1]
            axes[0, 0].plot(ma_x, moving_avg, 'r-', linewidth=2, alpha=0.8, label=f'MA({window})')
            axes[0, 0].legend()
    
    # Actor loss with smoothing
    if "actor_loss" in log and len(log["actor_loss"]) > 0:
        x_vals = get_update_steps("actor_loss")
        # Plot raw data lightly
        axes[0, 1].plot(x_vals, log["actor_loss"], 'g-', linewidth=0.5, alpha=0.3)
        # Plot moving average prominently
        if len(log["actor_loss"]) > 20:
            window = min(100, len(log["actor_loss"]) // 20)
            moving_avg = np.convolve(log["actor_loss"], np.ones(window)/window, mode='valid')
            ma_x = x_vals[window-1:len(moving_avg)+window-1]
            axes[0, 1].plot(ma_x, moving_avg, 'g-', linewidth=2, alpha=0.9, label=f'MA({window})')
            axes[0, 1].legend()
        axes[0, 1].set_title("Actor Loss")
        axes[0, 1].set_xlabel("Updates")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Critic losses with smoothing
    if "critic1_loss" in log and "critic2_loss" in log:
        if len(log["critic1_loss"]) > 0 and len(log["critic2_loss"]) > 0:
            x_vals = get_update_steps("critic1_loss")
            # Plot raw data lightly
            axes[0, 2].plot(x_vals, log["critic1_loss"], 'r-', linewidth=0.5, alpha=0.2)
            axes[0, 2].plot(x_vals, log["critic2_loss"], 'orange', linewidth=0.5, alpha=0.2)
            
            # Plot moving averages prominently
            if len(log["critic1_loss"]) > 20:
                window = min(100, len(log["critic1_loss"]) // 20)
                c1_ma = np.convolve(log["critic1_loss"], np.ones(window)/window, mode='valid')
                c2_ma = np.convolve(log["critic2_loss"], np.ones(window)/window, mode='valid')
                ma_x = x_vals[window-1:len(c1_ma)+window-1]
                axes[0, 2].plot(ma_x, c1_ma, 'r-', linewidth=2, alpha=0.9, label="Critic 1")
                axes[0, 2].plot(ma_x, c2_ma, 'orange', linewidth=2, alpha=0.9, label="Critic 2")
            axes[0, 2].set_title("Critic Losses")
            axes[0, 2].set_xlabel("Updates")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
    
    # Q-values with smoothing
    if "q1" in log and "q2" in log:
        if len(log["q1"]) > 0 and len(log["q2"]) > 0:
            x_vals = get_update_steps("q1")
            # Plot raw data lightly
            axes[1, 0].plot(x_vals, log["q1"], 'b-', linewidth=0.5, alpha=0.3)
            axes[1, 0].plot(x_vals, log["q2"], 'purple', linewidth=0.5, alpha=0.3)
            
            # Plot moving averages prominently
            if len(log["q1"]) > 20:
                window = min(100, len(log["q1"]) // 20)
                q1_ma = np.convolve(log["q1"], np.ones(window)/window, mode='valid')
                q2_ma = np.convolve(log["q2"], np.ones(window)/window, mode='valid')
                ma_x = x_vals[window-1:len(q1_ma)+window-1]
                axes[1, 0].plot(ma_x, q1_ma, 'b-', linewidth=2, alpha=0.9, label="Q1")
                axes[1, 0].plot(ma_x, q2_ma, 'purple', linewidth=2, alpha=0.9, label="Q2")
            axes[1, 0].set_title("Q-Values")
            axes[1, 0].set_xlabel("Updates")
            axes[1, 0].set_ylabel("Q-Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
    # Entropy (SAC-specific)
    if "entropy" in log and len(log["entropy"]) > 0:
        x_vals = get_update_steps("entropy")
        # Add smoothing for entropy too
        axes[1, 1].plot(x_vals, log["entropy"], 'm-', linewidth=0.5, alpha=0.3)
        if len(log["entropy"]) > 20:
            window = min(100, len(log["entropy"]) // 20)
            entropy_ma = np.convolve(log["entropy"], np.ones(window)/window, mode='valid')
            ma_x = x_vals[window-1:len(entropy_ma)+window-1]
            axes[1, 1].plot(ma_x, entropy_ma, 'm-', linewidth=2, alpha=0.9, label=f'MA({window})')
            axes[1, 1].legend()
        axes[1, 1].set_title("Policy Entropy")
        axes[1, 1].set_xlabel("Updates")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    # Evaluation results
    if "eval_mean" in log and "eval_steps" in log and len(log["eval_mean"]) > 0:
        eval_std = log.get("eval_std", [0] * len(log["eval_mean"]))
        axes[1, 2].errorbar(log["eval_steps"], log["eval_mean"], 
                          yerr=eval_std, marker='o', linewidth=2, alpha=0.8)
        axes[1, 2].set_title("Evaluation Returns")
        axes[1, 2].set_xlabel("Environment Steps")
        axes[1, 2].set_ylabel("Mean Return")
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')
        


def plot_basic_metrics(log, axes):
    """Fallback plotting for unknown agent types"""
    
    # Plot episodic returns if available
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        steps = log.get("steps", list(range(len(log["episodic_return"]))))[:len(log["episodic_return"])]
        axes[0].plot(steps, log["episodic_return"], 'b-', alpha=0.7, linewidth=1.5)
        axes[0].set_title("Episode Returns")
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Return")
        axes[0].grid(True, alpha=0.3)
    
    # Plot any loss-like metrics
    axes[1].set_title("Loss Metrics")
    axes[1].set_xlabel("Updates")
    axes[1].set_ylabel("Loss Value")
    axes[1].grid(True, alpha=0.3)
    
    loss_keys = [k for k in log.keys() if "loss" in k.lower() and len(log[k]) > 0]
    for key in loss_keys[:5]:  # Limit to 5 metrics to avoid clutter
        if len(log[key]) > 0:
            axes[1].plot(range(len(log[key])), log[key], label=key, alpha=0.8, linewidth=1.5)
    
    if loss_keys:
        axes[1].legend()
        axes[1].set_yscale('symlog', linthresh=1e-3)