# runner.py
import argparse, os
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from td3_agent import TD3Agent
from utils import set_seed, make_env, evaluate_policy, record_eval_video, plot_curves

def create_agent(agent_type, env_info, args, device):
    """Factory function to create the appropriate agent"""
    if agent_type.lower() == "ppo":
        return PPOAgent(
            env_info=env_info,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            rollout_steps=args.rollout_steps,
            device=device,
        )
    elif agent_type.lower() == "sac":
        return SACAgent(
            env_info=env_info,
            lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            batch_size=args.batch_size,
            update_every=args.update_every,
            buffer_size=args.buffer_size,
            warmup_steps=args.warmup_steps,
            utd_ratio=args.utd_ratio,
            device=device,
        )
    elif agent_type.lower() == "td3":
        return TD3Agent(
            env_info=env_info,
            lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            delay=args.delay,
            batch_size=args.batch_size,
            update_every=args.update_every,
            buffer_size=args.buffer_size,
            warmup_steps=args.warmup_steps,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            exploration_noise=args.exploration_noise,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose 'ppo', 'sac', or 'td3'")

def initialize_log(agent_type):
    """Initialize logging dictionary based on agent type"""
    base_log = {
        "steps": [],
        "episodic_return": [],
        "eval_mean": [],
        "eval_std": [],
        "eval_steps": [],
    }
    
    if agent_type.lower() == "ppo":
        base_log.update({
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clipfrac": [],
        })
    elif agent_type.lower() == "sac":
        base_log.update({
            "actor_loss": [],
            "critic1_loss": [],
            "critic2_loss": [],
            "q1": [],
            "q2": [],
            "entropy": [],
        })
    elif agent_type.lower() == "td3":
        base_log.update({
            "actor_loss": [],
            "critic1_loss": [],
            "critic2_loss": [],
            "q1": [],
            "q2": [],
        })
    
    return base_log

def log_training_stats(agent_type, stats, log, total_steps, updates_performed):
    """Log training statistics in a unified way"""
    
    if not stats:  # No update occurred
        return
        
    recent_returns = log['episodic_return'][-10:] if log['episodic_return'] else [0]
    avg_return = np.mean(recent_returns)
    
    if agent_type.lower() == "ppo":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Loss: {log['loss'][-1]:>7.4f} | "
            f"π: {log['policy_loss'][-1]:>7.4f} | "
            f"V: {log['value_loss'][-1]:>7.4f} | "
            f"H: {log['entropy'][-1]:>6.3f} | "
            f"KL: {log['kl'][-1]:>8.5f} | "
            f"Clip%: {log['clipfrac'][-1]:>5.1%} | "
            f"Ret: {avg_return:>7.1f}"
        )
    elif agent_type.lower() == "sac":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Actor: {log['actor_loss'][-1]:>7.4f} | "
            f"C1: {log['critic1_loss'][-1]:>7.4f} | "
            f"C2: {log['critic2_loss'][-1]:>7.4f} | "
            f"Q1: {log['q1'][-1]:>7.2f} | "
            f"Q2: {log['q2'][-1]:>7.2f} | "
            f"H: {log['entropy'][-1]:>6.3f} | "
            f"Ret: {avg_return:>7.1f}"
        )
    elif agent_type.lower() == "td3":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Actor: {log['actor_loss'][-1]:>7.4f} | "
            f"C1: {log['critic1_loss'][-1]:>7.4f} | "
            f"C2: {log['critic2_loss'][-1]:>7.4f} | "
            f"Q1: {log['q1'][-1]:>7.2f} | "
            f"Q2: {log['q2'][-1]:>7.2f} | "
            f"Ret: {avg_return:>7.1f}"
        )

def create_transition(agent_type, obs, action, reward, next_obs, terminated, truncated, action_info):
    """Create transition dict appropriate for the agent type"""
    # For off-policy methods (SAC/TD3), treat truncations as terminals to avoid
    # bootstrapping across env resets (TimeLimit). PPO handles truncation separately.
    if agent_type.lower() in ["sac", "td3"]:
        done_flag = bool(terminated or truncated)
    else:
        done_flag = bool(terminated)

    base_transition = {
        "obs": obs.copy(),
        "action": action.copy(),
        "reward": float(reward),
        "next_obs": next_obs.copy(),
        "done": done_flag,
        "truncated": bool(truncated),
    }
    
    if agent_type.lower() == "ppo":
        # PPO needs log_prob and value from action_info
        base_transition.update({
            "log_prob": action_info.get("log_prob", 0.0),
            "value": action_info.get("value", 0.0),
        })
    # SAC doesn't need extra info beyond base transition
    
    return base_transition

def save_final_model(agent, agent_type, out_dir):
    """Save the final model with appropriate attribute access"""
    final_model_path = os.path.join(out_dir, "final.pt")
    
    if agent_type.lower() == "ppo":
        if hasattr(agent, 'actor'):
            torch.save(agent.actor.state_dict(), final_model_path)
        elif hasattr(agent, 'policy'):
            torch.save(agent.policy.state_dict(), final_model_path)
        else:
            print("Warning: Could not find policy to save for PPO agent")
            return
    elif agent_type.lower() in ["sac", "td3"]:
        torch.save(agent.actor.state_dict(), final_model_path)
    
    print(f"Final model saved: {final_model_path}")

def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Environment setup
    env = make_env(args.env_id, render=False)
    print(f"Environment: {args.env_id}")
    
    # Set seeds
    set_seed(args.seed)

    # Environment spaces
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env_info = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_low": env.action_space.low,
        "act_high": env.action_space.high
    }
    print(f"Obs dim: {obs_dim}, Action dim: {act_dim}")

    # Create agent
    agent = create_agent(args.agent, env_info, args, device)
    print(f"Created {args.agent.upper()} agent")

    # Setup output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Initialize logging
    log = initialize_log(args.agent)

    # Training state
    obs, _ = env.reset(seed = args.seed)
    ep_ret = 0.0
    total_steps = 0
    best_eval = -1e9
    episode_idx = 0
    updates_performed = 0

    print(f"Starting {args.agent.upper()} training...")
    print(f"Target steps: {args.total_steps}")
    if args.agent.lower() == "ppo":
        print(f"Rollout steps: {args.rollout_steps}")
        print(f"Update epochs: {args.update_epochs}")
        print(f"Minibatch size: {args.minibatch_size}")
    else:  # SAC
        print(f"Batch size: {args.batch_size}")
        print(f"Update every: {args.update_every}")
        print(f"Buffer size: {args.buffer_size}")
        print(f"Warmup steps: {args.warmup_steps}")
    print("-" * 50)

    last_stat_update = 0
    while total_steps < args.total_steps:
        # Get action from agent
        action_info = agent.act(obs)
        action = action_info["action"]

        # Ensure action is in correct format for environment
        a_env = np.asarray(action, dtype=np.float32).reshape(-1)
        a_env = np.clip(a_env, env.action_space.low, env.action_space.high)

        # Environment step
        next_obs, r, terminated, truncated, _ = env.step(a_env)
        stop = bool(terminated or truncated)
        
        # Create transition appropriate for agent type
        transition = create_transition(
            args.agent, obs, action, r, next_obs, terminated, truncated, action_info
        )

        # Step the agent
        stats = agent.step(transition)

        ep_ret += r
        obs = next_obs
        total_steps += 1

        # Handle episode boundary
        if stop:
            log["steps"].append(total_steps)
            log["episodic_return"].append(ep_ret)
            
            # Reset environment
            episode_idx += 1
            obs, _ = env.reset(seed=args.seed + 100 * episode_idx)
            ep_ret = 0.0

        # Collect optimization stats when update occurred
        if stats:  # stats is non-empty dict when update happened
            updates_performed += 1
            # Add all stats to log (different agents have different metrics)
            for k, v in stats.items():
                if k in log:
                    log[k].append(v)
            
        if stats:
            if total_steps - last_stat_update >= args.log_every:
                log_training_stats(args.agent, stats, log, total_steps, updates_performed)
                last_stat_update = total_steps

        # Periodic evaluation
        if args.eval_every > 0 and total_steps % args.eval_every == 0:
            print(f"\n--- Evaluation at step {total_steps} ---")
            with torch.no_grad():
                mean_r, std_r = evaluate_policy(
                    agent, 
                    env_id=args.env_id, 
                    episodes=args.eval_episodes,
                    seed=1000
                )
            log["eval_mean"].append(mean_r)
            log["eval_std"].append(std_r)
            log["eval_steps"].append(total_steps)
            
            print(f"Eval: {mean_r:.1f} ± {std_r:.1f}")
            
            # Save best model
            if mean_r > best_eval:
                best_eval = mean_r
                model_path = os.path.join(args.out_dir, "best.pt")
                # Save appropriate model part
                if hasattr(agent, 'actor'):
                    torch.save(agent.actor.state_dict(), model_path)
                elif hasattr(agent, 'policy'):
                    torch.save(agent.policy.state_dict(), model_path)
                print(f"New best model saved! Return: {mean_r:.1f}")
            print("--- End Evaluation ---\n")

        # Periodic video recording
        if args.video_every > 0 and total_steps % args.video_every == 0:
            print(f"Recording video at step {total_steps}...")
            name = f"{args.video_prefix}_t{total_steps}"
            try:
                with torch.no_grad():
                    path = record_eval_video(
                        agent, 
                        video_dir=args.video_dir, 
                        video_name=name, 
                        env_id=args.env_id, 
                        episodes=1
                    )
                print(f"Video saved: {path}")
            except Exception as e:
                print(f"Video recording failed: {e}")
                import traceback
                traceback.print_exc()

        # Periodic plot refresh
        if args.plot_every > 0 and total_steps % args.plot_every == 0:
            if any(len(log.get(k, [])) > 0 for k in log.keys() if k not in ['eval_mean', 'eval_std', 'eval_steps']):
                try:
                    plot_path = os.path.join(args.out_dir, "training_curves.png")
                    plot_curves(log, out_path=plot_path)
                    # print(f"Training plots updated: {plot_path}")
                except Exception as e:
                    print(f"Plotting failed: {e}")
                    import traceback
                    traceback.print_exc()

    # Final cleanup and saving
    print(f"\nTraining completed! Total steps: {total_steps}")
    print(f"Total {args.agent.upper()} updates: {updates_performed}")
    
    # Final evaluation
    if args.eval_episodes > 0:
        # load best model for final evaluation
        if best_eval > -1e9:
            model_path = os.path.join(args.out_dir, "best.pt")
            if os.path.exists(model_path):
                if hasattr(agent, 'actor'):
                    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
                elif hasattr(agent, 'policy'):
                    agent.policy.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded best model from {model_path} for final evaluation")
            else:
                print(f"Best model file not found at {model_path}, using current model for final eval")
        
        print("Final evaluation...")
        with torch.no_grad():
            mean_r, std_r = evaluate_policy(
                agent, 
                env_id=args.env_id, 
                episodes=args.eval_episodes * 2,  # More episodes for final eval
                seed=4200
            )
        print(f"Final performance: {mean_r:.1f} ± {std_r:.1f}")
        log["eval_mean"].append(mean_r)
        log["eval_std"].append(std_r)
        log["eval_steps"].append(total_steps)

    # Save final artifacts
    try:
        plot_path = os.path.join(args.out_dir, "training_curves.png")
        plot_curves(log, out_path=plot_path)
        print(f"Final training curves saved: {plot_path}")
    except Exception as e:
        print(f"Final plotting failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    save_final_model(agent, args.agent, args.out_dir)
    
    # Save training log as JSON for analysis
    import json
    log_path = os.path.join(args.out_dir, "training_log.json")
    # Convert numpy types to Python types for JSON serialization
    json_log = {}
    for k, v in log.items():
        if isinstance(v, list):
            json_log[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        else:
            json_log[k] = v
    
    with open(log_path, 'w') as f:
        json.dump(json_log, f, indent=2)
    print(f"Training log saved: {log_path}")
    
    env.close()
    print(f"\nAll artifacts saved to: {args.out_dir}")
    print("Training complete!")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified PPO/SAC Training")
    
    # Algorithm selection
    p.add_argument("--agent", type=str, choices=["ppo", "sac", "td3"], default="sac",
                   help="RL algorithm to use")
    
    # Environment
    p.add_argument("--env_id", type=str, default="LunarLanderContinuous-v3",
                   help="Gym environment ID")
    
    # Training
    p.add_argument("--total_steps", type=int, default=1_000_000,
                   help="Total environment steps")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU usage")
    p.add_argument("--seed", type=int, default=42000,
                   help="Random seed")
    
    # Common hyperparameters
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor")
    
    # PPO-specific hyperparameters
    p.add_argument("--rollout_steps", type=int, default=4096,
                   help="Steps per PPO rollout")
    p.add_argument("--update_epochs", type=int, default=10,
                   help="PPO update epochs per rollout")
    p.add_argument("--minibatch_size", type=int, default=128,
                   help="Minibatch size for PPO updates")
    p.add_argument("--gae_lambda", type=float, default=0.98,
                   help="GAE lambda")
    p.add_argument("--clip_coef", type=float, default=0.2,
                   help="PPO clip coefficient")
    p.add_argument("--vf_coef", type=float, default=0.5,
                   help="Value function coefficient")
    p.add_argument("--ent_coef", type=float, default=0.01,
                   help="Entropy coefficient")
    p.add_argument("--max_grad_norm", type=float, default=0.5,
                   help="Max gradient norm for clipping")
    
    # SAC-specific hyperparameters
    p.add_argument("--tau", type=float, default=0.005,
                   help="Target network soft update rate")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="SAC entropy coefficient")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size for SAC updates")
    p.add_argument("--update_every", type=int, default=1,
                   help="Update frequency (every N steps)")
    p.add_argument("--buffer_size", type=int, default=100000,
                   help="Replay buffer size")
    p.add_argument("--warmup_steps", type=int, default=5000,
                   help="Steps before starting updates")
    p.add_argument("--utd_ratio", type=int, default=1,
                   help="SAC updates per env step when updates occur (UTD ratio)")
    # TD3-specific noise hyperparameters
    p.add_argument("--policy_noise", type=float, default=0.2,
                   help="TD3 target policy smoothing noise std")
    p.add_argument("--delay", type=float, default=2,
                   help="TD3 target policy delay")
    p.add_argument("--noise_clip", type=float, default=0.5,
                   help="TD3 target policy noise clip")
    p.add_argument("--exploration_noise", type=float, default=0.1,
                   help="TD3 exploration noise std during env interaction")
    
    # Logging and evaluation
    p.add_argument("--log_every", type=int, default=10_000,
                   help="Print training stats every N steps")
    p.add_argument("--plot_every", type=int, default=1_000,
                   help="Update plots every N steps")
    p.add_argument("--eval_every", type=int, default=50_000,
                   help="Run evaluation every N steps (0 to disable)")
    p.add_argument("--eval_episodes", type=int, default=100,
                   help="Episodes per evaluation")
    
    # Video recording
    p.add_argument("--video_every", type=int, default=100_000,
                   help="Record video every N steps (0 to disable)")
    p.add_argument("--video_dir", type=str, default="videos",
                   help="Video output directory")
    p.add_argument("--video_prefix", type=str, default="eval",
                   help="Video filename prefix")

    
    # Output
    p.add_argument("--out_dir", type=str, default="runs/rl_training",
                   help="Output directory for models and logs")
    
    args = p.parse_args()
    
    # Build base output directory
    if args.out_dir == "runs/rl_training":
        base_out_dir = f"runs/{args.agent}_{args.env_id.lower().replace('-', '_')}"
    else:
        base_out_dir = args.out_dir

    # Append timestamp to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir = os.path.join(base_out_dir, timestamp)
    
    # record how long the command takes
    start = datetime.now()

    # Place videos inside this run directory
    args.video_dir = os.path.join(args.out_dir, "videos")
    
    # Print configuration
    print("=" * 60)
    print(f"{args.agent.upper()} Training Configuration")
    print("=" * 60)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:20s}: {value}")
    print("=" * 60)
    
    run(args)

    # record how long the command takes
    end = datetime.now()
    duration = end - start
    print(f"Training Run Took: {duration}")