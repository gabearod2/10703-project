# train.py
"""
Baseline PPO with a diagonal-Gaussian policy for 1v1 MPE2 Simple Tag (Rudolph et al. 2025 PPO setup, App. G.5).

    python train.py                                           # naive self-play: one network plays both roles
    python train.py --learner adv --opponent random           # best-response check: chaser learns vs fixed opponent

Per iteration:  collect_rollout (algos/rollout.py) -> compute_gae -> PPO.update (algos/ppo.py) -> log / eval.
"""
import argparse, json, os, time
from datetime import datetime
import numpy as np
import torch

from envs.tag import TagVecEnv, ADV, GOOD
from algos.networks import GaussianActorCritic
from algos.ppo import PPO, compute_gae
from algos.rollout import collect_rollout
from opponents import make_opponent
from utils.evaluate import evaluate
from utils.plotting import plot_log


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    env = TagVecEnv(args.num_envs, args.max_cycles, seed=args.seed, symmetric=args.symmetric)
    N, T, A, D = args.num_envs, args.num_steps, env.act_dim, env.obs_dim
    # learner="both": self-play (one network, both roles). learner="adv"/"good": that role learns vs a FIXED opponent.
    roles = [ADV, GOOD] if args.learner == "both" else [ADV if args.learner == "adv" else GOOD]
    opp_role = None if args.learner == "both" else (GOOD if roles[0] == ADV else ADV)
    opponent = None if opp_role is None else make_opponent(args.opponent, opp_role, A, args.seed + 7)

    policy = GaussianActorCritic(D, A, env.act_low, env.act_high, tuple(args.hidden), args.init_log_std).to(device)
    ppo = PPO(policy, args)

    num_iters = args.total_steps // (T * N)
    print(f"obs_dim={D} act_dim={A} | {N} envs x {T} steps x {len(roles)} learner(s) = {N * T * len(roles)} samples/iter, "
          f"{num_iters} iterations, device={device}")

    log = {k: [] for k in ["iter", "env_steps", "ep_ret_adv", "ep_ret_good", "loss", "policy_loss", "value_loss",
                           "entropy", "kl", "clipfrac", "explained_var", "std", "lr", "sps"]}
    log["eval"] = []
    obs_np = env.reset()
    recent = []                                        # finished-episode returns [adv, good]
    t0, env_steps = time.time(), 0

    for it in range(1, num_iters + 1):
        lr = args.lr * (1.0 - (it - 1) / num_iters) if args.anneal_lr else args.lr
        ppo.set_lr(lr)

        roll = collect_rollout(env, policy, obs_np, roles, opponent, T, args.gamma, device)
        obs_np = roll["next_obs_np"]
        recent += roll["episodes"]
        env_steps += T * N

        with torch.no_grad():
            last_v = policy.value(torch.as_tensor(obs_np[:, roles].reshape(-1, D), device=device))
        adv, ret = compute_gae(roll["rew"], roll["val"], roll["done"], last_v, args.gamma, args.gae_lambda)
        batch = dict(obs=roll["obs"].reshape(-1, D), act=roll["act"].reshape(-1, A), logp=roll["logp"].reshape(-1),
                     val=roll["val"].reshape(-1), adv=adv.reshape(-1), ret=ret.reshape(-1))
        m = ppo.update(batch)

        # ---- logging ----
        rec = np.array(recent[-40:]) if recent else np.zeros((1, 2))
        m.update(iter=it, env_steps=env_steps, ep_ret_adv=float(rec[:, ADV].mean()), ep_ret_good=float(rec[:, GOOD].mean()),
                 std=float(policy.log_std.detach().exp().mean()), lr=lr, sps=env_steps / (time.time() - t0))
        for k in log:
            if k in m: log[k].append(m[k])
        if it % args.log_every == 0 or it == 1:
            print(f"[it {it:>4d} | {env_steps:>8d} steps] ret adv {m['ep_ret_adv']:>6.1f} good {m['ep_ret_good']:>6.1f} | "
                  f"H {m['entropy']:>5.2f} std {m['std']:.2f} KL {m['kl']:.4f} clip {m['clipfrac']:.2f} "
                  f"V {m['value_loss']:.3f} EV {m['explained_var']:.2f} | {m['sps']:.0f} sps")

        if args.eval_every > 0 and (it % args.eval_every == 0 or it == num_iters):
            ev = evaluate(policy, args.eval_episodes, args.max_cycles, device=device, symmetric=args.symmetric)
            ev["env_steps"] = env_steps
            log["eval"].append(ev)
            print("   eval:", {k: round(v, 1) for k, v in ev.items() if k != "env_steps"})
            torch.save(policy.state_dict(), os.path.join(args.out_dir, "latest.pt"))

    torch.save(policy.state_dict(), os.path.join(args.out_dir, "final.pt"))
    with open(os.path.join(args.out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    plot_log(log, os.path.join(args.out_dir, "training_curves.png"))
    env.close()
    print("saved to", args.out_dir)
    return policy, log


def get_args(argv=None):
    p = argparse.ArgumentParser("Self-play PPO (Gaussian) on 1v1 MPE2 Simple Tag")
    p.add_argument("--total_steps", type=int, default=2_000_000, help="env steps (per-env transitions, summed over envs)")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--max_cycles", type=int, default=25, help="episode length")
    p.add_argument("--asymmetric", dest="symmetric", action="store_false",
                   help="use original MPE2 roles (faster/smaller runner); default = identical dynamics for both")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal_lr", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--clip_coef", type=float, default=0.1)
    p.add_argument("--ent_coef", type=float, default=0.05)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--norm_adv", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--clip_vloss", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--target_kl", type=float, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--init_log_std", type=float, default=-0.5)
    p.add_argument("--learner", choices=["both", "adv", "good"], default="both",
                   help="both = self-play (default). adv/good = only that role learns vs a fixed --opponent")
    p.add_argument("--opponent", choices=["random", "scripted"], default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=50, help="iterations between evals (0=off)")
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="runs/selfplay_ppo")
    a = p.parse_args(argv)
    a.out_dir = os.path.join(a.out_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(a.out_dir, exist_ok=True)
    return a


if __name__ == "__main__":
    args = get_args()
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    train(args)
