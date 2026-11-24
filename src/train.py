# train.py
"""
Baseline PPO with a diagonal-Gaussian policy for 1v1 MPE2 Simple Tag (Rudolph et al. 2025 PPO setup, App. G.5).

    python train.py                                           # naive self-play: one network plays both roles
    python train.py --learner adv --opponent random           # best-response check: chaser learns vs fixed opponent
    python train.py --algo fsp                                # fictitious self-play: learn vs a pool of own past snapshots

Per iteration:  collect_rollout (algos/rollout.py) -> compute_gae -> PPO.update (algos/ppo.py) -> log / eval.
"""
import argparse, json, os, time
from datetime import datetime
from uuid import uuid4
import numpy as np
import torch

from envs.tag import ADV, GOOD
from envs.fast_tag import make_vec_env
from algos.networks import GaussianActorCritic
from algos.ppo import PENALTY_DISTANCES, PPO, compute_gae
from algos.rollout import collect_rollout, learner_obs
from opponents import make_opponent, load_policy, PoolOpponent
from utils.evaluate import evaluate
from utils.plotting import plot_log


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    env = make_vec_env(args.num_envs, args.max_cycles, seed=args.seed, backend=getattr(args, "env_backend", "fast"),
                       symmetric=args.symmetric, zero_sum=args.zero_sum, share_vel=args.share_vel)
    try:
        return _train(args, env, device)
    finally:
        env.close()


def _train(args, env, device):
    N, T, A, D = args.num_envs, args.num_steps, env.act_dim, env.obs_dim
    # learner="both": self-play (one network, both roles). learner="adv"/"good": that role learns vs a FIXED opponent.
    roles = [ADV, GOOD] if args.learner == "both" else [ADV if args.learner == "adv" else GOOD]
    opp_role = None if args.learner == "both" else (GOOD if roles[0] == ADV else ADV)
    opponent = None if opp_role is None else make_opponent(args.opponent, opp_role, A, args.seed + 7,
                                                           ckpt=args.opponent_ckpt, deterministic=args.opponent_deterministic)

    PA = A - 1 if args.mask_noop else A      # policy action dims: env dim 0 (no_op) does nothing, so it is masked by default
    policy = GaussianActorCritic(D, PA, env.act_low, env.act_high, tuple(args.hidden), args.init_log_std,
                                 args.action_transform).to(device)
    if args.init_ckpt:                       # warm start (e.g. best response initialised from the policy it responds to)
        init = load_policy(args.init_ckpt)
        if init.act_dim != PA:
            raise ValueError(f"--init_ckpt has {init.act_dim} action dims but this run uses {PA}: "
                             + ("add --no_mask_noop" if args.mask_noop else "remove --no_mask_noop"))
        policy.load_state_dict(init.state_dict())
        policy.action_transform = args.action_transform
    ppo = PPO(policy, args)

    # Fictitious self-play: the network still learns both roles, but in every env it controls ONE role (alternating by
    # env index) and the other role is a snapshot of itself, drawn uniformly from the pool at the start of each episode.
    fsp = args.algo == "fsp"
    pool_dir = os.path.join(args.out_dir, "pool")
    if fsp:
        roles = np.arange(N) % 2                       # env i: learner is ADV if i is even, GOOD if odd
        opponent = PoolOpponent(args.seed + 7, args.opponent_deterministic, device)

    def snapshot(it):
        os.makedirs(pool_dir, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(pool_dir, f"snap_{it:06d}.pt"))
        if fsp:
            opponent.add(policy)

    if args.snapshot_every > 0:
        snapshot(0)                                    # the initial policy is the pool's first member

    num_iters = args.total_steps // (T * N)
    per_env = 2 if args.learner == "both" and not fsp else 1          # learning agents per env
    print(f"obs_dim={D} act_dim={PA}{' (no_op masked)' if args.mask_noop else ''} | {N} envs x {T} steps x {per_env} learner(s) = {N * T * per_env} samples/iter, "
          f"{num_iters} iterations, device={device}" + (f" | FSP, snapshot every {args.snapshot_every} iterations" if fsp else ""))

    log = {k: [] for k in ["iter", "env_steps", "ep_ret_adv", "ep_ret_good", "loss", "policy_loss", "value_loss",
                           "entropy", "kl", "penalty_distance_final", "kl_exact_final", "penalty_coef_used", "penalty_coef_next", "penalty_loss",
                           "clipfrac", "explained_var", "std", "lr", "sps"]}
    log["eval"] = []
    obs_np = env.reset()
    recent = []                                        # finished-episode returns [adv, good]
    recent_role = {ADV: [], GOOD: []}                  # FSP: the learner's own return, split by the role it played
    t0, env_steps = time.time(), 0

    for it in range(1, num_iters + 1):
        lr = args.lr * (1.0 - (it - 1) / num_iters) if args.anneal_lr else args.lr
        ppo.set_lr(lr)

        roll = collect_rollout(env, policy, obs_np, roles, opponent, T, args.gamma, device)
        obs_np = roll["next_obs_np"]
        recent += roll["episodes"]
        if fsp:
            for ret, i in zip(roll["episodes"], roll["episode_env"]):
                recent_role[roles[i]].append(ret[roles[i]])
        env_steps += T * N

        with torch.no_grad():
            last_v = policy.value(torch.as_tensor(learner_obs(obs_np, roles), device=device))
        adv, ret = compute_gae(roll["rew"], roll["val"], roll["done"], last_v, args.gamma, args.gae_lambda)
        batch = dict(obs=roll["obs"].reshape(-1, D), act=roll["act"].reshape(-1, PA), logp=roll["logp"].reshape(-1),
                     val=roll["val"].reshape(-1), adv=adv.reshape(-1), ret=ret.reshape(-1))
        m = ppo.update(batch)

        # ---- logging ----
        rec = np.array(recent[-40:]) if recent else np.zeros((1, 2))
        ep_adv, ep_good = float(rec[:, ADV].mean()), float(rec[:, GOOD].mean())
        if fsp:                                        # learner as chaser vs pool runner / learner as runner vs pool chaser
            ep_adv, ep_good = (float(np.mean(recent_role[r][-20:])) if recent_role[r] else 0.0 for r in (ADV, GOOD))
        m.update(iter=it, env_steps=env_steps, ep_ret_adv=ep_adv, ep_ret_good=ep_good,
                 std=float(policy.log_std.detach().exp().mean()), lr=lr, sps=env_steps / (time.time() - t0))
        for k in log:
            if k in m: log[k].append(m[k])
        if it % args.log_every == 0 or it == 1:
            print(f"[it {it:>4d} | {env_steps:>8d} steps] ret adv {m['ep_ret_adv']:>6.1f} good {m['ep_ret_good']:>6.1f} | "
                  f"H {m['entropy']:>5.2f} std {m['std']:.2f} KL {m['kl']:.4f} clip {m['clipfrac']:.2f} "
                  f"V {m['value_loss']:.3f} EV {m['explained_var']:.2f} | {m['sps']:.0f} sps")
            if args.penalty != "none":
                print(f"   {args.penalty} final {m['penalty_distance_final']:.5f} target {args.penalty_target:g} | "
                      f"beta {m['penalty_coef_used']:g} -> {m['penalty_coef_next']:g} | penalty {m['penalty_loss']:.5f}")

        if args.eval_every > 0 and (it % args.eval_every == 0 or it == num_iters):
            ev = evaluate(policy, args.eval_episodes, args.max_cycles, device=device, symmetric=args.symmetric,
                          zero_sum=args.zero_sum, share_vel=args.share_vel, backend=getattr(args, "env_backend", "fast"))
            ev["env_steps"] = env_steps
            log["eval"].append(ev)
            print("   eval:", {k: round(v, 1) for k, v in ev.items() if k != "env_steps"})
            torch.save(policy.state_dict(), os.path.join(args.out_dir, "latest.pt"))

        if args.snapshot_every > 0 and it % args.snapshot_every == 0 and it < num_iters:
            snapshot(it)                               # joins the pool from the next episode on

    torch.save(policy.state_dict(), os.path.join(args.out_dir, "final.pt"))
    with open(os.path.join(args.out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    plot_log(log, os.path.join(args.out_dir, "training_curves.png"))
    print("saved to", args.out_dir)
    return policy, log


def create_run_dir(out_dir):
    """Reserve a unique run directory, even for simultaneous trials sharing a root."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    while True:
        path = os.path.join(out_dir, f"{timestamp}-{uuid4().hex}")
        try:
            os.mkdir(path)  # Exclusive creation; never reuse another trial's directory.
        except FileExistsError:
            continue
        return path


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def argument_parser():
    p = argparse.ArgumentParser("Self-play PPO (Gaussian) on 1v1 MPE2 Simple Tag")
    p.add_argument("--total_steps", type=int, default=2_000_000, help="env steps (per-env transitions, summed over envs)")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--max_cycles", type=int, default=25, help="episode length")
    p.add_argument("--env_backend", choices=["fast", "mpe"], default="fast",
                   help="fast = batched NumPy Simple Tag (envs/fast_tag.py, same dynamics); mpe = the MPE2/PettingZoo reference")
    p.add_argument("--asymmetric", dest="symmetric", action="store_false",
                   help="use original MPE2 roles (faster/smaller runner); default = identical dynamics for both")
    p.add_argument("--no_zero_sum", dest="zero_sum", action="store_false",
                   help="original rewards. Default: chaser reward = -(runner reward), so the game is exactly zero-sum")
    p.add_argument("--no_share_vel", dest="share_vel", action="store_false",
                   help="original observations. Default: the runner also observes the chaser's velocity")
    p.add_argument("--no_mask_noop", dest="mask_noop", action="store_false",
                   help="original 5-dim policy. Default: 4 action dims, the unused MPE2 no_op dim is always sent as 0")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal_lr", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--clip_coef", type=float, default=0.1, help="shared policy and value clipping threshold")
    p.add_argument("--clip_policy", type=lambda s: s.lower() != "false", default=True,
                   help="enable policy-ratio clipping (true/false); independent of --clip_vloss")
    p.add_argument("--penalty", choices=["none", *PENALTY_DISTANCES], default="none",
                   help="adaptive Gaussian distance penalty (default: none)")
    p.add_argument("--penalty_target", type=positive_float, default=0.01,
                   help="target final selected distance for penalty adaptation; separate from --target_kl early stopping")
    p.add_argument("--penalty_coef_init", type=positive_float, default=1.0,
                   help="initial adaptive distance penalty coefficient")
    p.add_argument("--ent_coef", type=float, default=0.05)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--norm_adv", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--clip_vloss", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--target_kl", type=float, default=None,
                   help="optional approximate-KL early stopping threshold; independent of penalty adaptation")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--action_transform", choices=("tanh", "clip"), default="tanh",
                   help="tanh with directional action entropy, or legacy hard clipping with Gaussian entropy")
    p.add_argument("--init_log_std", type=float, default=-0.5)
    p.add_argument("--learner", choices=["both", "adv", "good"], default="both",
                   help="both = self-play (default). adv/good = only that role learns vs a fixed --opponent")
    p.add_argument("--algo", choices=["sp", "fsp"], default="sp",
                   help="with --learner both. sp = naive self-play (default). fsp = fictitious self-play: each env pairs the "
                        "learner (one role) with a uniformly drawn past snapshot of itself (the other role)")
    p.add_argument("--snapshot_every", type=int, default=None,
                   help="save a snapshot to <run>/pool/ every this many iterations (0 = off). Default: 50 for fsp, 0 for sp")
    p.add_argument("--opponent", choices=["random", "scripted", "ckpt", "pool"], default="random",
                   help="fixed opponent for --learner adv/good; ckpt = frozen checkpoint given by --opponent_ckpt; "
                        "pool = uniform mixture over the snapshots in the directory given by --opponent_ckpt")
    p.add_argument("--opponent_ckpt", type=str, default=None,
                   help="a .pt file or run dir (final.pt), or a pool dir for --opponent pool. It plays the non-learning role.")
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="initialise the learner from this checkpoint (.pt or run dir); architecture must match --hidden")
    p.add_argument("--opponent_deterministic", action="store_true",
                   help="frozen opponent (or FSP pool member) plays its mean action (default: samples, i.e. the policy as trained)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=50, help="iterations between evals (0=off)")
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="runs/selfplay_ppo")
    return p


def finalize_args(a):
    if a.opponent in ("ckpt", "pool") and a.learner != "both" and not a.opponent_ckpt:
        raise ValueError(f"--opponent {a.opponent} requires --opponent_ckpt")
    if a.algo == "fsp" and a.learner != "both":
        raise ValueError("--algo fsp trains both roles; do not combine it with --learner adv/good")
    if a.snapshot_every is None:
        a.snapshot_every = 50 if a.algo == "fsp" else 0
    if a.algo == "fsp" and a.snapshot_every <= 0:
        raise ValueError("--algo fsp needs --snapshot_every > 0")
    if a.algo == "fsp" and a.num_envs < 2:
        raise ValueError("--algo fsp needs at least two environments because learner roles are assigned by environment")
    return a


def get_args(argv=None, create_dir=True):
    p = argument_parser()
    try:
        a = finalize_args(p.parse_args(argv))
    except ValueError as error:
        p.error(str(error))
    if create_dir:
        a.out_dir = create_run_dir(a.out_dir)
    return a


if __name__ == "__main__":
    args = get_args()
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    train(args)
