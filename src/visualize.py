# visualize.py
"""
Roll out a trained policy (both roles use the shared network) and save
  <out>.gif  : animated episodes side by side (red = chaser, green = runner, grey = obstacles, dashed = arena edge)
  <out>.png  : full trajectories of the same episodes

    python visualize.py --run_dir runs/selfplay_ppo/20261105-170000            # uses final.pt + config.json
    python visualize.py --run_dir <dir> --ckpt latest.pt --episodes 6 --stochastic
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from envs.tag import TagVecEnv, ADV, GOOD
from algos.networks import GaussianActorCritic

COL = {ADV: "#d62728", GOOD: "#2ca02c"}


def rollout(policy, venv, episodes, seed, deterministic):
    e, out = venv.envs[0], []
    for ep in range(episodes):
        o, _ = e.reset(seed=seed + ep)
        states, cum, rets = [venv.get_state(0)], np.zeros(2), [np.zeros(2)]
        for _ in range(venv.max_cycles):
            x = torch.as_tensor(venv._encode(o))                                  # [2, D]
            with torch.no_grad():
                a, _, _ = policy.act(x, deterministic=deterministic)
            a = policy.to_env(a).numpy()
            o, r, *_ = e.step({"adversary_0": a[ADV], "agent_0": a[GOOD]})
            cum = cum + np.array([r["adversary_0"], r["agent_0"]])
            states.append(venv.get_state(0)); rets.append(cum.copy())
        out.append(dict(states=states, rets=rets))
    return out


def draw(ax, ep, t, lim=1.7):
    ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k--", lw=0.6, alpha=0.5)
    st = ep["states"][t]
    for pos, size in st["landmarks"]:
        ax.add_patch(Circle(pos, size, color="0.6"))
    for r in (ADV, GOOD):
        tr = np.array([s["agents"][r][0] for s in ep["states"][: t + 1]])
        ax.plot(tr[:, 0], tr[:, 1], color=COL[r], lw=1.2, alpha=0.6)
    dist = np.linalg.norm(st["agents"][ADV][0] - st["agents"][GOOD][0])
    touching = dist < st["agents"][ADV][1] + st["agents"][GOOD][1]
    for r in (ADV, GOOD):
        pos, size = st["agents"][r]
        ax.add_patch(Circle(pos, size, color=COL[r], ec="k" if touching else "none", lw=1.5))
    ax.set_title(f"t={t:2d}  chaser {ep['rets'][t][ADV]:+.0f}  runner {ep['rets'][t][GOOD]:+.0f}", fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", default="final.pt")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--stochastic", action="store_true", help="sample actions (default: deterministic mean)")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default=None, help="output prefix (default <run_dir>/rollout)")
    a = p.parse_args()

    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    venv = TagVecEnv(1, cfg["max_cycles"], seed=a.seed, symmetric=cfg.get("symmetric", True))
    policy = GaussianActorCritic(venv.obs_dim, venv.act_dim, venv.act_low, venv.act_high, tuple(cfg["hidden"]))
    policy.load_state_dict(torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu"))
    policy.eval()
    eps = rollout(policy, venv, a.episodes, a.seed, deterministic=not a.stochastic)
    out = a.out or os.path.join(a.run_dir, "rollout")
    for i, ep in enumerate(eps):
        print(f"episode {i}: chaser {ep['rets'][-1][ADV]:+.1f}  runner {ep['rets'][-1][GOOD]:+.1f}")

    E, T = len(eps), venv.max_cycles
    fig, axs = plt.subplots(1, E, figsize=(3.2 * E, 3.4), squeeze=False); axs = axs[0]
    anim = FuncAnimation(fig, lambda t: [draw(axs[i], eps[i], t) for i in range(E)], frames=T + 1)
    anim.save(out + ".gif", writer=PillowWriter(fps=a.fps)); plt.close(fig)

    fig, axs = plt.subplots(1, E, figsize=(3.2 * E, 3.4), squeeze=False); axs = axs[0]
    for i in range(E):
        draw(axs[i], eps[i], T)
    plt.tight_layout(); plt.savefig(out + ".png", dpi=130); plt.close(fig)
    print("saved", out + ".gif", "and", out + ".png")


if __name__ == "__main__":
    main()
