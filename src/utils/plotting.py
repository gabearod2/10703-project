# utils/plotting.py
"""Training-curve figure."""
import numpy as np


def plot_log(log, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = log["env_steps"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax[0, 0].plot(x, log["ep_ret_adv"], label="adversary (chaser)"); ax[0, 0].plot(x, log["ep_ret_good"], label="good (runner)")
    ax[0, 0].set_title("self-play episode return (train, stochastic)"); ax[0, 0].legend()
    if log["eval"]:
        ex = [e["env_steps"] for e in log["eval"]]
        for k, lab in [("adv_vs_random_adv", "chaser vs random"), ("adv_vs_flee_adv", "chaser vs scripted flee"),
                       ("good_vs_random_good", "runner vs random"), ("good_vs_chase_good", "runner vs scripted chase")]:
            ax[0, 1].plot(ex, [e[k] for e in log["eval"]], marker="o", label=lab)
        ax[0, 1].set_title("eval vs fixed opponents (deterministic)"); ax[0, 1].legend(fontsize=8)
    for a, k in zip([ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]], ["entropy", "value_loss", "kl", "clipfrac"]):
        a.plot(x, log[k]); a.set_title(k)
    for a in ax.flat: a.set_xlabel("env steps (per env, summed)"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
