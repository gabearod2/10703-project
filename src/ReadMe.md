# Self-play PPO on 1v1 Simple Tag (MPE2)

## Setup
```bash
micromamba env create -f environment.yaml      # or: conda env create -f environment.yaml
micromamba activate rl_project_env             # or: conda activate rl_project_env
```

## Commands (run from this folder)
```bash
# check everything works (~1 min)
python tests/test_sanity.py
python tests/test_ppo_controls.py

# naive self-play: one network plays both roles
python train.py --total_steps 3000000 --seed 0 --out_dir runs/sp_s0

# best response: one role learns against a fixed opponent (random or scripted)
python train.py --learner adv  --opponent random
python train.py --learner good --opponent scripted

# GIF + trajectory plot of a finished run (saved in that run's folder)
python visualize.py --run_dir runs/sp_s0/<timestamp-uuid>
python visualize.py --run_dir <dir> --episodes 6 --stochastic
```

Useful flags for `train.py`: `--ent_coef`, `--lr`, `--total_steps`, `--seed`, `--threads`, `--device cuda`,
`--asymmetric` (original MPE2 roles; default is identical dynamics for both players), `--out_dir`.
Every PPO hyperparameter is a flag; `python train.py -h` lists them all.

Each run saves `config.json`, `training_log.json`, `training_curves.png`, `latest.pt` and `final.pt`
to `<out_dir>/<timestamp>-<uuid>/`

## Policy clipping and adaptive penalties

Policy clipping and the adaptive distance penalty have independent switches:

```bash
python train.py --clip_policy true  --penalty none  # clipping only (default)
python train.py --clip_policy false --penalty kl   # KL penalty only
python train.py --clip_policy true  --penalty kl   # clipping + KL penalty
python train.py --clip_policy false --penalty none  # unclipped surrogate, no KL penalty
python train.py --penalty wasserstein-2                 # clipping + Wasserstein-2
python train.py --clip_policy false --penalty wasserstein-2 # Wasserstein-2 only
```

`--clip_coef` (default `0.1`)
`--clip_policy false` disables only policy clipping; `--clip_vloss` independently controls value clipping.
Entropy regularization (`--ent_coef`) still applies in every mode.

With `--penalty kl`, the minimized loss adds `beta * mean(KL(old || new))`.
`--penalty_coef_init` (default `1.0`) initializes beta; `--penalty_target` (default `0.01`) sets its adaptation target.

With `--penalty wasserstein-2`, the loss instead adds `beta * mean(W2)` where
`W2 = sqrt(sum((new_mean - old_mean)^2 + (new_std - old_std)^2))` across action dimensions.

`--target_kl` is the existing, separate **approximate-KL early-stopping threshold**, disabled by default.

`training_log.json` metrics:

- `kl_exact_final`: full-batch analytical KL after the last optimizer step, logged in every mode.
- `penalty_coef_used` / `penalty_coef_next`: beta used for this update and for the next (both zero when disabled).
- `penalty_distance_final`: selected full-batch distance after optimization (zero when disabled).
- `penalty_loss`: mean weighted penalty across optimization minibatches (zero when disabled).

`kl` remains the approximate KL averaged during optimization. `clipfrac` remains the fraction of
sampled ratios outside the clipping interval, even when policy clipping is disabled.

`hw3_reference/` holds the original HW3 starter code (Bronars & Jang), used only by the tests.
