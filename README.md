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

## Policy clipping and adaptive KL

Policy clipping and the adaptive KL penalty have independent switches:

```bash
python train.py --clip_policy true  --adaptive_kl false  # clipping only (default)
python train.py --clip_policy false --adaptive_kl true   # KL penalty only
python train.py --clip_policy true  --adaptive_kl true   # clipping + KL penalty
python train.py --clip_policy false --adaptive_kl false  # unclipped surrogate, no KL penalty
```

`--clip_coef` (default `0.1`)
`--clip_policy false` disables only policy clipping; `--clip_vloss` independently controls value clipping.
Entropy regularization (`--ent_coef`) still applies in every mode.

With `--adaptive_kl true`, the minimized loss adds `beta * mean(KL(old || new))`.
`--kl_coef_init` (default `1.0`) initializes beta; `--kl_target` (default `0.01`) sets its adaptation target.

`--target_kl` is the existing, separate **approximate-KL early-stopping threshold**, disabled by default.

`training_log.json` metrics:

- `kl_exact_final`: full-batch analytical KL after the last optimizer step, logged in every mode.
- `kl_coef_used` / `kl_coef_next`: beta used for this update and for the next (both zero when disabled).
- `kl_penalty`: mean weighted penalty across optimization minibatches (zero when disabled).

`kl` remains the approximate KL averaged during optimization. `clipfrac` remains the fraction of
sampled ratios outside the clipping interval, even when policy clipping is disabled.

## Evaluation: exploitability and cross-play (`exploit.py`)

```bash
# train a best response (BR) per role against a frozen checkpoint; NashConv = sum_i [u_i(BR_i, pi_-i) - u_i(pi)]
python exploit.py nashconv --ckpt runs/sp_s0/<run>/final.pt --br_steps 1000000 --br_seeds 2
python exploit.py nashconv --ckpt <ckpt> --br_args "--ent_coef 0.0"        # extra train.py flags for the BR

# chaser-vs-runner return matrix between checkpoints (rows = chaser from ckpt i, cols = runner from ckpt j)
python exploit.py crossplay --ckpts runs/a/final.pt runs/b/final.pt --out results/crossplay
```

`nashconv` writes `<run dir>/exploit/exploitability.json` (BR value, policy value, gap and per-seed candidates for each role).
The BR is trained with PPO, so the number is a **lower bound** on true exploitability: check that the BR is not undertrained
(a chaser/runner BR that scores below the policy's own value is clamped to a gap of 0, and `gap_raw` keeps the unclamped value).
By default the BR is warm-started from the checkpoint (`--br_init warm`, i.e. `train.py --init_ckpt`), so it never starts below
the policy; `--br_init scratch` trains from random init, which needs about the budget of the original self-play run.
Gaps smaller than the printed standard errors are noise: increase `--episodes` and `--br_steps` before reading anything into them.
The frozen policy samples actions by default (the policy as trained); `--opp_deterministic` plays its mean.
Returns are per 25-step episode. BR seeds are selected on `--eval_seed` episodes and reported on fresh `--report_seed` episodes.

The same machinery is available directly: `python train.py --learner adv --opponent ckpt --opponent_ckpt <ckpt or run dir>`
trains one role against a frozen checkpoint, and `utils.evaluate.play_match(adv_ctrl, good_ctrl)` plays any two controllers
(`opponents.PolicyOpponent`, scripted, random) against each other. Tests: `python tests/test_eval.py`.

`hw3_reference/` holds the original HW3 starter code (Bronars & Jang), used only by the tests.
