# Self-play PPO on 1v1 Simple Tag (MPE2)

PPO with a diagonal-Gaussian policy on two-player, continuous-action Simple Tag. Train by self-play, then judge the policy
by how exploitable it is (best responses, cross-play), not only by its training return.

## Layout

- `src/` holds all the code; **run every command below from `src/`**.
  - `train.py`: PPO / self-play training. `exploit.py`: exploitability and cross-play. `visualize.py`: GIFs and trajectory plots.
  - `algos/`: PPO, rollouts, networks. `envs/tag.py`: vectorised Simple Tag. `opponents.py`: random, scripted and frozen-checkpoint
    opponents. `utils/`: evaluation and plotting. `tests/`: tests.
  - `hw3_reference/`: original HW3 starter code (Bronars & Jang), used only by the tests.
- `papers/`: reference papers.

**Roles.** The *chaser* is `adv` in flags and logs (MPE2 `adversary_0`: +10 per contact). The *runner* is `good`
(MPE2 `agent_0`: -10 per contact plus an out-of-bounds penalty). One network plays both roles; the role is a one-hot in the observation.

## Setup
```bash
cd src
micromamba env create -f environment.yaml      # or: conda env create -f environment.yaml
micromamba activate rl_project_env             # or: conda activate rl_project_env
```

## Commands (run from `src/`)
```bash
# check everything works (~1 min)
python tests/test_sanity.py
python tests/test_ppo_controls.py
python tests/test_eval.py

# naive self-play: one network plays both roles
python train.py --total_steps 3000000 --seed 0 --out_dir runs/sp_s0

# best response: one role learns against a fixed opponent (random, scripted, or a frozen checkpoint)
python train.py --learner adv  --opponent random
python train.py --learner good --opponent scripted
python train.py --learner good --opponent ckpt --opponent_ckpt runs/sp_s0/<run>/final.pt

# GIF + trajectory plot of a finished run (saved in that run's folder)
python visualize.py --run_dir runs/sp_s0/<timestamp-uuid>
python visualize.py --run_dir <dir> --episodes 6 --stochastic
```

Useful flags for `train.py`: `--ent_coef`, `--lr`, `--total_steps`, `--seed`, `--threads`, `--device cuda`,
`--asymmetric` (original MPE2 roles; default is identical dynamics for both players), `--out_dir`.
Frozen-opponent flags: `--opponent ckpt --opponent_ckpt <.pt or run dir>`, `--opponent_deterministic` (opponent plays its mean
action; default samples), `--init_ckpt <.pt or run dir>` (start the learner from a checkpoint).
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

## Evaluation: exploitability and cross-play

Goal: judge a trained self-play policy by **how hard it is to exploit**. Code: `exploit.py`, `opponents.py`
(`PolicyOpponent`, `load_policy`), `utils/evaluate.py` (`evaluate`, `play_match`). Tests: `python tests/test_eval.py`.

### Terms

- **Return**: sum of rewards over one 25-step episode. Every number below is in these units.
- **Policy value** `u_role(pi)`: mean return of that role when the checkpoint plays both roles (actions sampled).
- **Best response (BR)**: a policy trained with PPO to maximize one role's return against the frozen checkpoint in the other role.

### Metrics

| Metric | Where | What it means | How to read it |
|---|---|---|---|
| Training return `ep_ret_adv/good` | `training_log.json` | Mean of the last 40 finished self-play episodes, sampled actions | Progress only, not quality |
| Sanity evals (`eval`: `self`, `*_vs_random`, `adv_vs_flee`, `good_vs_chase`) | `training_log.json`, every 50 iterations | Return per role vs random / scripted opponents, mean action (deterministic) | Checks basic skills. A policy can pass and still be exploitable. |
| `policy_value` | `exploit/exploitability.json` | `u_role(pi)` on the report episodes | Baseline the BR must beat |
| `br_value` | same | Return of the best trained BR for that role vs the frozen checkpoint | Higher than `policy_value` means that role can be exploited |
| `gap` / `gap_raw` | same | `gap = max(0, br_value - policy_value)`; `gap_raw` is unclamped | Per-role exploitability |
| **`nashconv`** | same | `gap_adv + gap_good` | About 0: no detectable exploit. Larger: easier to exploit. **A lower bound.** |
| `*_se` | same | Standard error over evaluation episodes | Gaps within about 2 s.e. are noise |
| Cross-play matrix | `crossplay.json` / `.png` | `chaser_return[i][j]`, `runner_return[i][j]`: chaser from checkpoint i vs runner from checkpoint j | Relative strength and cycling; payoff table for PSRO / ELO later |

### How `nashconv` is computed

```bash
python exploit.py nashconv --ckpt runs/sp_s0/<run>/final.pt --br_steps 1000000 --br_seeds 2 --episodes 200
python exploit.py nashconv --ckpt <ckpt> --br_args "--ent_coef 0.0"     # extra train.py flags for the BR
```

1. Load the checkpoint. The frozen opponent samples from its Gaussian (`--opp_deterministic` plays the mean instead).
2. For each role, train `--br_seeds` PPO best responses for `--br_steps` steps against the frozen policy in the other role.
   They are warm-started from the checkpoint (`--br_init warm`, the default), so a BR never starts below the policy.
   `--br_init scratch` trains from random init and needs about the budget of the original self-play run.
3. Each BR is a candidate, played deterministically or sampled. The best candidate is picked on `--eval_seed` episodes.
4. The picked BR and the policy's own value are re-measured on fresh `--report_seed` episodes, so picking the best seed
   does not bias the number upward.
5. `gap = max(0, br_value - policy_value)` per role; `nashconv` is the sum. The game is not exactly zero-sum (out-of-bounds
   penalty), so each role's gap uses measured values and never assumes `u_adv = -u_good`.

Results go to `<run dir>/exploit/exploitability.json`. Top level: `ckpt`, `br_steps`, `br_init`, `episodes`,
`opp_deterministic`, `nashconv`. Per role (`adv`, `good`): `br_value`, `br_se`, `policy_value`, `policy_se`, `gap`, `gap_raw`,
`selected` (seed, deterministic flag, run dir), `candidates` (every seed x {deterministic, sampled}).

### Reading the number

- **It is a lower bound.** PPO may not find the true best response. Check the BR's `training_curves.png`
  (`exploit/br_<role>_s<seed>/<run>/`): if its return is still rising, or seeds disagree (`candidates` in the JSON), raise
  `--br_steps` / `--br_seeds` before trusting the gap.
- **BR hyperparameters can be changed** with `--br_args`, e.g. `--br_args "--ent_coef 0.0"`. With the default entropy bonus the
  action std keeps growing during BR training (about 9 to 20 over 1M steps); with `--ent_coef 0.0` it stays flat. On one
  checkpoint (seed 0, 3M steps) this did not noticeably change NashConv (24.2 default vs 24.4 with `--ent_coef 0.0`), and the
  BR seed-to-seed spread on the runner (about +-5) was larger than the difference. Treat the default as fine, but check the
  curves when comparing runs.
- **Compare only like with like**: same env settings, `--episodes`, and BR budget.
- **Cost** is about `br_steps / steps-per-second`. At roughly 2400 steps/s, one 1M-step BR takes about 7 minutes, so
  `--br_seeds 2` over both roles takes about 30 minutes.

### Cross-play

```bash
python exploit.py crossplay --ckpts runs/a/final.pt runs/b/final.pt --episodes 100 --out results/crossplay
```

Plays every chaser against every runner. Row = chaser's checkpoint, column = runner's checkpoint, cell = mean chaser return
(high favours the chaser). The diagonal is each checkpoint against itself. Cycles (A's chaser beats B's runner, B's beats C's,
C's beats A's) show up as a pattern off the diagonal. It needs two or more checkpoints; `train.py` currently saves only
`latest.pt` and `final.pt`.
