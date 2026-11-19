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

**Game variant (defaults).** The contact reward is zero-sum by itself; the runner's out-of-bounds penalty and the runner not seeing
the chaser's velocity were the asymmetries. By default both are removed:
- zero-sum: chaser reward = -(runner reward), so `u_adv + u_good = 0` at every step. The runner's own reward is unchanged; the chaser
  also gains when the runner leaves the arena. NashConv is then the standard zero-sum exploitability. Opt out with `--no_zero_sum`.
- shared velocity: the runner also observes the chaser's velocity (in padded slots, so `obs_dim` stays 14). Opt out with `--no_share_vel`.

- masked `no_op`: MPE2's continuous action is `[no_op, left, right, down, up]` and the dynamics ignore `no_op`. The policy therefore has
  4 action dims and `no_op` is always sent as 0, so a dimension that does nothing no longer adds to entropy, KL or W2. Opt out with
  `--no_mask_noop`. Older 5-dim checkpoints still load (the size is read from the checkpoint) and get 5-dim best responses.

`exploit.py` / `visualize.py` read these from the run's `config.json`, so best responses are trained in the same game. Runs made
before these flags existed have no entry and are treated as the original game. Numbers from different variants are not comparable.

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

# fictitious self-play baseline: learn against a pool of own past snapshots (see "Fictitious self-play" below)
python train.py --algo fsp --num_envs 16 --total_steps 6000000 --seed 0 --out_dir runs/fsp_s0

# best response: one role learns against a fixed opponent (random, scripted, or a frozen checkpoint)
python train.py --learner adv  --opponent random
python train.py --learner good --opponent scripted
python train.py --learner good --opponent ckpt --opponent_ckpt runs/sp_s0/<run>/final.pt

# GIF + trajectory plot of a finished run (saved in that run's folder)
python visualize.py --run_dir runs/sp_s0/<timestamp-uuid>
python visualize.py --run_dir <dir> --episodes 6 --stochastic
```

Useful flags for `train.py`: `--ent_coef`, `--lr`, `--total_steps`, `--seed`, `--threads`, `--device cuda`,
`--asymmetric` (original MPE2 roles; default is identical dynamics for both players), `--no_zero_sum`, `--no_share_vel`, `--no_mask_noop`, `--algo fsp`, `--snapshot_every`, `--out_dir`.
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

## Fictitious self-play (`--algo fsp`)

Baseline to compare naive self-play against. Naive self-play always trains against the *current* policy; fictitious play
best-responds to the *average* of all past policies, and it is that average which converges to a Nash equilibrium in two-player zero-sum games.

- Every `--snapshot_every` iterations (default 50 for fsp) the network is saved to `<run>/pool/snap_<iter>.pt` and joins the pool
  (the initial policy is member 0).
- In each env the learner controls one role (chaser in even envs, runner in odd ones) and a pool member plays the other. The member
  is drawn uniformly at the start of every episode, so the opponent is the uniform average of the past snapshots.
- One network still learns both roles with the same PPO update. Logged `ep_ret_adv` / `ep_ret_good` are the learner's return as
  chaser / runner against the pool.
- Only one agent per env learns, so fsp collects half the learner samples of self-play per env step. `--num_envs 16 --total_steps 6000000`
  matches a 3M-step self-play run in samples per update and number of updates.
- A run produces two policies: `final.pt` (the last iterate, a best response to the average) and the **average policy** (the pool).
  Evaluate both; fictitious play's guarantee is about the average:

```bash
python exploit.py nashconv --ckpt runs/fsp_s0/<run>/final.pt        # last iterate
python exploit.py nashconv --ckpt runs/fsp_s0/<run> --average       # average policy -> <run>/exploit_avg/
python exploit.py crossplay --ckpts runs/sp_s0/<run>/final.pt runs/fsp_s0/<run>/final.pt runs/fsp_s0/<run>/pool
```

`sweep.py` handles fsp configs too: with `algo: fsp` in `fixed` it scores both the final iterate (`nashconv`) and the average
policy (`nashconv_avg`), and an optional top-level `crossplay_with` list adds outside checkpoints (e.g. the self-play baseline)
to the final cross-play. See `configs/fsp_comparison.yaml`.

The average policy is played as a mixture (one member per episode), not distilled into one network. With `--average` the best response
trains against that mixture and is warm-started from `final.pt`. `--snapshot_every N` also works with plain self-play: it only saves
the snapshots (about 4 MB each), e.g. for cross-play over training time.

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
| **`nashconv`** | same | `gap_adv + gap_good` | About 0: no detectable exploit. Larger: easier to exploit. **A finite-budget estimate.** |
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

- **It is an estimate.** PPO may not find the true best response. Check the BR's `training_curves.png`
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


## Preliminary entropy/clipping sweep

Run from `src/` after installing `PyYAML` (`python -m pip install PyYAML`):

```bash
python sweep.py --config configs/ppo_preliminary.yaml --out_dir runs/preliminary
```

The YAML explicitly defines entropy `[0.01, 0.2]` × clipping `[0.1, 0.3]`, three
shared candidate seeds, fixed PPO settings, fixed best-response settings, and
budgets. The four settings use no KL/W2 penalty. The fixed PPO parameters match
the paper repository's winning classical Phantom TTT configuration.

The runner is sequential and resumes completed candidate and exploitability
stages when given the same directory and unchanged config. Interrupted stages
restart from scratch. Failed trials retain tracebacks and are retried on resume.
Only configurations with all seeds completed are ranked, by mean learned
NashConv (lower is better), with training-seed standard deviation. Exact ties
use configuration index. Treat learned NashConv as an estimate, not a certified
bound. Best responses are warm-started; deterministic/stochastic BR selection
uses separate episodes from score reporting.

Outputs include `configurations.json`, `trials.csv`, `rankings.csv`, `summary.json`,
`finalists.json` with exact checkpoint paths, candidate/attacker logs and weights,
per-episode exploitation records, and exploitability/role-gap plots. All four
configurations are also compared against random/scripted opponents and via
cross-play, with CSV/JSON results and plots. Use `--no_comparisons` to postpone
comparisons; rerun without it to evaluate saved checkpoints.

For future explicit grids, change `grid` and `fixed` or add `variants`, each with
its own `fixed` and `grid` mapping (e.g. a penalty-specific `penalty_target` grid).
Only the preliminary config is supplied. The runner does exhaustive enumeration;
it does not perform adaptive Optuna sampling.

Evaluation now rejects non-finite results and protected BR argument overrides.
`play_match(..., return_episodes=True)` additionally returns per-episode rewards;
its stochastic controllers use independent role seeds. NashConv reports raw and
clipped gaps plus paired `gap_se`, using one shared self-play baseline.
