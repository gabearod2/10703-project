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

# naive self-play: one network plays both roles
python train.py --total_steps 3000000 --seed 0 --out_dir runs/sp_s0

# best response: one role learns against a fixed opponent (random or scripted)
python train.py --learner adv  --opponent random
python train.py --learner good --opponent scripted

# GIF + trajectory plot of a finished run (saved in that run's folder)
python visualize.py --run_dir runs/sp_s0/<timestamp>
python visualize.py --run_dir <dir> --episodes 6 --stochastic
```

Useful flags for `train.py`: `--ent_coef`, `--lr`, `--total_steps`, `--seed`, `--threads`, `--device cuda`,
`--asymmetric` (original MPE2 roles; default is identical dynamics for both players), `--out_dir`.
Every PPO hyperparameter is a flag; `python train.py -h` lists them all.

Each run saves `config.json`, `training_log.json`, `training_curves.png`, `latest.pt` and `final.pt`
to `<out_dir>/<timestamp>/`.

`hw3_reference/` holds the original HW3 starter code (Bronars & Jang), used only by the tests.
