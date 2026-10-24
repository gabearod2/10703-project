# 10-703 Project: PPO in Imperfect-Information Games

Course project for CMU 10-703 (Deep Reinforcement Learning & Control).

This repo builds on:
- [IIG-RL-Benchmark](https://github.com/nathanlct/IIG-RL-Benchmark) for the training code, and
- [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) (`eas/`, a git submodule) for exact exploitability.

To get running, follow the [Installation](#installation) steps below. They include a
small patch to `eas` (see [Required patch to eas](#required-patch-to-eas-easpyprojecttoml)).

---

# Upstream README: IIG-RL-Benchmark

IIG-RL-Benchmark is a library for running game theoretical and deep RL algorithms on [OpenSpiel](https://github.com/google-deepmind/open_spiel) games. Furthermore, we compute exact exploitability using the [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) library which currently supports Phantom Tic-Tac-Toe and 3x3 Dark Hex, as well as their abrupt versions.

Paper: [arxiv.org/abs/2502.08938](https://arxiv.org/abs/2502.08938)

Play against RL: [Online demo](https://www.nathanlichtle.com/research/2p0s)

## Citation

```
@article{rudolph2025reevaluating,
  title={Reevaluating Policy Gradient Methods for Imperfect-Information Games},
  author={Rudolph, Max and Lichtlé, Nathan and Mohammadpour, Sobhan and Bayen, Alexandre and Kolter, J. Zico and Zhang, Amy and Farina, Gabriele and Vinitsky, Eugene and Sokota, Samuel},
  journal={arXiv preprint arXiv:2502.08938},
  year={2025}
}
```

## Installation

Ensure that your conda is not activated to prevent spawning dual virtual environments.

```bash
# Download this repository (IIG-RL-Benchmark) and install dependencies
git clone https://github.com/nathanlct/IIG-RL-Benchmark.git
cd IIG-RL-Benchmark

# Update the exp-a-spiel (eas) dependency
git submodule init
git submodule update

# Patch eas for newer scikit-build-core (see "Required patch to eas" below)
sed -i 's/^cmake\.\(targets\|verbose\)/build.\1/' eas/pyproject.toml

# This will install eas and its dependencies. Make sure to do this from the top level repo.
pixi install
pixi shell

# Install our custom OpenSpiel (with fixes fixes for PTTT and DH3 and the addition of abrupt PTTT) (our default is python = 3.11)
# If your system isn't linux or you're not on python 3.11, choose the variant for your system (https://github.com/nathanlct/open_spiel/releases)
pip uninstall open_spiel -y
pip install https://github.com/nathanlct/open_spiel/releases/download/v1.pttt/open_spiel-1.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Lastly, install the other requirements
pip install -r requirements.txt
```

### Required patch to eas (`eas/pyproject.toml`)

`pixi install` fails on a fresh clone because `eas/pyproject.toml` (upstream commit `2d83a35`) uses
scikit-build-core option names that were renamed in scikit-build-core 0.10. The build stops with:

```
ERROR: Use build.verbose instead of cmake.verbose for scikit-build-core >= 0.10
```

The fix is to rename two keys under `[tool.scikit-build]` in `eas/pyproject.toml`:

```diff
 [tool.scikit-build]
-cmake.targets = ["pyeas"]
-cmake.verbose = true
+build.targets = ["pyeas"]
+build.verbose = true
```

Nothing else in that file needs to change, and we did not pin `scikit-build-core`.
The `sed` command in the Installation steps above applies this edit.

`eas` is a git submodule pointing at upstream (we did not fork it), so this edit
is **not committed to this repo**. After applying it, `git status` will show `eas` as having
modified content. That is expected. Do not commit it, and re-apply the patch if you
re-clone or reset the submodule (`git submodule update --force`).

### Installing exp-a-spiel on MacOS

The exp-a-spiel dependency (which is only required to compute exact exploitability of policies) supports installation on Linux. To install on OSX, use the following branch in exp-a-spiel (which has not been extensively tested): `osx_install` ([link](https://github.com/gabrfarina/exp-a-spiel/tree/osx_install)).

### (Optional alternative) Installing exp-a-spiel via `pip`

```
cd IIG-RL-Benchmark
pip install ./eas
```

## Usage

Train a PPO agent on Phantom Tic-Tac-Toe for 10M steps with an entropy coefficient of 0.05 and no exploitability computation.

```bash
python main.py algorithm=ppo game=classical_phantom_ttt max_steps=10000000 algorithm.ent_coef=0.05 compute_exploitability=False
```

Train an NFSP agent on Abrupt Dark Hex 3 for 10M steps and compute exact exploitability every 1M steps.

```bash
python main.py algorithm=nfsp game=abrupt_dark_hex max_steps=10000000 compute_exploitability=True compute_exploitability_every=1000000
```

The following algorithms are implemented: `ppo`, `nfsp`, `ppg`, `mmd`, `rnad`, `escher`, `psro`.

Exact exploitability is supported for the following games: `classical_phantom_ttt`, `abrupt_phantom_ttt`, `classical_dark_hex`, `abrupt_dark_hex`.

See the `configs` folder for all available command-line arguments.

## Evaluating head2head 

- Input the paths and algorithm names into a `yaml` file like the example in `head2head/example.yaml`

```
python head2head/head2head_eval.py --agents-yaml path/to/example.yaml --save-dir <save-dir>
```

This command will kick off head2head evaluation for the agents listed in the eval `yaml` using the `eas` library. 

## Games

The following games are also available to play against our RL agents in this [online demo](https://www.nathanlichtle.com/research/2p0s).

### Phantom Tic-Tac-Toe (PTTT)

**Tic-Tac-Toe:** Players take turn placing their symbol (X or O) on a 3x3 grid. The goal is to form a horizontal, vertical, or diagonal line of your symbol. X plays first.

**Phantom:** The opponent's moves are hidden. Selecting an occupied cell reveals it, and the player's turn continues.

**Abrupt PTTT**: In the abrupt variant, if a player's move reveals an opponent's symbol, their turn ends.

### 3x3 Dark Hex (DH3)

**Hex:** Players take turns placing their stones (red or blue) on a 3x3 hexagonal board. The goal is to form a chain of adjacent stones connecting both sides of your color. Red plays first.

**Dark:** The opponent's moves are hidden. Selecting an occupied cell reveals it, and the player's turn continues.

**Abrupt DH3:** In the abrupt variant, if a player's move reveals an opponent's symbol, their turn ends.
