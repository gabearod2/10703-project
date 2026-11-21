"""Where does NashConv noise come from: the learned best response, or the training seed?

Re-scores EXISTING checkpoints with several independent best-response (BR) trainings, then splits the variance:
  within-checkpoint SD  = noise of the metric (BR training seed; evaluation episodes are held fixed)
  between-checkpoint SD = noise from the training seed, after removing the BR part
It reuses the BR settings of the sweep config, so each draw is exactly one "sweep-style" NashConv measurement.
Nothing is retrained and no existing run directory is modified; results go to runs/variance_check/<tag>/.

Run from src/:
  python variance_check.py run --group sp                       # 3 self-play ckpts x 4 BR seeds, ~2 h on GPU
  python variance_check.py run --group fsp                      # same for the FSP final iterates
  python variance_check.py run --group sp --br_steps 3000000 --br_seeds 20000 20001 --tag long_br   # stronger BR
  python variance_check.py report --tag default                 # tables + variance split (also --tag long_br)
Draws run in parallel with --jobs N (one process each); several `run` processes may also share the same draws.
Finished draws are skipped on rerun. Processes started by an OLDER version of this file don't claim draws, so
keep those on different --ckpts.
"""
import argparse, contextlib, csv, json, os, traceback
from pathlib import Path

import numpy as np
import yaml

GROUPS = {
    'sp': [f'runs/preliminary/config_000/seed_{s}/candidate' for s in (0, 1, 2)],
    'fsp': [f'runs/fsp_comparison/config_000/seed_{s}/candidate' for s in (0, 1, 2)],
}
ROOT = Path('runs/variance_check')


def label(ckpt):
    p = Path(ckpt).resolve().parts
    return '_'.join(p[-4:-1])                      # e.g. preliminary_config_000_seed_0


def draw_dir(tag, ckpt, br_seed):
    return ROOT / tag / label(ckpt) / f'br_{br_seed}'


def claim(out):
    """Take a draw so concurrent `run` processes never train the same one (atomic exclusive create).
    A claim left behind by a dead process is stale and is taken over."""
    path = out / '.claim'
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return True
        except FileExistsError:
            try:
                os.kill(int(path.read_text().strip()), 0)       # owner still alive?
                return False
            except ProcessLookupError:
                path.unlink(missing_ok=True)                    # stale: retry once
            except (ValueError, PermissionError, FileNotFoundError):
                return False
    return False


def run_draw(ckpt, s, steps, device, threads, b, tag):
    """One best-response NashConv draw (resumable). Returns a one-line message. Safe to run in a worker process."""
    import exploit                                  # lazy: `report` needs no torch
    out = draw_dir(tag, ckpt, s)
    if (out / 'exploitability.json').exists():
        return f'skip {out}'
    out.mkdir(parents=True, exist_ok=True)
    if not claim(out):
        return f'skip {out} (another process is running it; delete {out}/.claim if that process is dead)'
    try:
        with (out / 'exploit.log').open('w') as log, contextlib.redirect_stdout(log):
            r = exploit.nashconv(str(ckpt), br_steps=steps, br_seeds=1, seed=s, episodes=b['episodes'],
                                 eval_seed=b['eval_seed'], report_seed=b['report_seed'], out_dir=str(out),
                                 br_init=b['init'], br_args=b['args'], device=device, threads=threads)
        return (f"{label(ckpt)} br_seed {s}: NashConv={r['nashconv']:.2f} "
                f"(chaser {r['adv']['gap']:.2f}, runner {r['good']['gap']:.2f})")
    except Exception:
        (out / 'failure.txt').write_text(traceback.format_exc())
        return f'{label(ckpt)} br_seed {s}: FAILED, see {out}/failure.txt'
    finally:
        (out / '.claim').unlink(missing_ok=True)


def run(a):
    from parallel import run_parallel
    cfg = yaml.safe_load(Path(a.config).read_text())
    b, fixed = cfg['br'], cfg['fixed']
    steps = a.br_steps or b['steps']
    device = a.device or fixed.get('device', 'cpu')
    ckpts = a.ckpts or GROUPS[a.group]
    reserved = set(range(b['seed'], b['seed'] + 1))
    if reserved & set(a.br_seeds):
        raise SystemExit(f"BR seed {b['seed']} is the sweep's own draw; pick other seeds")
    tasks = [(ckpt, s, steps, device, fixed.get('threads', 1), b, a.tag) for ckpt in ckpts for s in a.br_seeds]
    run_parallel(run_draw, tasks, a.jobs,
                 lambda pos, res: print(res if isinstance(res, str) else f'draw {pos} crashed: {res!r}', flush=True))


def load_draws(a):
    cfg = yaml.safe_load(Path(a.config).read_text())
    rows = []
    base = ROOT / a.tag
    for d in sorted(base.glob('*/br_*/exploitability.json')):
        j = json.loads(d.read_text())
        rows.append(dict(ckpt=d.parent.parent.name, br_seed=int(d.parent.name.split('_')[1]), source='variance_check',
                         nashconv=j['nashconv'], adv_gap=j['adv']['gap'], good_gap=j['good']['gap'],
                         adv_gap_se=j['adv'].get('gap_se', np.nan), good_gap_se=j['good'].get('gap_se', np.nan),
                         br_steps=j['br_steps']))
    # the sweep's own measurement is one more draw with the same settings (only valid for the same BR budget)
    steps = {r['br_steps'] for r in rows}
    if not a.no_sweep_ref and steps == {cfg['br']['steps']}:
        for label_dir in {r['ckpt'] for r in rows}:
            parts = label_dir.split('_')              # <sweep>_config_<i>_seed_<s>
            sweep, conf, seed = parts[0], '_'.join(parts[1:3]), '_'.join(parts[3:5])
            m = Path('runs') / sweep / conf / seed / 'metrics.json'
            if m.exists():
                j = json.loads(m.read_text())
                rows.append(dict(ckpt=label_dir, br_seed=cfg['br']['seed'], source='sweep', nashconv=j['nashconv'],
                                 adv_gap=j['adv']['gap'], good_gap=j['good']['gap'],
                                 adv_gap_se=j['adv'].get('gap_se', np.nan), good_gap_se=j['good'].get('gap_se', np.nan),
                                 br_steps=cfg['br']['steps']))
    return rows


def split(rows, key):
    """One-way random-effects split of `key` over checkpoints (groups) with repeated BR draws."""
    groups = {}
    for r in rows:
        groups.setdefault(r['ckpt'], []).append(r[key])
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    if len(groups) < 2:
        return None
    within = np.mean([np.var(v, ddof=1) for v in groups.values()])
    n = np.mean([len(v) for v in groups.values()])
    means = np.array([np.mean(v) for v in groups.values()])
    between = max(0.0, np.var(means, ddof=1) - within / n)          # true checkpoint-to-checkpoint variance
    total = within + between
    return dict(sd_br=float(np.sqrt(within)), sd_ckpt=float(np.sqrt(between)),
                sd_single_draw=float(np.sqrt(total)), br_share=float(within / total) if total else float('nan'),
                n_ckpts=len(groups), draws_per_ckpt=float(n))


def report(a):
    rows = load_draws(a)
    if not rows:
        raise SystemExit(f'no finished draws under {ROOT / a.tag}')
    out = ROOT / a.tag
    with (out / 'draws.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'{"checkpoint":<36}{"n":>3}{"mean":>8}{"sd":>7}{"min":>8}{"max":>8}   draws')
    for c in sorted({r['ckpt'] for r in rows}):
        v = [r['nashconv'] for r in rows if r['ckpt'] == c]
        print(f'{c:<36}{len(v):>3}{np.mean(v):>8.1f}{np.std(v, ddof=1) if len(v) > 1 else float("nan"):>7.1f}'
              f'{min(v):>8.1f}{max(v):>8.1f}   {" ".join(f"{x:.1f}" for x in v)}')
    summary = {k: split(rows, k) for k in ('nashconv', 'adv_gap', 'good_gap')}
    print('\nvariance split (SD in NashConv units):')
    for k, s in summary.items():
        if s:
            print(f'  {k:<9} BR-only SD {s["sd_br"]:5.1f} | checkpoint SD {s["sd_ckpt"]:5.1f} | '
                  f'single-draw SD {s["sd_single_draw"]:5.1f} | BR share of variance {100 * s["br_share"]:3.0f}%')
    se = np.nanmean([r['adv_gap_se'] ** 2 + r['good_gap_se'] ** 2 for r in rows]) ** 0.5
    print(f'  evaluation-episode noise (gap_se, both roles combined): ~{se:.1f}')
    print('\nReading it: a high BR share means extra BR seeds on saved checkpoints are the cheap fix; a low share means '
          'you need more training seeds. Compare the means across --tag default and --tag long_br to see whether a '
          'stronger BR raises NashConv.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'report'):
        s = sub.add_parser(name)
        s.add_argument('--config', default='configs/ppo_preliminary.yaml', help='source of the BR settings')
        s.add_argument('--tag', default='default', help='output subfolder under runs/variance_check/')
        s.add_argument('--no_sweep_ref', action='store_true', help='report: do not add the sweep value as a draw')
    r = sub.choices['run']
    r.add_argument('--group', choices=GROUPS, default='sp')
    r.add_argument('--ckpts', nargs='+', help='run dirs (the sweep\'s .../candidate); overrides --group')
    r.add_argument('--br_seeds', type=int, nargs='+', default=[20000, 20001, 20002, 20003])
    r.add_argument('--br_steps', type=int, help='default: the config\'s BR budget')
    r.add_argument('--device')
    r.add_argument('--jobs', type=int, default=1, help='draws to run at once, each in its own process (default 1 = serial)')
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
