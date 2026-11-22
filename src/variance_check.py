"""Where does NashConv noise come from: the learned best response, or the training seed?

Re-scores EXISTING checkpoints with several independent best-response (BR) trainings, then splits the variance:
  within-checkpoint SD  = noise of the metric (BR training seed; evaluation episodes are held fixed)
  between-checkpoint SD = noise from the training seed, after removing the BR part
It reuses the BR settings of the sweep config, so each draw is exactly one "sweep-style" NashConv measurement.
Candidate checkpoints are frozen throughout; results go to runs/variance_check/<tag>/.

Run from src/:
  python variance_check.py run --group sp                       # 3 self-play ckpts x 4 BR seeds, ~2 h on GPU
  python variance_check.py run --group fsp                      # same for the FSP final iterates
  python variance_check.py run --group sp --br_steps 3000000 --br_seeds 20000 20001 --tag long_br   # stronger BR
  python variance_check.py report --tag default                 # tables + variance split (also --tag long_br)
Draws run in parallel with --jobs N (one process each); several `run` processes may also share the same draws.
Reruns reuse compatible draws. Use separate tags for different BR settings and for workers running older code.
"""
import argparse, contextlib, csv, fcntl, json, traceback
from pathlib import Path

import numpy as np
import yaml

from utils.measurement import read_measurement

GROUPS = {
    'sp': [f'runs/preliminary/config_000/seed_{s}/candidate' for s in (0, 1, 2)],
    'fsp': [f'runs/fsp_comparison/config_000/seed_{s}/candidate' for s in (0, 1, 2)],
}
ROOT = Path('runs/variance_check')


def label(ckpt):
    run_dir = Path(ckpt).resolve()
    if run_dir.name == 'final.pt':
        run_dir = run_dir.parent
    p = run_dir.parts
    return '_'.join(p[-4:-1])                      # e.g. preliminary_config_000_seed_0


def draw_dir(tag, ckpt, br_seed):
    return ROOT / tag / label(ckpt) / f'br_{br_seed}'


def measurement(b, steps, device, threads):
    return dict(br_steps=steps, br_seeds=b['seeds'], episodes=b['episodes'], eval_seed=b['eval_seed'],
                report_seed=b['report_seed'], br_init=b['init'], br_args=b['args'], device=device,
                threads=threads, opp_deterministic=False)


def validate(result, expected, path):
    for key, value in expected.items():
        if key not in result or result[key] != value:
            raise ValueError(f'{path}: incompatible {key} (expected {value!r}, found {result.get(key)!r}); use a fresh tag')


def run_draw(ckpt, s, settings, tag):
    import exploit
    out = draw_dir(tag, ckpt, s)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(ckpt).resolve()
    if checkpoint.is_dir():
        checkpoint /= 'final.pt'
    backend = exploit.env_options(exploit.load_cfg(ckpt))['backend']
    with (out / '.claim').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return f'skip {out} (draw in progress)'
        path = out / 'exploitability.json'
        if path.exists():
            cached = read_measurement(path)
            cached['ckpt'] = str(Path(cached['ckpt']).resolve())
            validate(cached, dict(settings, ckpt=str(checkpoint), seed=s, backend=backend), path)
            return f'skip {out}'
        try:
            with (out / 'exploit.log').open('w') as log, contextlib.redirect_stdout(log):
                r = exploit.nashconv(str(checkpoint), seed=s, out_dir=str(out), **settings)
        except Exception:
            (out / 'failure.txt').write_text(traceback.format_exc())
            raise
    return f"{label(ckpt)} br_seed {s}: NashConv={r['nashconv']:.2f}"


def check_blocks(starts, count):
    ordered = sorted(starts)
    if count < 1 or any(b - a < count for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f'BR seed blocks must be disjoint (each uses {count} consecutive seeds)')


def run(a):
    from parallel import run_parallel
    cfg = yaml.safe_load(Path(a.config).read_text())
    b, fixed = cfg['br'], cfg['fixed']
    settings = measurement(b, a.br_steps if a.br_steps is not None else b['steps'],
                           a.device or fixed.get('device', 'cpu'), fixed.get('threads', 1))
    starts = a.br_seeds or [20000 + i * b['seeds'] for i in range(4)]
    check_blocks([b['seed'], *starts], b['seeds'])
    ckpts = a.ckpts or GROUPS[a.group]
    for path in (ROOT / a.tag).glob('*/br_*/exploitability.json'):
        validate(read_measurement(path), settings, path)
    for ckpt in ckpts:
        cached = {int(path.parent.name[3:]) for path in (ROOT / a.tag / label(ckpt)).glob('br_*/exploitability.json')}
        check_blocks([b['seed'], *(cached | set(starts))], b['seeds'])
    tasks = [(ckpt, s, settings, a.tag) for ckpt in ckpts for s in starts]
    errors = []

    def completed(pos, result):
        if isinstance(result, Exception):
            errors.append(result)
        print(result, flush=True)

    run_parallel(run_draw, tasks, a.jobs, completed)
    if errors:
        raise SystemExit(f'{len(errors)} draw(s) failed: {errors[0]}')


def paired_se(j):
    """Paired episode bootstrap of the sum of clipped role means."""
    baseline = np.asarray(j['baseline_episodes'])
    gains = np.column_stack([np.asarray(j[name]['report_episodes'])[:, role] - baseline[:, role]
                             for role, name in enumerate(('adv', 'good'))])
    n = len(gains)
    if n < 2:
        return float('nan')
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(32):
        indices = rng.integers(n, size=(64, n))
        samples.extend(gains[indices].mean(axis=1).clip(min=0).sum(axis=1))
    return float(np.std(samples, ddof=1))


def load_draws(a):
    b = yaml.safe_load(Path(a.config).read_text())['br']
    rows = []
    groups = {}
    expected = None

    def row(j, checkpoint, source):
        return dict(configuration=str(checkpoint.parents[2]), ckpt=str(checkpoint), br_seed=j['seed'],
                    source=source, nashconv=j['nashconv'], adv_gap=j['adv']['gap'], good_gap=j['good']['gap'],
                    adv_gap_se=j['adv']['gap_se'], good_gap_se=j['good']['gap_se'], nashconv_se=paired_se(j),
                    br_steps=j['br_steps'])

    for d in sorted((ROOT / a.tag).glob('*/br_*/exploitability.json')):
        j = read_measurement(d)
        if expected is None:
            expected = measurement(b, j['br_steps'], j.get('device'), j.get('threads'))
        validate(j, expected, d)
        checkpoint = Path(j['ckpt']).resolve()
        if 'backend' not in j:
            raise ValueError(f'{d}: missing measurement backend; use a fresh tag')
        config_path = checkpoint.parent / 'config.json'
        if config_path.exists():
            backend = json.loads(config_path.read_text()).get('env_backend', 'mpe')
            validate(j, dict(backend=backend), d)
        rows.append(row(j, checkpoint, 'variance_check'))
        groups.setdefault(checkpoint, []).append(j)
    games = {}
    for checkpoint, draws in groups.items():
        first = draws[0]
        common = {k: first[k] for k in ('backend', 'baseline_episodes', 'symmetric', 'max_cycles', 'zero_sum', 'share_vel')}
        game = {k: first[k] for k in ('backend', 'symmetric', 'max_cycles', 'zero_sum', 'share_vel')}
        validate(first, games.setdefault(checkpoint.parents[2], game), checkpoint)
        for j in draws[1:]:
            validate(j, common, checkpoint)
        starts = [j['seed'] for j in draws]
        check_blocks(starts, b['seeds'])
        metrics = checkpoint.parent.parent / 'metrics.json'
        if a.no_sweep_ref or expected['br_steps'] != b['steps'] or not metrics.exists():
            continue
        j = read_measurement(metrics)
        required = dict(expected, **common, seed=b['seed'])
        if Path(j['ckpt']).resolve() == checkpoint and all(j.get(k) == v for k, v in required.items()):
            check_blocks([*starts, j['seed']], b['seeds'])
            rows.append(row(j, checkpoint, 'sweep'))
    return rows


def split(rows, key):
    """One-way random-effects split of `key` over checkpoints (groups) with repeated BR draws."""
    groups = {}
    for r in rows:
        groups.setdefault(r['ckpt'], []).append(r[key])
    counts = np.array([len(v) for v in groups.values()])
    residual_df = counts.sum() - len(groups)
    if len(groups) < 2 or residual_df == 0:
        return None
    within = sum(np.sum((v - np.mean(v)) ** 2) for v in groups.values()) / residual_df
    means = np.array([np.mean(v) for v in groups.values()])
    between = max(0.0, np.var(means, ddof=1) - within * np.mean(1 / counts))
    total = within + between
    return dict(sd_br=float(np.sqrt(within)), sd_ckpt=float(np.sqrt(between)),
                sd_single_draw=float(np.sqrt(total)), br_share=float(within / total) if total else float('nan'),
                n_ckpts=len(groups), draws_per_ckpt=float(counts.mean()))


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
    summary = {}
    for configuration in sorted({r['configuration'] for r in rows}):
        subset = [r for r in rows if r['configuration'] == configuration]
        estimates = {k: split(subset, k) for k in ('nashconv', 'adv_gap', 'good_gap')}
        summary[configuration] = estimates
        print(f'\nvariance split for {configuration} (SD in NashConv units):')
        for k, s in estimates.items():
            if s:
                print(f'  {k:<9} BR-only SD {s["sd_br"]:5.1f} | checkpoint SD {s["sd_ckpt"]:5.1f} | '
                      f'single-draw SD {s["sd_single_draw"]:5.1f} | BR share of variance {100 * s["br_share"]:3.0f}%')
        se = np.mean([r['nashconv_se'] ** 2 for r in subset]) ** 0.5
        summary[configuration]['evaluation_se'] = float(se)
        print(f'  evaluation-episode noise (paired bootstrap NashConv SE): {se:.1f}')
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
    r.add_argument('--br_seeds', type=int, nargs='+', help='first training seed of each replicate block (default: four disjoint blocks starting at 20000)')
    r.add_argument('--br_steps', type=int, help='default: the config\'s BR budget')
    r.add_argument('--device')
    r.add_argument('--jobs', type=int, default=1, help='draws to run at once, each in its own process (default 1 = serial)')
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
