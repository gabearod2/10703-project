"""Explicit PPO grids: python sweep.py --config configs/ppo_preliminary.yaml --out_dir runs/preliminary."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from argparse import Namespace
import contextlib

import numpy as np
import yaml
import torch

import train
import exploit
from opponents import load_policy, PolicyOpponent, make_opponent
from utils.evaluate import play_match


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def table(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def expand(cfg):
    defaults = vars(train.get_args([], create_dir=False))
    reserved = {'seed', 'out_dir', 'learner', 'opponent', 'opponent_ckpt', 'init_ckpt', 'eval_every'}
    names = set(cfg['grid']) | set(cfg['fixed'])
    if names - defaults.keys() or names & reserved or set(cfg['grid']) & set(cfg['fixed']):
        raise ValueError('Unknown, reserved, or overlapping grid/fixed parameters')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Provide distinct training seeds')
    branches = cfg.get('variants', [{}])
    result = []
    for branch in branches:
        # Future penalty grids can define branch fixed values and branch-specific target grids.
        grid = dict(cfg['grid'], **branch.get('grid', {}))
        fixed = dict(cfg['fixed'], **branch.get('fixed', {}))
        if (set(grid) | set(fixed)) - defaults.keys() or (set(grid) | set(fixed)) & reserved:
            raise ValueError('Invalid variant parameters')
        for values in itertools.product(*grid.values()):
            args = dict(defaults, **fixed)
            args.update(zip(grid, values))
            args.update(learner='both', eval_every=0)
            n = args['num_envs'] * args['num_steps'] * 2
            if args['num_minibatches'] < 1 or n % args['num_minibatches'] or n // args['num_minibatches'] < 2:
                raise ValueError('Invalid minibatch division')
            if args['total_steps'] < args['num_envs'] * args['num_steps']:
                raise ValueError('Training budget is shorter than one rollout')
            result.append(args)
    envs = {(a['symmetric'], a['zero_sum'], a['share_vel'], a['max_cycles']) for a in result}
    if len(envs) > 1:
        raise ValueError('All sweep configurations must use the same game')
    if not result:
        raise ValueError('Empty grid')
    return result


def run(cfg, root, compare=True):
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'sweep_config.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise ValueError('Config changed; use a new output directory')
    configurations = expand(cfg)
    save(snapshot, cfg)
    rows, checkpoints = [], []
    for index, settings in enumerate(configurations):
        for seed in cfg['seeds']:
            directory = root / f'config_{index:03d}' / f'seed_{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            candidate = directory / 'candidate'
            candidate.mkdir(exist_ok=True)
            args = dict(settings, seed=seed, out_dir=str(candidate))
            metrics_path = directory / 'metrics.json'
            try:
                if not (candidate / 'complete.json').exists():
                    save(candidate / 'config.json', args)
                    with (candidate / 'console.log').open('w') as log, contextlib.redirect_stdout(log):
                        train.train(Namespace(**args))
                    save(candidate / 'complete.json', {'finished': True})
                b = cfg['br']
                if metrics_path.exists():
                    metric = json.loads(metrics_path.read_text())
                else:
                    with (directory / 'exploit.log').open('w') as log, contextlib.redirect_stdout(log):
                        metric = exploit.nashconv(str(candidate), br_steps=b['steps'], br_seeds=b['seeds'],
                            seed=b['seed'], episodes=b['episodes'], eval_seed=b['eval_seed'], report_seed=b['report_seed'],
                            out_dir=str(directory / 'exploit'), br_init=b['init'], br_args=b['args'],
                            device=args['device'], threads=args['threads'])
                    save(metrics_path, metric)
                row = dict(configuration=index, seed=seed, status='complete', nashconv=metric['nashconv'],
                           adv_gap=metric['adv']['gap'], good_gap=metric['good']['gap'],
                           checkpoint=str(candidate / 'final.pt'))
                note = ''
                if settings.get('algo') == 'fsp':
                    # Fictitious self-play: also score the AVERAGE policy (uniform mixture over candidate/pool),
                    # which is the policy fictitious play's convergence guarantee is about.
                    average_path = directory / 'metrics_avg.json'
                    if average_path.exists():
                        average = json.loads(average_path.read_text())
                    else:
                        with (directory / 'exploit_avg.log').open('w') as log, contextlib.redirect_stdout(log):
                            average = exploit.nashconv(str(candidate), br_steps=b['steps'], br_seeds=b['seeds'],
                                seed=b['seed'], episodes=b['episodes'], eval_seed=b['eval_seed'], report_seed=b['report_seed'],
                                out_dir=str(directory / 'exploit_avg'), br_init=b['init'], br_args=b['args'],
                                device=args['device'], threads=args['threads'], average=True)
                        save(average_path, average)
                    row.update(nashconv_avg=average['nashconv'], adv_gap_avg=average['adv']['gap'],
                               good_gap_avg=average['good']['gap'], pool=str(candidate / 'pool'))
                    note = f' | average policy NashConv={average["nashconv"]:.4f}'
                rows.append(row)
                checkpoints.append(str(candidate / 'final.pt'))
                print(f'config {index}, seed {seed}: NashConv={metric["nashconv"]:.4f}{note}', flush=True)
            except Exception as error:
                import traceback
                save(directory / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
                rows.append(dict(configuration=index, seed=seed, status='failed', error=repr(error)))
                print(f'config {index}, seed {seed}: FAILED {error}', flush=True)
            table(root / 'trials.csv', rows)
    rankings = []
    for i in range(len(configurations)):
        scores = [r['nashconv'] for r in rows if r['configuration'] == i and r['status'] == 'complete']
        if len(scores) == len(cfg['seeds']):
            rankings.append(dict(configuration=i, mean_nashconv=float(np.mean(scores)),
                                 std_nashconv=float(np.std(scores, ddof=1)) if len(scores)>1 else 0.,
                                 n_seeds=len(scores)))
            averages = [r['nashconv_avg'] for r in rows if r['configuration'] == i and 'nashconv_avg' in r]
            if len(averages) == len(scores):                      # fsp configurations: average-policy score as well
                rankings[-1].update(mean_nashconv_avg=float(np.mean(averages)),
                                    std_nashconv_avg=float(np.std(averages, ddof=1)) if len(averages)>1 else 0.)
    rankings.sort(key=lambda r: (r['mean_nashconv'], r['configuration']))
    table(root / 'rankings.csv', rankings)
    save(root / 'configurations.json', configurations)
    save(root / 'summary.json', dict(rankings=rankings, winner=rankings[0] if rankings else None))
    save(root / 'finalists.json', dict(checkpoints=checkpoints, trials=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if rankings:
        fig, ax = plt.subplots()
        ax.bar([str(r['configuration']) for r in rankings], [r['mean_nashconv'] for r in rankings],
               yerr=[r['std_nashconv'] for r in rankings], capsize=4)
        ax.set(xlabel='Configuration', ylabel='Learned NashConv', title='Mean ± training-seed SD')
        fig.tight_layout(); fig.savefig(root / 'exploitability.png'); plt.close(fig)
        fig, ax = plt.subplots()
        for role in ('adv', 'good'):
            ax.plot([str(r['configuration']) for r in rankings],
                    [np.mean([t[role+'_gap'] for t in rows if t['configuration']==r['configuration'] and t['status']=='complete']) for r in rankings],
                    marker='o', label=role)
        ax.set(ylabel='Mean role deviation gain'); ax.legend()
        fig.tight_layout(); fig.savefig(root / 'role_gaps.png'); plt.close(fig)
    if compare and checkpoints:
        c = cfg['comparisons']
        successful = [r for r in rows if r['status'] == 'complete']
        extra = list(cfg.get('crossplay_with', []))               # e.g. self-play checkpoints from another sweep
        for path in [e for e in extra if not Path(e).exists()]:
            print(f'crossplay_with: {path} not found, skipped', flush=True)
        extra = [e for e in extra if Path(e).exists()]
        pooled = [r for r in successful if r.get('pool')]          # a pool directory plays as its average policy
        # Extras and pools get no configuration id, so they appear in the per-checkpoint plot but not the
        # configuration-averaged one.
        exploit.crossplay(extra + checkpoints + [r['pool'] for r in pooled], episodes=c['episodes'], seed=c['seed'],
                          out=str(root / 'crossplay'), device=settings['device'],
                          labels=[exploit.checkpoint_label(n, e) for n, e in enumerate(extra)]
                                 + [f"Config {r['configuration']} | seed {r['seed']}" for r in successful]
                                 + [f"Config {r['configuration']} | seed {r['seed']} (avg)" for r in pooled],
                          configuration_ids=[None] * len(extra) + [r['configuration'] for r in successful]
                                            + [None] * len(pooled),
                          seeds=[None] * len(extra) + [r['seed'] for r in successful] + [None] * len(pooled),
                          expected_seeds=cfg['seeds'])
        fixed, episode_records = [], []
        for row in rows:
            if row['status'] != 'complete': continue
            policy = load_policy(row['checkpoint'], settings['device'])
            opts = configurations[row['configuration']]
            for role in (0, 1):
                for name in ('random', 'scripted'):
                    ctrl = PolicyOpponent(policy, device=opts['device'])
                    opponent = make_opponent(name, role ^ 1, policy.low.numel(), seed=c['seed'])   # env action dim (policy's excludes masked no_op)
                    pair = (ctrl, opponent) if role == 0 else (opponent, ctrl)
                    m = play_match(*pair, c['episodes'], opts['max_cycles'], c['seed'], opts['symmetric'],
                                   return_episodes=True, zero_sum=opts['zero_sum'], share_vel=opts['share_vel'])
                    fixed.append(dict(configuration=row['configuration'], seed=row['seed'], role=role,
                                      opponent=name, mean=m['adv' if role==0 else 'good'],
                                      se=m['adv_se' if role==0 else 'good_se']))
                    episode_records.append(dict(configuration=row['configuration'], seed=row['seed'], role=role, opponent=name, returns=m['returns']))
                    save(root / 'fixed_episodes.json', episode_records)
        table(root / 'fixed_opponents.csv', fixed)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (role, name) in zip(axes.flat, itertools.product((0,1), ('random','scripted'))):
            for i in range(len(configurations)):
                values=[r['mean'] for r in fixed if r['configuration']==i and r['role']==role and r['opponent']==name]
                if values: ax.bar(str(i),np.mean(values),yerr=np.std(values))
            ax.set(title=f'{("chaser","runner")[role]} vs {name}', ylabel='Return')
        fig.tight_layout(); fig.savefig(root / 'fixed_opponents.png'); plt.close(fig)
    if any(r['status']=='failed' for r in rows):
        raise RuntimeError('Some trials failed; see failure.json files. Resume to retry them.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/ppo_preliminary.yaml')
    p.add_argument('--out_dir', required=True, type=Path)
    p.add_argument('--no_comparisons', action='store_true')
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    import fcntl
    a.out_dir.mkdir(parents=True,exist_ok=True)
    with (a.out_dir/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg,a.out_dir.resolve(),not a.no_comparisons)

if __name__=='__main__': main()
