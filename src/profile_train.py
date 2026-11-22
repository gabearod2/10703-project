"""Where does one PPO iteration spend its time?  Run from src/ (stop other GPU jobs first for clean numbers):

    python profile_train.py --device cuda                       # num_envs 8 (your BR/sweep setting), 64, 256
    python profile_train.py --device cpu --num_envs 8           # CPU reference

Mimics a best-response training: chaser learns against a frozen policy opponent (which, as in train.py, lives on the CPU).
Every timed section is bracketed by cuda.synchronize(), so GPU time is attributed to the section that queued it.
"""
import argparse, time
from collections import defaultdict

import numpy as np
import torch

import train
from envs.tag import ADV
from envs.fast_tag import make_vec_env
from algos.networks import GaussianActorCritic
from algos.ppo import PPO, compute_gae
from algos.rollout import collect_rollout, learner_obs
from opponents import PolicyOpponent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_envs', type=int, nargs='+', default=[8, 64, 256])
    p.add_argument('--num_steps', type=int, default=128)
    p.add_argument('--iters', type=int, default=8, help='timed iterations per setting (after 2 warm-up)')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--backend', choices=['fast', 'mpe'], default='fast')
    a = p.parse_args()
    dev = torch.device(a.device)
    torch.set_num_threads(a.threads)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == 'cuda' else (lambda: None)

    print(f'{"envs":>5} {"ms/iter":>8} {"sps":>7} | {"env.step":>9} {"policy.act":>10} {"opponent":>9} {"glue":>7} {"gae":>6} {"update":>7}   (ms per iteration, share in %)')
    for n in a.num_envs:
        args = train.get_args(['--num_envs', str(n), '--num_steps', str(a.num_steps), '--device', a.device,
                               '--learner', 'adv', '--hidden', '512', '512', '512'], create_dir=False)
        torch.manual_seed(0); np.random.seed(0)
        env = make_vec_env(n, args.max_cycles, seed=0, backend=a.backend)
        A, D = env.act_dim, env.obs_dim
        policy = GaussianActorCritic(D, A - 1, env.act_low, env.act_high, tuple(args.hidden), args.init_log_std).to(dev)
        opponent = PolicyOpponent(GaussianActorCritic(D, A - 1, env.act_low, env.act_high, tuple(args.hidden), args.init_log_std))
        ppo = PPO(policy, args)

        acc = defaultdict(float)

        def timed(name, fn, gpu):
            def wrapper(*x, **k):
                if gpu: sync()
                t = time.perf_counter(); out = fn(*x, **k)
                if gpu: sync()
                acc[name] += time.perf_counter() - t
                return out
            return wrapper

        env.step = timed('env', env.step, False)
        policy.act = timed('act', policy.act, True)
        timed_opponent = timed('opp', opponent, False)
        roles = [ADV]
        obs_np = env.reset()
        total = 0.0
        for it in range(a.iters + 2):
            if it == 2:
                acc.clear(); total = 0.0
            sync(); t0 = time.perf_counter()
            roll = collect_rollout(env, policy, obs_np, roles, timed_opponent, a.num_steps, args.gamma, dev)
            sync(); t1 = time.perf_counter()
            obs_np = roll['next_obs_np']
            with torch.no_grad():
                last_v = policy.value(torch.as_tensor(learner_obs(obs_np, roles), device=dev))
            adv, ret = compute_gae(roll['rew'], roll['val'], roll['done'], last_v, args.gamma, args.gae_lambda)
            sync(); t2 = time.perf_counter()
            batch = dict(obs=roll['obs'].reshape(-1, D), act=roll['act'].reshape(-1, A - 1), logp=roll['logp'].reshape(-1),
                         val=roll['val'].reshape(-1), adv=adv.reshape(-1), ret=ret.reshape(-1))
            ppo.update(batch)
            sync(); t3 = time.perf_counter()
            if it >= 2:
                acc['rollout'] += t1 - t0; acc['gae'] += t2 - t1; acc['update'] += t3 - t2; total += t3 - t0
        k = a.iters
        glue = acc['rollout'] - acc['env'] - acc['act'] - acc['opp']
        parts = [acc['env'], acc['act'], acc['opp'], glue, acc['gae'], acc['update']]
        cell = lambda x: f'{1000 * x / k:6.0f}/{100 * x / total:2.0f}%'
        print(f'{n:>5} {1000 * total / k:>8.0f} {n * a.num_steps * k / total:>7.0f} | ' + ' '.join(f'{cell(x):>9}' for x in parts), flush=True)
        env.close()
    print('\nglue = rollout time not in env/act/opponent (tensor<->numpy copies, buffer writes, truncation bootstrap). '
          'If env.step dominates, a GPU env is the lever; if act/update dominate at num_envs=8, the GPU is latency-bound '
          '(many tiny kernels + syncs) and more parallel trials is the lever.')


if __name__ == '__main__':
    main()
