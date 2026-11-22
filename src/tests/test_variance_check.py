import contextlib
import fcntl
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import variance_check as vc
from tests.test_eval import save_tiny


class VarianceCheck(unittest.TestCase):
    def test_unbalanced_groups_and_paired_clipping(self):
        rows = [dict(ckpt=k, score=s) for k, values in [('a', [0, 2]), ('b', [4, 6, 8])] for s in values]
        result = vc.split(rows, 'score')
        within = (2 + 8) / 3
        self.assertAlmostEqual(result['sd_ckpt'] ** 2, 12.5 - within * (1/2 + 1/3) / 2)
        baseline = [[1, -1], [-1, 1]]
        metric = dict(baseline_episodes=baseline, adv=dict(report_episodes=[[2, 0]] * 2),
                      good=dict(report_episodes=[[0, 2]] * 2))
        self.assertEqual(vc.paired_se(metric), 0)
        metric['good']['report_episodes'] = [[0, -3]] * 2
        self.assertGreater(vc.paired_se(metric), 0)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            vc.check_blocks([100, 103], 4)

    def test_multiseed_resume_and_sweep_reference(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(vc, 'ROOT', Path(tmp) / 'draws'):
            b = dict(steps=32, seeds=2, seed=10, episodes=4, eval_seed=100, report_seed=200,
                     init='warm', args='--num_envs 2 --num_steps 16')
            cfg = Path(tmp) / 'test.yaml'
            cfg.write_text(yaml.safe_dump(dict(br=b, fixed={})))
            args = Namespace(config=cfg, tag='test', no_sweep_ref=False)
            settings = vc.measurement(b, 32, 'cpu', 1)
            candidate = Path(tmp) / 'fsp_comparison/config_000/seed_1/candidate'
            candidate.mkdir(parents=True)
            save_tiny(path=candidate / 'final.pt')
            out = vc.draw_dir('test', candidate, 20)
            out.mkdir(parents=True)
            with (out / '.claim').open('a') as other:
                fcntl.flock(other, fcntl.LOCK_EX)
                self.assertIn('in progress', vc.run_draw(candidate, 20, settings, 'test'))
            vc.run_draw(candidate, 20, settings, 'test')
            self.assertIn('skip', vc.run_draw(candidate, 20, settings, 'test'))
            with self.assertRaisesRegex(ValueError, 'br_steps'):
                vc.run_draw(candidate, 20, dict(settings, br_steps=64), 'test')
            result = json.loads((out / 'exploitability.json').read_text())
            self.assertEqual(result['backend'], 'mpe')
            for role in ('adv', 'good'):
                self.assertEqual({c['seed'] for c in result[role]['candidates']}, {20, 21})
                br_config = Path(result[role]['selected']['run_dir']) / 'config.json'
                self.assertEqual(json.loads(br_config.read_text())['env_backend'], 'mpe')
            reference = candidate.parent / 'metrics.json'
            reference.write_text(json.dumps(dict(result, seed=10)))
            self.assertEqual([r['source'] for r in vc.load_draws(args)], ['variance_check', 'sweep'])
            reference.write_text(json.dumps(dict(result, seed=10, br_seeds=1)))
            self.assertEqual(len(vc.load_draws(args)), 1)

            second = Path(tmp) / 'preliminary/config_000/seed_1/candidate'
            second_out = vc.draw_dir('test', second, 20)
            second_out.mkdir(parents=True)
            (second_out / 'exploitability.json').write_text(json.dumps(dict(result, ckpt=str(second / 'final.pt'))))
            with contextlib.redirect_stdout(io.StringIO()):
                vc.report(args)
            summary = json.loads((vc.ROOT / 'test/summary.json').read_text())
            self.assertEqual(set(summary), {str(c.resolve().parents[1]) for c in (candidate, second)})
            self.assertTrue(all(s['nashconv'] is None for s in summary.values()))
