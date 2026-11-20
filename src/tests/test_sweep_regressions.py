import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep
import train


class SweepRegressions(unittest.TestCase):
    def test_algorithm_defaults_and_validation(self):
        self.assertEqual(train.get_args(['--algo', 'fsp'], create_dir=False).snapshot_every, 50)
        cfg = dict(seeds=[0], fixed={}, grid={'algo': ['sp', 'fsp']})
        self.assertEqual([a['snapshot_every'] for a in sweep.expand(cfg)], [0, 50])
        cfg.update(grid={}, variants=[{'fixed': {'algo': 'fsp', 'snapshot_every': 7}}])
        self.assertEqual(sweep.expand(cfg)[0]['snapshot_every'], 7)
        for setting, message in [({'snapshot_every': 0}, 'snapshot_every'), ({'num_envs': 1}, 'two environments')]:
            with self.assertRaisesRegex(ValueError, message):
                sweep.expand(dict(seeds=[0], grid={}, fixed=dict(algo='fsp', **setting)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'runs'
            with self.assertRaises(SystemExit):
                train.get_args(['--algo', 'fsp', '--num_envs', '1', '--out_dir', str(root)])
            self.assertFalse(root.exists())

    def test_fsp_minibatches_and_initial_snapshot(self):
        fixed = dict(num_envs=2, num_steps=2, total_steps=4, num_minibatches=4, hidden=[8])
        cfg = dict(seeds=[0], grid={}, fixed=fixed)
        self.assertEqual(len(sweep.expand(cfg)), 1)
        fixed['algo'] = 'fsp'
        with self.assertRaisesRegex(ValueError, 'minibatch'):
            sweep.expand(cfg)
        fixed['num_minibatches'] = 2
        args = Namespace(**sweep.expand(cfg)[0])
        with tempfile.TemporaryDirectory() as directory:
            args.out_dir = directory
            _, log = train.train(args)
            self.assertTrue((Path(args.out_dir) / 'pool/snap_000000.pt').is_file())
            self.assertTrue(np.isfinite(log['loss'][0]))

    def test_resume_checks_all_seeds_before_writing(self):
        cfg = dict(seeds=[0, 1], grid={}, fixed={}, br={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep.save(root / 'sweep_config.json', cfg)
            candidate = root / 'config_000/seed_1/candidate'
            candidate.mkdir(parents=True)
            sweep.save(candidate / 'complete.json', {'finished': True})
            with self.assertRaisesRegex(ValueError, 'missing config.json'):
                sweep.run(cfg, root)
            sweep.save(candidate / 'config.json', {})
            with self.assertRaisesRegex(ValueError, 'seed_1.*saved mask_noop=False, requested True'):
                sweep.run(cfg, root)
            self.assertFalse((root / 'config_000/seed_0').exists())
            self.assertFalse((root / 'trials.csv').exists())
            for seed in cfg['seeds']:
                trial = root / f'config_000/seed_{seed}'
                (trial / 'candidate').mkdir(parents=True, exist_ok=True)
                sweep.save(trial / 'candidate/config.json', {'mask_noop': True})
                sweep.save(trial / 'candidate/complete.json', {'finished': True})
                sweep.save(trial / 'metrics.json', dict(nashconv=0, adv={'gap': 0}, good={'gap': 0}))
            sweep.run(cfg, root, compare=False)
            self.assertEqual(json.loads((root / 'summary.json').read_text())['winner']['n_seeds'], 2)
            sweep.save(candidate / 'config.json', {'mask_noop': False})
            with self.assertRaisesRegex(ValueError, 'seed_1.*saved mask_noop=False'):
                sweep.run(cfg, root)
