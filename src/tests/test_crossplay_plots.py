"""Cross-play grouping and offline plot regeneration regressions."""
import json
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch

import numpy as np
import exploit


class CrossplayPlots(unittest.TestCase):
    def test_all_pairs_and_order(self):
        matrix = np.arange(72, dtype=float).reshape(6, 6, 2)
        # Interleave configurations and shuffle seed order to catch positional grouping.
        ids, means, counts = exploit.aggregate_crossplay(
            matrix, [1, 0, 1, 0, 1, 0], [2, 1, 0, 2, 1, 0], [0, 1, 2])
        self.assertEqual(ids, [0, 1])
        np.testing.assert_array_equal(counts, np.full((2, 2), 9))
        for i, rows in enumerate(([1, 3, 5], [0, 2, 4])):
            for j, cols in enumerate(([1, 3, 5], [0, 2, 4])):
                for role in range(2):
                    expected = sum(matrix[r, c, role] for r in rows for c in cols) / 9
                    self.assertEqual(means[i, j, role], expected)

    def test_incomplete_and_duplicate_seeds(self):
        matrix = np.zeros((3, 3, 2))
        with patch('builtins.print') as message:
            ids, _, counts = exploit.aggregate_crossplay(matrix, [0, 0, 1], [0, 1, 0], [0, 1])
        self.assertEqual(ids, [0])
        self.assertEqual(counts.tolist(), [[4]])
        message.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            exploit.aggregate_crossplay(matrix, [0, 0, 1], [0, 0, 0], [0, 1])

    def test_legacy_replot_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The checkpoint files deliberately do not exist, as on a relocated Colab run.
            ckpts = [f'/old/run/config_{c}/seed_{s}/candidate/final.pt'
                     for c in range(4) for s in range(3)]
            matrix = np.arange(288, dtype=float).reshape(12, 12, 2)
            original = dict(ckpts=ckpts, episodes=200, deterministic=False,
                            chaser_return=matrix[:, :, 0].tolist(), runner_return=matrix[:, :, 1].tolist())
            source = root / 'crossplay.json'
            source.write_text(json.dumps(original))
            trials = [dict(checkpoint=p, configuration=i // 3, seed=i % 3, status='complete')
                      for i, p in enumerate(ckpts)]
            (root / 'finalists.json').write_text(json.dumps(dict(trials=trials)))
            (root / 'sweep_config.json').write_text(json.dumps(dict(seeds=[0, 1, 2])))
            with patch.object(exploit, 'plot_crossplay_matrix') as plot, patch.object(exploit, 'load_policy') as load:
                exploit.replot_crossplay(source)
            load.assert_not_called()
            self.assertEqual(plot.call_count, 2)
            self.assertEqual(plot.call_args_list[0].args[1],
                             [f'Config {i // 3} | seed {i % 3}' for i in range(12)])
            self.assertEqual(plot.call_args_list[1].args[1], [f'Config {i}' for i in range(4)])
            restored = json.loads(source.read_text())
            for key in ('chaser_return', 'runner_return'):
                self.assertEqual(restored[key], original[key])
            grouped = json.loads((root / 'crossplay_configs.json').read_text())
            self.assertEqual(grouped['seed_pair_counts'], [[9] * 4] * 4)
            # A second replot must work from the enriched JSON without legacy metadata.
            (root / 'finalists.json').unlink()
            (root / 'sweep_config.json').unlink()
            exploit.replot_crossplay(source)
            for name in ('crossplay.png', 'crossplay_runner.png', 'crossplay_configs.png',
                         'crossplay_configs_runner.png'):
                self.assertTrue((root / name).is_file())


if __name__ == '__main__':
    unittest.main()
