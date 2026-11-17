"""Regression checks for seeded stochastic evaluation and sweep safety."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
from opponents import PolicyOpponent
from algos.networks import GaussianActorCritic
from utils.evaluate import play_match
from exploit import nashconv

class EvaluationSafety(unittest.TestCase):
    def test_stochastic_reproducibility(self):
        torch.set_num_threads(1)
        p=GaussianActorCritic(14,5,np.zeros(5),np.ones(5),(8,))
        ctrl=PolicyOpponent(p)
        a=play_match(ctrl,ctrl,8,8,123)
        torch.randn(500)
        self.assertEqual(a,play_match(ctrl,ctrl,8,8,123))

    def test_protected_overrides_and_overlapping_seeds(self):
        with self.assertRaisesRegex(ValueError,'Forbidden'):
            nashconv('missing',br_args='--learner good')
        with self.assertRaisesRegex(ValueError,'overlap'):
            nashconv('missing',episodes=500)

    def test_nonfinite_actions_fail(self):
        def invalid(obs): return np.full((len(obs),5),np.nan)
        with self.assertRaises(FloatingPointError):
            play_match(invalid,invalid,2,2)

if __name__=='__main__': unittest.main()
