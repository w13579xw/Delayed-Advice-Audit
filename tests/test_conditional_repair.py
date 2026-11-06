import importlib.util
from pathlib import Path
import sys

import numpy as np

from advice_audit.envs import ToyReachEnv

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
from audit_conditional_repair import future_snapshot, summarize_worlds


def test_resampling_changes_only_unrevealed_future():
    env = ToyReachEnv(seed=4, horizon=80, shift_at=40)
    original, obs = env.snapshot(), env.observe().copy()
    sampled = future_snapshot(original, np.random.default_rng(10))
    assert 20 <= sampled["shift_at"] < 60
    assert not np.array_equal(sampled["goal_shift"], original["goal_shift"])
    assert np.isclose(np.linalg.norm(sampled["goal_shift"]), .15)
    env.restore(sampled)
    assert np.array_equal(obs, env.observe())
    assert original["shift_at"] == 40
    original["t"] = 41
    past = future_snapshot(original, np.random.default_rng(22))
    assert np.array_equal(original["goal_shift"], past["goal_shift"])
    assert past["shift_at"] == 40


def test_future_oracle_can_exceed_out_of_future_choice():
    values = np.zeros((1, 4, 3, 5))
    values[0, :2, 0, 1] = 1.
    values[0, 2:, 0, 2] = 1.
    result = summarize_worlds(values)
    assert result["full_realized"][0] == 1.
    assert result["full_sample_conditional_max"][0] == .5
    assert result["full_crossfit"][0] == 0.
    assert result["cheap_realized"][0] == 0.
