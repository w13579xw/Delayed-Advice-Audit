"""Behavioral checks for causal branching, delayed release, and data separation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from daci.calibration import group_lower_offset
from daci.counterfactual import paired_gain
from daci.envs import make_env
from daci.evaluation import Gates, episode
from daci.features import arrival_features, query_features
from daci.model import GainModel, group_split
from daci.policies import expert_plan


class RecordingEnv:
    """Delegate to a real diagnostic environment while retaining tick records."""

    def __init__(self):
        self.base = make_env("toy", seed=9, horizon=24, shift_at=10, shift_scale=0.0)
        self.records = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self, seed=None):
        self.records = []
        return self.base.reset(seed=seed)

    def step(self, action):
        self.records.append((self.base.t, self.base.observe().copy(), np.asarray(action).copy()))
        return self.base.step(action)


class FixedPredictor:
    def predict(self, feature):
        return np.array([0.2]), np.array([0.0])


class PipelineTests(unittest.TestCase):
    def test_fixed_schedule_keeps_requests_equal_after_rejection(self):
        accepted, ta = episode("toy", "always", seed=7, delay=4, horizon=60,
                               chunk=4, period=12, fixed_schedule=True)
        rejected, tr = episode("toy", "ttl", seed=7, delay=4, horizon=60,
                               chunk=4, period=12, fixed_schedule=True, ttl=0)
        qa = [r["t"] for r in ta if r["event"] == "query"]
        qr = [r["t"] for r in tr if r["event"] == "query"]
        self.assertEqual(qa, [0, 12, 24, 36, 48])
        self.assertEqual(qa, qr)
        self.assertEqual(rejected["accepts"], 0)
        self.assertEqual(accepted["requests"], rejected["requests"])

    def test_world_ticks_and_local_policy_continue_during_pending_request(self):
        env = RecordingEnv()
        with patch("daci.evaluation.make_env", return_value=env):
            result, trace = episode("toy", "always", seed=9, delay=6, horizon=24,
                                    chunk=3, budget=1, shift_scale=0)
        self.assertEqual([(r[0]) for r in env.records], list(range(24)))
        self.assertEqual(trace[0]["event"], "query")
        self.assertEqual(trace[0]["ready_at"], 6)
        self.assertEqual(trace[1]["event"], "accept")
        self.assertEqual(trace[1]["t"], 6)
        self.assertEqual(result["simulated_wait_steps"], 6)
        # The arm moved before advice arrived; this is not a paused simulation.
        self.assertGreater(np.linalg.norm(env.records[5][1][:3] - env.records[0][1][:3]), 1e-3)

    def test_zero_delay_can_release_and_execute_in_same_tick(self):
        result, trace = episode("toy", "always", seed=9, delay=0, horizon=24,
                                chunk=3, budget=1, shift_scale=0)
        self.assertEqual(trace[0]["t"], trace[1]["t"])
        self.assertEqual(result["simulated_wait_steps"], 0)
        self.assertEqual(result["requests"], 1)

    def test_query_cost_charged_once_per_request(self):
        low, _ = episode("toy", "always", seed=7, delay=2, horizon=30,
                          budget=2, chunk=4, cost=0.1)
        high, _ = episode("toy", "always", seed=7, delay=2, horizon=30,
                           budget=2, chunk=4, cost=0.5)
        self.assertEqual(low["requests"], 2)
        self.assertEqual(low["accepts"], high["accepts"])
        self.assertEqual(low["mean_reward"], high["mean_reward"])
        self.assertAlmostEqual(low["utility"] - high["utility"], 0.4 * low["requests"])

    def test_acceptance_uses_sunk_cost_rule(self):
        gates = object.__new__(Gates)
        gates.query = FixedPredictor()
        gates.arrival = FixedPredictor()
        gates.config = {"uncertainty_penalty": 0.5, "cost": 100.0,
                        "conformal_offset": 0.0}
        obs = np.zeros(12)
        plan = np.zeros((3, 3))
        self.assertFalse(gates.ask(obs, delay=2, chunk=3, cost=100.0))
        # Once paid, even a large original query cost must not veto positive gain.
        self.assertTrue(gates.accept(obs, obs, plan, delay=2))

    def test_counterfactual_labels_do_not_mutate_live_environment(self):
        env = make_env("toy", seed=2, horizon=30, shift_at=5)
        try:
            before = env.snapshot()
            gain = paired_gain(env, expert_plan(env.observe(), horizon=8), horizon=12)
            after = env.snapshot()
            self.assertTrue(np.isfinite(gain))
            for key in ("position", "velocity", "goal", "goal_shift", "last_action"):
                np.testing.assert_array_equal(before[key], after[key])
            self.assertEqual(before["t"], after["t"])
            self.assertEqual(before["hold_count"], after["hold_count"])
        finally:
            env.close()

    def test_offline_audit_cannot_change_online_decisions(self):
        plain, trace_plain = episode("toy", "always", seed=4, delay=5, horizon=30,
                                     chunk=4, budget=2, audit=False)
        audited, trace_audited = episode("toy", "always", seed=4, delay=5, horizon=30,
                                         chunk=4, budget=2, audit=True, label_horizon=8)
        for field in ("success", "final_distance", "requests", "accepts", "rejects", "utility"):
            self.assertEqual(plain[field], audited[field])
        self.assertEqual([(x["t"], x["event"]) for x in trace_plain],
                         [(x["t"], x["event"]) for x in trace_audited])

    def test_group_split_keeps_all_branches_of_episode_together(self):
        groups = np.repeat(np.arange(15), 7)
        masks, ids = group_split(groups, seed=11)
        self.assertTrue(np.all(np.sum(masks, axis=0) == 1))
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertFalse(set(ids[left]).intersection(ids[right]))
        for episode_id in np.unique(groups):
            partitions = [bool(np.any(mask[groups == episode_id])) for mask in masks]
            self.assertEqual(sum(partitions), 1)
        masks_again, _ = group_split(groups, seed=11)
        for first, second in zip(masks, masks_again):
            np.testing.assert_array_equal(first, second)

    def test_calibration_uses_group_max_and_infinite_small_sample_bound(self):
        predicted = np.array([0.1, 0.5, 0.3, 0.2, 0.4])
        target = np.zeros(5)
        groups = np.array([0, 0, 1, 2, 3])
        offset, scores = group_lower_offset(predicted, target, groups, alpha=0.25)
        np.testing.assert_allclose(scores, [0.5, 0.3, 0.2, 0.4])
        self.assertAlmostEqual(offset, 0.5)
        offset_small, _ = group_lower_offset(predicted[:3], target[:3], groups[:3], alpha=0.1)
        self.assertTrue(np.isposinf(offset_small))

    def test_calibrated_gate_rejects_when_sample_insufficient(self):
        gates = object.__new__(Gates)
        gates.arrival = FixedPredictor()
        gates.config = {"uncertainty_penalty": 0.5, "conformal_offset": None}
        obs = np.zeros(12)
        self.assertFalse(gates.accept(obs, obs, np.zeros((3, 3)), delay=2, calibrated=True))

    def test_model_preprocessing_fit_and_round_trip(self):
        rng = np.random.RandomState(12)
        train = rng.normal(size=(40, 4))
        labels = train[:, 0] - train[:, 1]
        model = GainModel(seed=1, width=8, members=3).fit(train, labels)
        np.testing.assert_allclose(model.mean, train.mean(axis=0))
        original_mean = model.mean.copy()
        extreme_test = np.full((4, 4), 1000.0)
        model.predict(extreme_test)
        np.testing.assert_array_equal(original_mean, model.mean)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            model.save(path)
            restored = GainModel.load(path)
            for first, second in zip(model.predict(train[:4]), restored.predict(train[:4])):
                np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_feature_functions_have_no_snapshot_dependency(self):
        obs = np.arange(12, dtype=np.float32) / 20
        requested = obs.copy()
        plan = np.zeros((4, 3))
        q = query_features(obs, delay=2, chunk=4)
        a = arrival_features(obs, requested, plan, delay=2)
        self.assertEqual(q.shape, (18,))
        self.assertEqual(a.shape, (34,))
        self.assertTrue(np.all(np.isfinite(a)))


if __name__ == "__main__":
    unittest.main()
