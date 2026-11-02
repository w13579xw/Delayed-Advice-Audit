"""Snapshot integrity and causal observation checks; stdlib unittest."""

import importlib.util
import unittest

import numpy as np

from daci.envs import make_env
from daci.policies import expert_plan, local_action


class ToyEnvironmentTests(unittest.TestCase):
    backend = "toy"

    def make(self, **kwargs):
        options = dict(backend=self.backend, seed=13, horizon=30, shift_at=12)
        options.update(kwargs)
        return make_env(**options)

    def test_restore_preserves_future_trajectory_and_goal_change(self):
        env = self.make()
        try:
            for _ in range(8):
                env.step(local_action(env.observe(), backend=self.backend))
            snap = env.snapshot()
            actions = np.random.RandomState(2).uniform(-1, 1, (13, 3))
            first = [env.step(a)[0] for a in actions]
            env.restore(snap)
            second = [env.step(a)[0] for a in actions]
            np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
        finally:
            env.close()

    def test_clone_is_independent(self):
        env = self.make()
        cloned = env.clone()
        try:
            before = env.observe().copy()
            cloned.step(np.ones(3))
            np.testing.assert_array_equal(before, env.observe())
            self.assertEqual(env.t, 0)
            self.assertEqual(cloned.t, 1)
        finally:
            env.close()
            cloned.close()

    def test_future_shift_not_exposed(self):
        first = self.make(shift_scale=0.0)
        second = self.make(shift_scale=0.3)
        try:
            np.testing.assert_array_equal(first.observe(), second.observe())
            for _ in range(11):
                a, b = first.step(np.zeros(3)), second.step(np.zeros(3))
                np.testing.assert_array_equal(a[0], b[0])
            a, b = first.step(np.zeros(3)), second.step(np.zeros(3))
            self.assertFalse(np.allclose(a[0][6:9], b[0][6:9]))
        finally:
            first.close()
            second.close()

    def test_episode_runs_to_deadline(self):
        env = self.make()
        try:
            for t in range(env.horizon):
                _, _, done, _ = env.step(local_action(env.observe(), backend=self.backend))
                self.assertEqual(done, t == env.horizon - 1)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(3))
        finally:
            env.close()

    def test_policies_do_not_mutate_observation(self):
        env = self.make()
        try:
            obs = env.observe()
            original = obs.copy()
            self.assertEqual(local_action(obs, backend=self.backend).shape, (3,))
            plan = expert_plan(obs, horizon=7, backend=self.backend)
            self.assertEqual(plan.shape, (7, 3))
            self.assertTrue(np.all(np.abs(plan) <= 1.0))
            np.testing.assert_array_equal(original, obs)
        finally:
            env.close()


@unittest.skipUnless(importlib.util.find_spec("gymnasium_robotics") is not None,
                     "optional MuJoCo dependencies not installed")
class FetchEnvironmentTests(ToyEnvironmentTests):
    backend = "fetch"


if __name__ == "__main__":
    unittest.main()
