import numpy as np
import pytest

from advice_audit.metaworld_adapter import BCLocal, ContactEnv, TASKS, expert_for, expert_action


@pytest.mark.parametrize("task", TASKS)
def test_contact_snapshot_replays_physics_and_history(task):
    env = ContactEnv(task, 960000)
    policy = expert_for(task)
    try:
        for _ in range(100):
            env.step(expert_action(policy, env.observe()))
        snapshot = env.snapshot()
        before = env.observe()
        first = []
        for _ in range(25):
            reward = env.step(expert_action(policy, env.observe()))
            first.append((env.observe(), reward))
        env.restore(snapshot)
        assert np.array_equal(env.observe(), before)
        assert env.t == 100
        for obs, reward in first:
            again = env.step(expert_action(policy, env.observe()))
            assert np.allclose(obs, env.observe(), atol=1e-10, rtol=0)
            assert abs(reward-again) < 1e-10
        assert env.t == 125
    finally:
        env.close()


def test_policy_boundary_copies_mutating_expert_input():
    class Mutator:
        def get_action(self, obs):
            obs[:] = -100
            return np.ones(4)*3
    obs = np.ones(39)
    assert np.array_equal(expert_action(Mutator(), obs), np.ones(4))
    assert np.array_equal(obs, np.ones(39))


def test_bc_policy_uses_only_observation_and_saved_demonstrations():
    obs = np.zeros((4, 39))
    obs[:, 0] = np.arange(4)*.1
    actions = np.arange(16).reshape(4, 4)/16
    model = BCLocal(obs, actions, k=1)
    assert np.allclose(model.action(obs[2]), actions[2])
    with pytest.raises(ValueError):
        model.action(np.zeros(12))
