import numpy as np
import pytest

from advice_audit.counterfactual import paired_outcomes
from advice_audit.envs import make_env
from advice_audit.policies import expert_plan, local_action
from advice_audit.repair import audited_return, local_nominal_plan, nominal_reference, repair_action


def test_repair_modes_obey_expiry_and_preserve_inputs():
    obs = np.zeros(12)
    obs[6] = .2
    plan = expert_plan(obs)
    before = plan.copy()
    reference = nominal_reference(obs, plan, "toy")
    assert repair_action(obs, plan, reference, 12, 0, "suffix") is None
    assert repair_action(obs, plan, reference, 12, 0, "track") is None
    assert repair_action(obs, plan, reference, 0, 3, "prefix3") is None
    np.testing.assert_array_equal(repair_action(obs, plan, reference, 4, 0, "suffix"), plan[4])
    np.testing.assert_array_equal(before, plan)
    assert local_nominal_plan(obs).shape == (12, 3)
    np.testing.assert_array_equal(obs[:3], np.zeros(3))


@pytest.mark.parametrize("backend", ["toy", "fetch"])
def test_raw_audit_matches_original_deadline_gain_and_does_not_mutate_live_env(backend):
    env = make_env(backend, seed=41, horizon=30, shift_at=13)
    try:
        requested = env.observe().copy()
        plan = expert_plan(requested, backend=backend)
        for _ in range(4):
            env.step(local_action(env.observe(), backend))
        before = env.observe().copy()
        control = audited_return(env, requested, plan, 4, "reject")
        raw = audited_return(env, requested, plan, 4, "raw")
        expected = paired_outcomes(env, plan)["gain_deadline"]
        assert raw["contribution"]-control["contribution"] == pytest.approx(expected, abs=1e-12)
        np.testing.assert_array_equal(before, env.observe())
        assert env.t == 4
        expired = audited_return(env, requested, plan, 12, "suffix")
        assert expired == control
    finally:
        env.close()


def test_feedback_tracking_uses_current_position_not_future_goal():
    requested = np.zeros(12)
    requested[6] = .1
    plan = expert_plan(requested)
    ref = nominal_reference(requested, plan, "toy")
    a = requested.copy()
    b = requested.copy()
    b[6:9] = 100
    np.testing.assert_array_equal(repair_action(a, plan, ref, 4, 0, "track"), repair_action(b, plan, ref, 4, 0, "track"))
