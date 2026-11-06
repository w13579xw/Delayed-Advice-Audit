"""Whole-horizon replay must be bitwise exact, also through contact phases."""
import numpy as np
import pytest
from advice_audit.metaworld_adapter import ContactEnv,expert_for,expert_action
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy

@pytest.mark.parametrize('task',['push-v3','drawer-open-v3','pick-place-v3'])
@pytest.mark.parametrize('seed',[4700000,4700007])
@pytest.mark.parametrize('cut',[16,64,192])
def test_full_native_tail_replays_bitwise(task,seed,cut):
    env=ContactEnv(task,seed)
    policy=SawyerPickPlaceV3Policy() if task=='pick-place-v3' else expert_for(task)
    try:
        for _ in range(cut):env.step(expert_action(policy,env.observe()))
        snapshot=env.snapshot()
        def tail():
            observations=[];rewards=[]
            while env.t<500:
                rewards.append(env.step(expert_action(policy,env.observe())))
                observations.append(env.observe())
            return np.asarray(observations),np.asarray(rewards),env.seen_success,env.last_success
        first=tail();env.restore(snapshot);second=tail()
        np.testing.assert_array_equal(first[0],second[0])
        np.testing.assert_array_equal(first[1],second[1])
        assert first[2:]==second[2:]
        # Repeating restoration must not consume/mutate the saved state.
        env.restore(snapshot);third=tail()
        np.testing.assert_array_equal(first[0],third[0])
        np.testing.assert_array_equal(first[1],third[1])
    finally:env.close()


def test_pick_place_native_reward_view_aliases_survive_restore():
    env=ContactEnv('pick-place-v3',4700000)
    try:
        assert np.shares_memory(env.env.init_left_pad,env.env.data.xpos)
        assert np.shares_memory(env.env.init_right_pad,env.env.data.xpos)
        snapshot=env.snapshot()
        env.step(np.array([.5,.5,-.5,1.]))
        env.restore(snapshot)
        assert np.shares_memory(env.env.init_left_pad,env.env.data.xpos)
        assert np.shares_memory(env.env.init_right_pad,env.env.data.xpos)
        assert not np.shares_memory(env.env.init_left_pad,snapshot['data'].xpos)
    finally:env.close()
