"""Research validity: horizon alignment, calibration isolation and audit passivity."""
import json

import numpy as np
import pytest

from advice_audit.counterfactual import collect, paired_gain, paired_outcomes
from advice_audit.envs import make_env
from advice_audit.evaluation import Gates, episode
from advice_audit.features import arrival_columns, arrival_features
from advice_audit.model import fit_models
from advice_audit.policies import expert_plan, local_action
from advice_audit.protocol import canonical_hash, load_split, make_split


@pytest.mark.parametrize("backend", ["toy", "fetch"])
def test_deadline_gain_matches_actual_episode_difference_and_preserves_live_state(backend):
    env = make_env(backend, seed=31, horizon=32, shift_at=17)
    try:
        past = []
        for _ in range(7):
            _, reward, _, _ = env.step(local_action(env.observe(), backend))
            past.append(reward)
        before = env.observe().copy()
        plan = expert_plan(before, horizon=6, backend=backend)
        outcomes = paired_outcomes(env, plan, horizon=8)
        assert outcomes["gain"] == pytest.approx(paired_gain(env, plan, horizon=8), abs=1e-10)
        utilities = []
        for candidate in (None, plan):
            branch = env.clone()
            try:
                rewards = past.copy()
                for k in range(25):
                    action = candidate[k] if candidate is not None and k < len(candidate) else local_action(branch.observe(), backend)
                    rewards.append(branch.step(action)[1])
                utilities.append(np.mean(rewards) + float(branch.success))
            finally:
                branch.close()
        assert outcomes["gain_deadline"] == pytest.approx(utilities[1]-utilities[0], abs=1e-10)
        assert outcomes["control"]["steps"] == outcomes["treatment"]["steps"] == 25
        np.testing.assert_array_equal(before, env.observe())
        assert env.t == 7
    finally:
        env.close()


def test_four_way_manifest_rejects_tampering_and_cross_partition_episodes(tmp_path):
    groups = np.repeat(np.arange(40), 3)
    manifest = make_split(np.arange(40), seed=5)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    masks, _ = load_split(path, groups)
    assert sorted(int(mask.sum()) for mask in masks.values()) == [12, 18, 18, 72]
    assert np.all(sum(masks.values()) == 1)
    manifest["group_ids"]["test"][0] = manifest["group_ids"]["train"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_split(path, groups)
    manifest["sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "sha256"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint"):
        load_split(path, groups)


def test_validation_and_test_labels_cannot_change_fit_or_conformal_offset(tmp_path):
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(40), 3)
    split = make_split(np.arange(40))
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(split))
    payload = dict(query=rng.normal(size=(120, 18)), arrival=rng.normal(size=(120, 34)),
                   gain=rng.normal(size=120), gain_deadline=rng.normal(size=120), episode=groups,
                   metadata=json.dumps(dict(backend="toy", seed_start=0, seed_end=39, chunk=12)))
    data = tmp_path / "data.npz"
    np.savez(data, **payload)
    a = fit_models(data, tmp_path / "a", split_manifest=manifest, target="gain_deadline")
    changed = np.isin(groups, split["group_ids"]["validation"] + split["group_ids"]["test"])
    payload["gain_deadline"][changed] += 1000
    np.savez(data, **payload)
    b = fit_models(data, tmp_path / "b", split_manifest=manifest, target="gain_deadline")
    assert a["conformal_offset"] == b["conformal_offset"]
    for model in ("query", "arrival"):
        with np.load(tmp_path / f"a/{model}.npz") as first, np.load(tmp_path / f"b/{model}.npz") as second:
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
    assert a["test"]["gain_mae"] != b["test"]["gain_mae"]


@pytest.mark.parametrize("backend", ["toy", "fetch"])
@pytest.mark.parametrize("method", ["always", "reject"])
def test_full_deadline_audit_cannot_change_actions_and_rejection_accounting(backend, method):
    options = dict(backend=backend, method=method, seed=77, delay=3, horizon=28,
                   chunk=4, period=8, fixed_schedule=True)
    plain, plain_trace = episode(**options)
    audited, trace = episode(**options, audit=True, audit_deadline=True)
    for field in ("success", "final_distance", "mean_reward", "requests", "accepts", "rejects", "utility"):
        assert plain[field] == audited[field]
    assert [(t["t"], t["event"]) for t in trace] == [(t["t"], t["event"]) for t in plain_trace]
    arrivals = [t for t in trace if t["event"] != "query"]
    assert audited["audited_candidates"] == len(arrivals) == audited["requests"]
    assert audited["deadline_positive_candidates"] == sum(t["audit_gain_deadline"] > 1e-6 for t in arrivals)
    if method == "reject":
        assert audited["deadline_missed_positive"] == audited["deadline_positive_candidates"]
        assert audited["accepts"] == audited["audited_accepts"] == 0
    else:
        assert audited["deadline_missed_positive"] == 0
        assert audited["audited_accepts"] == len(arrivals)


def test_full_audit_retains_same_plan_identity_across_delays(tmp_path):
    path = tmp_path / "pairs.npz"
    collect(path, episodes=5, horizon=36, delays=[0, 2, 4], rollout_horizon=8, audit_deadline=True)
    with np.load(path, allow_pickle=False) as data:
        for ep, timestamp in set(zip(data["episode"].tolist(), data["timestamp"].tolist())):
            mask = (data["episode"] == ep) & (data["timestamp"] == timestamp)
            assert len(set(data["plan_id"][mask])) == 1
            assert set(data["delay"][mask]) == {0, 2, 4}
        np.testing.assert_allclose(data["gain_deadline"], data["treatment_deadline_contribution"] - data["control_deadline_contribution"])


@pytest.mark.parametrize("mode", ["full", "no_age", "no_candidate", "current_only"])
def test_gate_inference_uses_same_ablation_as_training(tmp_path, mode):
    path = tmp_path / "pairs.npz"
    collect(path, episodes=20, horizon=32, delays=[0, 4], rollout_horizon=8, audit_deadline=True)
    split = tmp_path / "split.json"
    split.write_text(json.dumps(make_split(np.arange(100, 120))))
    model = tmp_path / "model"
    report = fit_models(path, model, split_manifest=split, target="gain_deadline", feature_set=mode, linear=True)
    gate = Gates(model)
    obs = np.zeros(12)
    plan = np.zeros((12, 3))
    features = arrival_features(obs, obs, plan, 4)[arrival_columns(mode)]
    mean, _ = gate.arrival.predict(features)
    assert gate.lower(obs, obs, plan, 4) == mean[0]
    assert report["group_ids"]["calibration"] != report["group_ids"]["validation"]
