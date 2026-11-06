"""Late-slot collection, public clocks and a capacity-matched clock ablation."""
import json

import numpy as np
import pytest

from advice_audit.counterfactual import collect
from advice_audit.evaluation import Gates, episode
from advice_audit.features import arrival_features
from advice_audit.model import fit_models
from advice_audit.protocol import make_split
from advice_audit.report import crossed_summary


def test_clock_depends_only_on_public_budget_and_requires_consistent_inputs():
    obs, plan = np.zeros(12), np.zeros((12, 3))
    early = arrival_features(obs, obs, plan, 0, step=0, horizon=80)
    late = arrival_features(obs, obs, plan, 0, step=72, horizon=80)
    np.testing.assert_array_equal(early[:34], late[:34])
    np.testing.assert_allclose(early[34:], [1, 1])
    np.testing.assert_allclose(late[34:], [.1, 1])
    with pytest.raises(ValueError, match="Clock"):
        arrival_features(obs, obs, plan, 0, step=80, horizon=80)
    with pytest.raises(ValueError, match="Clock"):
        arrival_features(obs, obs, plan, 0, horizon=80)


@pytest.mark.parametrize("backend", ["toy", "fetch"])
def test_aligned_collection_includes_valid_late_slots_and_explicit_truncated_windows(tmp_path, backend):
    path = tmp_path/"pairs.npz"
    collect(path, backend=backend, episodes=5, horizon=80, delays=[0, 4, 12],
            request_period=24, audit_deadline=True, include_clock=True)
    with np.load(path, allow_pickle=False) as data:
        assert len(data["gain"]) == 55
        assert data["arrival"].shape == (55, 36)
        for seed in range(100, 105):
            for delay in [0, 4, 12]:
                mask = (data["episode"] == seed) & (data["delay"] == delay)
                expected = [t for t in [0, 24, 48, 72] if t+delay < 80]
                assert data["timestamp"][mask].tolist() == expected
        np.testing.assert_allclose(data["arrival"][:, 34], (80-data["arrival_step"])/80)
        assert set(data["short_window_steps"].tolist()) == {4, 8, 20, 24}
        assert np.isfinite(data["gain_deadline"]).all()
        for group, t in set(zip(data["episode"].tolist(), data["timestamp"].tolist())):
            mask = (data["episode"] == group) & (data["timestamp"] == t)
            assert len(set(data["plan_id"][mask])) == 1


@pytest.fixture
def clock_models(tmp_path):
    data = tmp_path/"pairs.npz"
    collect(data, episodes=20, horizon=40, delays=[0, 4], chunk=4,
            request_period=12, audit_deadline=True, include_clock=True)
    split = tmp_path/"split.json"
    split.write_text(json.dumps(make_split(np.arange(100, 120))))
    models = {}
    for mode in ("no_clock", "with_clock"):
        path = tmp_path/mode
        fit_models(data, path, seed=3, target="gain_deadline", split_manifest=split, feature_set=mode)
        models[mode] = Gates(path)
    return models


def test_clock_ablation_matches_random_projections_and_inference(clock_models):
    plain, clock = clock_models["no_clock"], clock_models["with_clock"]
    np.testing.assert_array_equal(plain.arrival.proj, clock.arrival.proj)
    np.testing.assert_array_equal(plain.arrival.bias, clock.arrival.bias)
    obs, plan = np.zeros(12), np.zeros((4, 3))
    for mode, gate in clock_models.items():
        feature = arrival_features(obs, obs, plan, 4, step=36, horizon=40)
        if mode == "no_clock":
            feature[34:] = 0
        mean, std = gate.arrival.predict(feature)
        assert gate.lower(obs, obs, plan, 4, step=36, horizon=40) == pytest.approx(mean[0]-.5*std[0])
    with pytest.raises(ValueError, match="clock"):
        clock.lower(obs, obs, plan, 4)


@pytest.mark.parametrize("mode", ["no_clock", "with_clock"])
def test_clock_gate_audit_is_passive(clock_models, mode):
    options = dict(backend="toy", method="veto_calibrated", seed=9331, delay=4,
                   gates=clock_models[mode], horizon=40, chunk=4, fixed_schedule=True, period=12)
    plain, _ = episode(**options)
    audited, _ = episode(**options, audit=True, audit_deadline=True)
    for key in ("utility", "mean_reward", "success", "requests", "accepts", "rejects"):
        assert plain[key] == audited[key]


def test_crossed_bootstrap_resamples_paired_contrasts_as_a_matrix():
    result = crossed_summary(np.full((5, 50), -.01))
    assert result["mean"] == pytest.approx(-.01)
    np.testing.assert_allclose(result["ci95"], [-.01, -.01])
    assert result["model_seeds"] == 5 and result["episodes"] == 50
    with pytest.raises(ValueError, match="matrix"):
        crossed_summary(np.ones(50))
