import json
from pathlib import Path

import numpy as np
import pytest

from advice_audit.policies import expert_plan
from advice_audit.protocol import canonical_hash
from advice_audit.repair import local_nominal_plan
from advice_audit.repair_selector import RepairSelector, fit_selector, option_mask

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fitted(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    from repair_value_pilot import collect_phase
    c = json.loads((ROOT/"configs/repair_selector_study_v1.json").read_text())
    c.update(backend="toy", train_seed=810000, train_episodes=5, validation_seed=820000,
             validation_episodes=5, calibration_seed=830000, calibration_episodes=5)
    (tmp_path/"protocol.json").write_text(json.dumps(dict(config=c, sha256=canonical_hash(c))))
    for role in ["train", "validation", "calibration"]:
        collect_phase(tmp_path, role)
    report = fit_selector(tmp_path, tmp_path/"model", 0, c)
    return tmp_path, c, report


def test_selector_training_is_independent_of_test_and_cheap_bound_is_not_handicapped(fitted):
    root, c, first = fitted
    assert first["cheap_offset"] <= first["offset"]
    np.savez(root/"test.npz", gain=np.full((2, 2, 5), 100000))
    second = fit_selector(root, root/"model_again", 0, c)
    assert first == second
    for index in range(12):
        with np.load(root/f"model/option_{index}.npz") as a, np.load(root/f"model_again/option_{index}.npz") as b:
            np.testing.assert_array_equal(a["weights"], b["weights"])


def test_cheap_only_selector_cannot_read_expert_content(fitted):
    root, _, _ = fitted
    selector = RepairSelector(root/"model")
    obs = np.zeros(12)
    obs[6] = .2
    plans = [expert_plan(obs), local_nominal_plan(obs), local_nominal_plan(obs, gain=12.)]
    changed = [np.ones((12, 3)), plans[1], plans[2]]
    for calibrated in [False, True]:
        first = selector.choose(obs, obs, plans, 4, 4, 80, calibrated=calibrated, cheap_only=True)
        second = selector.choose(obs, obs, changed, 4, 4, 80, calibrated=calibrated, cheap_only=True)
        assert first == second
        assert first[0] in (None, 1, 2)


def test_invalid_or_duplicate_suffix_options_are_removed_without_using_labels():
    mask = option_mask([0, 4, 12], 12)
    assert mask.shape == (3, 12)
    np.testing.assert_array_equal(mask[:, 2], [False, True, False])
    np.testing.assert_array_equal(mask[:, 3], [True, True, False])
    np.testing.assert_array_equal(mask[:, :4], mask[:, 4:8])
    np.testing.assert_array_equal(mask[:, :4], mask[:, 8:])
