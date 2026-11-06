"""Protocol guardrails and validation-only model selection."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("research_pilot", ROOT / "scripts/research_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_uses_real_validation_utility_and_conservative_tie_break(monkeypatch, tmp_path):
    module = runner(monkeypatch)
    candidates = []
    for index, (value, utility, accepts) in enumerate([(0, 0.1, 2), (4, 0.4, 2), (-1, 0.4, 0)]):
        path = tmp_path/f"validation_{index}.csv"
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "utility", "accepts"])
            writer.writeheader()
            writer.writerow(dict(method="ttl", utility=utility, accepts=accepts))
            writer.writerow(dict(method="local", utility=10, accepts=0))
        candidates.append((value, path, "ttl"))
    selected = module.select_threshold(candidates)
    assert selected["selected"]["value"] == -1
    assert selected["selected"]["utility"] == 0.4


def test_research_protocol_rejects_overlapping_pools_and_schedule(monkeypatch):
    module = runner(monkeypatch)
    config = json.loads((ROOT/"configs/state_pilot_v2.json").read_text())
    module.validate_config(config)
    config["test_seed"] = config["validation_seed"]
    with pytest.raises(ValueError, match="overlap"):
        module.validate_config(config)
    config["test_seed"] = 22000
    config["period"] = 12
    with pytest.raises(ValueError, match="overlap"):
        module.validate_config(config)


def test_fixed_action_cost_repricing_matches_recorded_utility_and_equal_budget_contrast():
    spec = importlib.util.spec_from_file_location("compare_research", ROOT/"scripts/compare_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for seed in [1, 2]:
        for method, reward, requests in [("local", -.1, 0), ("ttl", -.12, 4), ("veto_only", -.11, 4)]:
            rows.append(dict(seed=str(seed), method=method, delay="4", mean_reward=str(reward),
                             success="1", query_cost="0.01", requests=str(requests),
                             utility=str(reward+1-.01*requests)))
    actual = module.reprice(rows, "veto_only", "local", 4, .01)
    free = module.reprice(rows, "veto_only", "local", 4, 0)
    assert actual["paired_utility_delta"] == pytest.approx(-.05)
    assert free["paired_utility_delta"] == pytest.approx(-.01)
    assert module.reprice(rows, "veto_only", "ttl", 4, 0)["paired_utility_delta"] == pytest.approx(
        module.reprice(rows, "veto_only", "ttl", 4, .05)["paired_utility_delta"])
