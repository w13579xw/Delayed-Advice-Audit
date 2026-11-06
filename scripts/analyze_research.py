"""Audit a frozen state pilot and report evidence without a success requirement."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from advice_audit.calibration import group_coverage
from advice_audit.model import GainModel
from advice_audit.protocol import canonical_hash, load_split
from advice_audit.report import paired_ci


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decision_metrics(accepted, gain, groups, epsilon):
    positive, harmful = gain > epsilon, gain < -epsilon
    value = np.where(accepted, gain, 0)
    per_episode = [float(value[groups == group].mean()) for group in np.unique(groups)]
    oracle = np.maximum(gain, 0)
    return dict(rows=len(gain), episodes=len(per_episode), accepts=int(accepted.sum()),
                harmful_accept_fraction=float(harmful[accepted].mean()) if accepted.any() else None,
                missed_positive_fraction=float((~accepted)[positive].mean()) if positive.any() else None,
                mean_single_intervention_gain=float(value.mean()),
                mean_single_intervention_gain_ci95=paired_ci(per_episode),
                oracle_mean_single_intervention_gain=float(oracle.mean()),
                mean_oracle_regret=float((oracle-value).mean()),
                note="Means over local-only candidate labels; not a sum of sequential policy gains")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    protocol = load(root / "protocol.json")
    c = protocol["config"]
    assert canonical_hash(c) == protocol["sha256"]
    state = load(root / "run.json")
    assert state["protocol_sha256"] == protocol["sha256"]
    selection = load(root / "selection.json")
    result = dict(backend=c["backend"], protocol_sha256=protocol["sha256"],
                  horizon_audit={}, offline_test={}, online={},
                  selection=selection, limitations=c["limitations"])
    with np.load(root / "pairs.npz", allow_pickle=False) as data:
        masks, split = load_split(root / "split_manifest.json", data["episode"])
        groups = data["episode"]
        assert set(groups.tolist()) == set(range(c["collection_seed"], c["collection_seed"]+c["episodes"]))
        n_times = len(range(0, c["horizon"]-max(c["delays"])-c["rollout_horizon"], c["interval"]))
        assert len(data["gain"]) == c["episodes"] * n_times * len(c["delays"])
        for key in ("gain", "gain_deadline", "query", "arrival"):
            assert np.isfinite(data[key]).all()
        np.testing.assert_allclose(data["gain_deadline"], data["treatment_deadline_contribution"]-data["control_deadline_contribution"], rtol=0, atol=1e-12)
        keys = set(zip(groups.tolist(), data["timestamp"].tolist()))
        flips = {"short": 0, "deadline": 0}
        for ep, time in keys:
            mask = (groups == ep) & (data["timestamp"] == time)
            assert len(set(data["plan_id"][mask])) == 1
            assert set(data["delay"][mask]) == set(c["delays"])
            for name, field in [("short", "gain"), ("deadline", "gain_deadline")]:
                values = data[field][mask]
                flips[name] += int((values > c["epsilon"]).any() and (values < -c["epsilon"]).any())
        for name, mask in [("all", np.ones(len(groups), dtype=bool)), ("offline_test", masks["test"])]:
            short, deadline = data["gain"][mask], data["gain_deadline"][mask]
            ep = groups[mask]
            short_pos, deadline_pos = short > c["epsilon"], deadline > c["epsilon"]
            result["horizon_audit"][name] = dict(rows=len(short),
                short_positive=int(short_pos.sum()), deadline_positive=int(deadline_pos.sum()),
                short_positive_fraction=float(short_pos.mean()), deadline_positive_fraction=float(deadline_pos.mean()),
                short_positive_deadline_nonpositive=int((short_pos & ~deadline_pos).sum()),
                short_positive_deadline_nonpositive_fraction=float((~deadline_pos)[short_pos].mean()) if short_pos.any() else None,
                terminal_success_improvements=int((data["success_gain"][mask] > 0).sum()),
                terminal_success_harms=int((data["success_gain"][mask] < 0).sum()),
                mean_deadline_gain=float(deadline.mean()),
                oracle_mean_deadline_gain=float(np.maximum(deadline, 0).mean()),
                oracle_mean_deadline_gain_ci95=paired_ci([np.maximum(deadline[ep == g], 0).mean() for g in np.unique(ep)]))
        result["horizon_audit"].update(request_groups=len(keys), sign_flip_groups=flips, epsilon=c["epsilon"])
        for name in ("all", "offline_test"):
            stats = result["horizon_audit"][name]
            stats["oracle_net_after_one_forced_query_cost"] = stats["oracle_mean_deadline_gain"]-c["cost"]
            stats["oracle_net_ci95"] = [value-c["cost"] for value in stats["oracle_mean_deadline_gain_ci95"]]
            stats["cost_interpretation"] = "Optimistic one-shot bound on this local-only candidate population; not a bound for adaptive multi-intervention policies"
        test = masks["test"]
        gain, test_groups = data[c["target"]][test], groups[test]
        dataset_hash = hashlib.sha256((root/"pairs.npz").read_bytes()).hexdigest()
        for mode in c["feature_models"]:
            model_dir = root/"models"/mode
            training = load(model_dir/"training.json")
            assert training["split_manifest_sha256"] == split["sha256"]
            assert training["dataset_sha256"] == dataset_hash and training["target"] == c["target"]
            model = GainModel.load(model_dir/"arrival.npz")
            features = data["arrival"][test][:, training["arrival_feature_indices"]]
            mean, std = model.predict(features)
            score = mean-training["uncertainty_penalty"]*std
            offset = training["conformal_offset"]
            lower = score-(offset if offset is not None else float("inf"))
            point = decision_metrics(score > 0, gain, test_groups, c["epsilon"])
            calibrated = decision_metrics(lower > 0, gain, test_groups, c["epsilon"])
            calibrated["offline_episode_lower_coverage"] = group_coverage(lower, gain, test_groups)
            result["offline_test"][mode] = dict(point=point, calibrated=calibrated,
                gain_mae=float(np.abs(mean-gain).mean()), conformal_offset=offset)
        for name, feature in [("ttl", "delay"), ("drift", "drift")]:
            threshold = selection["selections"][name]["selected"]["value"]
            result["offline_test"][name] = decision_metrics(data[feature][test] <= threshold, gain, test_groups, c["epsilon"])
        result["offline_test"]["oracle"] = decision_metrics(gain > 0, gain, test_groups, c["epsilon"])
        result["partitions"] = {k: dict(ids=split["group_ids"][k], rows=int(v.sum())) for k, v in masks.items()}
    for scenario, shift in c["test_scenarios"].items():
        assert datetime.fromisoformat(selection["selected_utc"]) <= datetime.fromisoformat(state["phases"][scenario]["started_utc"])
        folder = root/scenario
        rows = list(csv.DictReader((folder/"episodes.csv").open()))
        manifest = load(folder/"manifest.json")
        traces = [json.loads(line) for line in (folder/"traces.jsonl").read_text().splitlines()]
        expected = len(c["delays"])*len(c["test_methods"])*c["test_episodes"]
        assert len(rows) == len(traces) == expected
        assert len({(r["method"], r["seed"], r["delay"]) for r in rows}) == expected
        assert manifest["config"]["shift_scale"] == shift
        assert manifest["config"]["audit"] and manifest["config"]["audit_deadline"]
        assert set(r["method"] for r in rows) == set(c["test_methods"])
        assert set(int(r["seed"]) for r in rows) == set(range(c["test_seed"], c["test_seed"]+c["test_episodes"]))
        assert all(np.isfinite(float(r["utility"])) for r in rows)
        for row, trace in zip(rows, traces):
            assert (row["method"], int(row["seed"]), int(row["delay"])) == (trace["method"], trace["seed"], trace["delay"])
            queries = [e["t"] for e in trace["events"] if e["event"] == "query"]
            expected_queries = [] if row["method"] == "local" else [t for t in range(0, c["horizon"], c["period"]) if t+int(row["delay"]) < c["horizon"]]
            assert queries == expected_queries
            assert int(row["requests"]) == int(row["audited_candidates"]) == len(queries)
            assert int(row["accepts"]) + int(row["rejects"]) == len(queries)
            assert int(row["audited_accepts"]) == int(row["accepts"])
            assert all(e.get("audit_gain_deadline") is not None for e in trace["events"] if e["event"] != "query")
        online = load(folder/"summary.json")
        # Report empirical per-episode joint coverage separately from point coverage.
        coverage = {}
        for method in ("veto_only", "veto_calibrated"):
            covered = []
            for seed in range(c["test_seed"], c["test_seed"]+c["test_episodes"]):
                events = [e for t in traces if t["method"] == method and t["seed"] == seed for e in t["events"] if e["event"] != "query"]
                covered.append(all(e["gate_lower_infinite"] or e["gate_lower"] <= e["audit_gain_deadline"] for e in events))
            coverage[method] = dict(episodes=len(covered), empirical_episode_joint_coverage=float(np.mean(covered)),
                                    note="Empirical adaptive-policy, all evaluated delays; no conformal guarantee")
        result["online"][scenario] = dict(rows=len(rows), shift_scale=shift, summary=online, coverage=coverage)
    checks = dict(checks="passed", backend=c["backend"], labels=sum(v["rows"] for v in result["partitions"].values()),
                  partitions={k: len(v["ids"]) for k, v in result["partitions"].items()},
                  scenario_rows={k: v["rows"] for k, v in result["online"].items()},
                  fixed_requests_and_timestamps="matched", calibration_and_selection="independent",
                  note="Integrity and protocol checks; scientific superiority is not required")
    (root/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (root/"checks.json").write_text(json.dumps(checks, indent=2))
    lines = ["# State-diagnostic falsification pilot", "", "Development evidence only: one gate training seed, analytic policies, no visual benchmark.",
             "", "## Label horizon audit", "", "```json", json.dumps(result["horizon_audit"], indent=2), "```",
             "", "## Frozen partitions", "", "```json", json.dumps(result["partitions"], indent=2), "```",
             "", "## Validation-selected thresholds", "",
             f"TTL={selection['selections']['ttl']['selected']['value']}; drift={selection['selections']['drift']['selected']['value']}. -1 means reject all.",
             "", "## Offline test: isolated intervention labels", "",
             "Each row averages per-candidate outcomes on local-only trajectories. Not a sequential oracle or task success score.", "",
             "|Model|Accepts|Harmful among accepted|Missed positive|Mean gain vs reject|Episode-bootstrap 95% CI|",
             "|---|---:|---:|---:|---:|---|"]
    for name, stats in result["offline_test"].items():
        row = stats.get("point", stats)
        lines.append(f"|{name}|{row['accepts']}|{row['harmful_accept_fraction']}|{row['missed_positive_fraction']}|{row['mean_single_intervention_gain']:.7f}|{row['mean_single_intervention_gain_ci95']}|")
    for scenario, online in result["online"].items():
        lines += ["", f"## Closed loop: {scenario}, shift={online['shift_scale']}", "",
                  "Harmful/missed fractions below use deadline labels at epsilon=1e-6; null means no denominator.",
                  "Local-only continuations in the audit are not a sequential adaptive-policy oracle.", "",
                  "|Delay|Method|Success|Utility delta vs local|95% paired CI|Mean requests|Mean accepts|Harmful accepted|Missed positive|",
                  "|---:|---|---:|---:|---|---:|---:|---:|---:|"]
        for r in online["summary"]:
            lines.append(f"|{r['delay']}|{r['method']}|{r['success']:.3f}|{r['paired_utility_delta']:.6f}|{r['utility_delta_ci95']}|{r['mean_queries']:.2f}|{r['mean_accepts']:.2f}|{r['deadline_harmful_accept_fraction']}|{r['deadline_missed_positive_fraction']}|")
        lines += ["", "Empirical closed-loop episode coverage: `"+json.dumps(online["coverage"])+"`."]
    lines += ["", "## Interpretation limits", "",
              "Thresholds selected only on closed-loop validation. Feature models were declared before collection; no best-test-model promotion.",
              "The no_age ablation removes explicit delay, but motion/history can still proxy age.",
              "Offline test label bootstrap clusters by episode. Closed-loop CIs pair environment seeds within delay and scenario; they exclude training-seed variation.",
              "Clean and shifted scenarios use paired seeds; this is not task OOD or an independent population replicate.",
              "No theorem of safe acceptance follows from offline or empirical closed-loop coverage."]
    (root/"analysis.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
