"""Verify and summarize a frozen checkpoint-by-clock study without positive-result gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from advice_audit.model import GainModel
from advice_audit.protocol import canonical_hash, load_split
from advice_audit.report import crossed_summary


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_data(path, c, expected_seed, episodes, aligned):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    assert set(arrays["episode"].tolist()) == set(range(expected_seed, expected_seed+episodes))
    for field in ("query", "arrival", "gain", "gain_deadline"):
        assert np.isfinite(arrays[field]).all()
    assert arrays["arrival"].shape == (episodes*(11 if aligned else 15), 36)
    np.testing.assert_allclose(arrays["arrival"][:, 34], (c["horizon"]-arrays["arrival_step"])/c["horizon"])
    assert (arrays["arrival_step"] < c["horizon"]).all()
    keys = set(zip(arrays["episode"].tolist(), arrays["timestamp"].tolist()))
    for ep, time in keys:
        mask = (arrays["episode"] == ep) & (arrays["timestamp"] == time)
        expected_delays = {d for d in c["delays"] if time+d < c["horizon"]}
        assert set(arrays["delay"][mask].tolist()) == expected_delays
        assert len(set(arrays["plan_id"][mask])) == 1
    expected_times = [0, 24, 48, 72] if aligned else [0, 10, 20, 30, 40]
    assert set(arrays["timestamp"].tolist()) == set(expected_times)
    return arrays


def offline_metrics(model_dir, data, episodes, feature):
    training = read(model_dir/"training.json")
    x = data["arrival"][:, training["arrival_feature_indices"]].copy()
    if feature == "no_clock":
        x[:, 34:] = 0
    mean, std = GainModel.load(model_dir/"arrival.npz").predict(x)
    score = mean-training["uncertainty_penalty"]*std
    offset = training["conformal_offset"]
    lower = score-(offset if offset is not None else float("inf"))
    y, groups = data["gain_deadline"], data["episode"]
    metrics = {k: [] for k in ("mae", "point_value", "calibrated_value", "point_accept", "calibrated_accept", "coverage")}
    for ep in episodes:
        mask = groups == ep
        a, ac = score[mask] > 0, lower[mask] > 0
        metrics["mae"].append(float(np.abs(mean[mask]-y[mask]).mean()))
        metrics["point_value"].append(float(np.where(a, y[mask], 0).mean()))
        metrics["calibrated_value"].append(float(np.where(ac, y[mask], 0).mean()))
        metrics["point_accept"].append(float(a.mean()))
        metrics["calibrated_accept"].append(float(ac.mean()))
        metrics["coverage"].append(float(np.all(lower[mask] <= y[mask])))
    return metrics


def read_closed(folder, c, methods):
    rows = list(csv.DictReader((folder/"episodes.csv").open()))
    traces = [json.loads(line) for line in (folder/"traces.jsonl").read_text().splitlines()]
    expected = len(methods)*len(c["delays"])*c["closed_loop_episodes"]
    assert len(rows) == len(traces) == expected
    assert len({(r["method"], r["seed"], r["delay"]) for r in rows}) == expected
    assert set(r["method"] for r in rows) == set(methods)
    lookup = {}
    for row, trace in zip(rows, traces):
        key = (row["method"], int(row["seed"]), int(row["delay"]))
        assert key == (trace["method"], trace["seed"], trace["delay"])
        times = [e["t"] for e in trace["events"] if e["event"] == "query"]
        expected_times = [] if key[0] == "local" else [t for t in range(0, c["horizon"], c["period"]) if t+key[2] < c["horizon"]]
        assert times == expected_times
        assert int(row["requests"]) == int(row["audited_candidates"]) == len(times)
        assert int(row["accepts"])+int(row["rejects"]) == len(times)
        assert int(row["audited_accepts"]) == int(row["accepts"])
        assert np.isfinite(float(row["utility"]))
        lookup[key] = row
    return lookup, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    protocol = read(root/"protocol.json")
    c = protocol["config"]
    assert canonical_hash(c) == protocol["sha256"]
    result = dict(backend=c["backend"], config=c, offline={}, contrasts={}, closed={}, data={})
    all_models = {}
    for collection in c["collection_protocols"]:
        path = root/"data"/(collection+".npz")
        data = verify_data(path, c, c["collection_seed"], c["episodes"], collection == "aligned")
        _, split = load_split(root/"split_manifest.json", data["episode"])
        result["data"][collection] = dict(rows=len(data["gain"]), request_times=sorted(set(data["timestamp"].tolist())),
            truncated_short_windows=int((data["short_window_steps"] < c["rollout_horizon"]).sum()))
        dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for feature in c["feature_sets"]:
            for seed in c["model_seeds"]:
                name = f"{collection}_{feature}_{seed}"
                folder = root/"models"/name
                training = read(folder/"training.json")
                assert training["dataset_sha256"] == dataset_hash
                assert training["split_manifest_sha256"] == split["sha256"]
                assert training["seed"] == seed and training["feature_set"] == feature
                all_models[name] = folder
        for seed in c["model_seeds"]:
            a = GainModel.load(all_models[f"{collection}_no_clock_{seed}"]/"arrival.npz")
            b = GainModel.load(all_models[f"{collection}_with_clock_{seed}"]/"arrival.npz")
            np.testing.assert_array_equal(a.proj, b.proj)
            np.testing.assert_array_equal(a.bias, b.bias)
    test_ids = list(range(c["test_seed"], c["test_seed"]+c["test_episodes"]))
    closed_ids = test_ids[:c["closed_loop_episodes"]]
    for scenario in c["test_scenarios"]:
        data = verify_data(root/"data"/("test_"+scenario+".npz"), c, c["test_seed"], c["test_episodes"], True)
        result["data"]["test_"+scenario] = dict(rows=len(data["gain"]), episodes=len(test_ids))
        scenario_offline = {}
        matrices = {}
        for collection in c["collection_protocols"]:
            for feature in c["feature_sets"]:
                name = collection+"_"+feature
                metrics = [offline_metrics(all_models[name+"_"+str(seed)], data, test_ids, feature) for seed in c["model_seeds"]]
                matrices[name] = {k: np.array([m[k] for m in metrics]) for k in metrics[0]}
                scenario_offline[name] = {k: crossed_summary(v) for k, v in matrices[name].items()}
        result["offline"][scenario] = scenario_offline
        result["contrasts"][scenario] = {}
        for name, reference in [("aligned_with_clock", "aligned_no_clock"), ("aligned_with_clock", "early_with_clock")]:
            result["contrasts"][scenario][name+" minus "+reference] = {
                k: crossed_summary(matrices[name][k]-matrices[reference][k])
                for k in ("mae", "coverage", "point_value", "calibrated_value")}
        baseline, _ = read_closed(root/"baselines"/scenario, c, c["baseline_methods"])
        scenario_closed = {}
        closed_values = {}
        for feature in c["feature_sets"]:
            collections = []
            cov_by_seed = {"veto_only": [], "veto_calibrated": []}
            for seed in c["model_seeds"]:
                folder = root/"closed"/scenario/f"aligned_{feature}_{seed}"
                lookup, traces = read_closed(folder, c, ["veto_only", "veto_calibrated"])
                collections.append(lookup)
                for method in cov_by_seed:
                    covered = []
                    for ep in closed_ids:
                        events = [e for t in traces if t["method"] == method and t["seed"] == ep
                                  for e in t["events"] if e["event"] != "query"]
                        covered.append(float(all(e["gate_lower_infinite"] or e["gate_lower"] <= e["audit_gain_deadline"] for e in events)))
                    cov_by_seed[method].append(covered)
            feature_stats = {}
            for method in ("veto_only", "veto_calibrated"):
                for delay in c["delays"]:
                    selected = [[lookup[(method, ep, delay)] for ep in closed_ids] for lookup in collections]
                    utility = np.array([[float(r["utility"]) for r in row] for row in selected])
                    key = f"{method},delay={delay}"
                    closed_values[(feature, key)] = utility
                    reference = np.array([float(baseline[("local", ep, delay)]["utility"]) for ep in closed_ids])
                    reject = np.array([float(baseline[("reject", ep, delay)]["utility"]) for ep in closed_ids])
                    accept_n = sum(int(r["accepts"]) for row in selected for r in row)
                    harm_n = sum(int(r["deadline_harmful_accepts"]) for row in selected for r in row)
                    positive_n = sum(int(r["deadline_positive_candidates"]) for row in selected for r in row)
                    missed_n = sum(int(r["deadline_missed_positive"]) for row in selected for r in row)
                    feature_stats[key] = dict(vs_local=crossed_summary(utility-reference), vs_reject=crossed_summary(utility-reject),
                        success=crossed_summary([[int(r["success"]) for r in row] for row in selected]),
                        accepts=accept_n, harmful_accepts=harm_n,
                        harmful_fraction=harm_n/accept_n if accept_n else None,
                        positive_candidates=positive_n, missed_positive=missed_n,
                        missed_fraction=missed_n/positive_n if positive_n else None)
            feature_stats["empirical_episode_coverage"] = {k: crossed_summary(v) for k, v in cov_by_seed.items()}
            scenario_closed[feature] = feature_stats
        scenario_closed["clock_minus_no_clock"] = {
            key: crossed_summary(closed_values[("with_clock", key)]-closed_values[("no_clock", key)])
            for key in [f"{method},delay={delay}" for method in ("veto_only", "veto_calibrated") for delay in c["delays"]]}
        result["closed"][scenario] = scenario_closed
    checks = dict(checks="passed", backend=c["backend"], models=len(all_models),
        collection_rows={k: v["rows"] for k, v in result["data"].items()},
        closed_rows_per_scenario=c["closed_loop_episodes"]*len(c["delays"])*(len(c["baseline_methods"])+2*len(c["feature_sets"])*len(c["model_seeds"])),
        model_seeds=c["model_seeds"], offline_test_episodes=c["test_episodes"],
        note="Protocol integrity only; no positive model effect required")
    (root/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (root/"checks.json").write_text(json.dumps(checks, indent=2))
    lines = ["# 检查点协议与时钟控制实验", "", "五个模型训练seed共享采集/校准数据；CI交叉重采样训练seed和环境episode，不覆盖任务/采集数据总体。",
        "离线50 episode；闭环12 episode/延迟。只有aligned/shifted的离线检查点与扰动协议匹配；clean与自适应闭环仍仅作经验评估。", ""]
    for scenario, stats in result["offline"].items():
        lines += [f"## 离线 {scenario}", "", "|训练协议/特征|MAE|episode coverage|点估计候选收益|校准候选收益|点估计接受率|校准接受率|", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, metrics in stats.items():
            lines.append("|"+name+"|"+"|".join(f"{metrics[k]['mean']:.7f}" for k in ["mae", "coverage", "point_value", "calibrated_value", "point_accept", "calibrated_accept"])+"|")
        lines += ["", "预先指定配对对照（MAE负值较好，coverage/候选收益正值较好）：", "", "|对照|指标|差值|95% crossed CI|", "|---|---|---:|---|"]
        for name, metrics in result["contrasts"][scenario].items():
            for key, v in metrics.items():
                lines.append(f"|{name}|{key}|{v['mean']:.7f}|{v['ci95']}|")
        lines += ["", f"## 闭环 {scenario}", "", "|特征|方法/延迟|成功率|相对local效用|相对reject效用|95% CI vs reject|接受|有害|有害率|漏失率|", "|---|---|---:|---:|---:|---|---:|---:|---:|---:|"]
        for feature in c["feature_sets"]:
            for key, v in result["closed"][scenario][feature].items():
                if key == "empirical_episode_coverage":
                    continue
                lines.append(f"|{feature}|{key}|{v['success']['mean']:.3f}|{v['vs_local']['mean']:.7f}|{v['vs_reject']['mean']:.7f}|{v['vs_reject']['ci95']}|{v['accepts']}|{v['harmful_accepts']}|{v['harmful_fraction']}|{v['missed_fraction']}|")
    lines += ["", "## 边界", "", "候选收益是local-only单次标签，不能相加当顺序策略oracle。",
        "early有15标签/episode、aligned有11；协议因素同时改变检查点分布与样本数，不声称只改变时刻。",
        "no_clock/with_clock都保留36维并共享随机投影/抽样，仅时钟输入不同。",
        "TTL/drift沿用v2 validation选出的全拒绝，不在本轮test重调。",
        "未验证非reach任务、独立建议源、视觉模型、真实异步服务或现实安全。"]
    (root/"analysis.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
