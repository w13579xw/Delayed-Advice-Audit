"""Cross-pilot measurement summary and exact cost repricing for fixed gates.

Repricing holds already selected thresholds and actions fixed. It is valid only
for the fixed-request, arrival-only methods used in this study. It is not a
re-evaluation of cost-dependent query gates or retuning at each cost.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from advice_audit.report import paired_ci


def reprice(rows, method, reference, delay, cost):
    samples = {int(r["seed"]): r for r in rows if r["method"] == method and int(r["delay"]) == delay}
    base = {int(r["seed"]): r for r in rows if r["method"] == reference and int(r["delay"]) == delay}
    if not samples or set(samples) != set(base):
        raise ValueError("Unpaired seeds")
    def utility(r):
        # Verify original recorded utility before changing the accounting cost.
        recorded = float(r["mean_reward"]) + float(r["success"]) - float(r["query_cost"])*int(r["requests"])
        if not np.isclose(recorded, float(r["utility"]), rtol=0, atol=1e-12):
            raise ValueError("Utility decomposition mismatch")
        return float(r["mean_reward"])+float(r["success"])-cost*int(r["requests"])
    diffs = [utility(samples[s])-utility(base[s]) for s in sorted(samples)]
    return dict(method=method, reference=reference, delay=delay, cost=cost,
                episodes=len(diffs), paired_utility_delta=float(np.mean(diffs)), ci95=paired_ci(diffs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new/empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for run in map(Path, args.runs):
        state = json.loads((run/"run.json").read_text())
        analysis = json.loads((run/"analysis.json").read_text())
        c = state["config"]
        if state["status"] != "complete":
            raise ValueError("Run not complete")
        baseline = max(("ttl", "drift"), key=lambda name: analysis["selection"]["selections"][name]["selected"]["utility"])
        entry = dict(run=str(run.resolve()), backend=c["backend"],
                     seconds=(datetime.fromisoformat(state['completed_utc'])-datetime.fromisoformat(state['created_utc'])).total_seconds(),
                     strongest_simple_baseline=baseline, cost_curves={}, main_comparisons={})
        for scenario in c["test_scenarios"]:
            manifest = json.loads((run/scenario/"manifest.json").read_text())
            if not manifest["config"]["fixed_schedule"]:
                raise ValueError("Repricing requires fixed schedule")
            allowed = {"local", "reject", "always", "ttl", "drift", "veto_only", "veto_calibrated"}
            if set(manifest["methods"])-allowed:
                raise ValueError("Cannot reprice cost-dependent query decisions")
            rows = list(csv.DictReader((run/scenario/"episodes.csv").open()))
            entry["cost_curves"][scenario] = [reprice(rows, method, "local", delay, cost)
                for delay in c["delays"] for method in c["test_methods"]
                for cost in (0, 1e-5, 1e-4, 1e-3, c["cost"], .05)]
            entry["main_comparisons"][scenario] = [reprice(rows, method, reference, delay, c["cost"])
                for delay in c["delays"] for method in ("veto_only", "veto_calibrated")
                for reference in (baseline, "reject")]
        entry["horizon_audit"] = analysis["horizon_audit"]
        with np.load(run/"pairs.npz", allow_pickle=False) as data:
            short, deadline = data["gain"], data["gain_deadline"]
            eps = c["epsilon"]
            entry["strict_horizon_audit"] = dict(
                strict_positive_short=int((short > 0).sum()),
                strict_positive_deadline=int((deadline > 0).sum()),
                strict_short_positive_deadline_negative=int(((short > 0) & (deadline < 0)).sum()),
                epsilon_short_positive_deadline_negative=int(((short > eps) & (deadline < -eps)).sum()),
                epsilon_short_positive_deadline_near_zero=int(((short > eps) & (np.abs(deadline) <= eps)).sum()),
                note="Separates actual sign reversals from crossing the epsilon tolerance under different gain normalizations")
            latest_request = int(data["timestamp"].max())
            latest_arrival = int((data["timestamp"]+data["delay"]).max())
        support = {}
        for scenario in c["test_scenarios"]:
            traces = [json.loads(line) for line in (run/scenario/"traces.jsonl").read_text().splitlines()]
            calibrated = [t for t in traces if t["method"] == "veto_calibrated"]
            queries = [e for t in calibrated for e in t["events"] if e["event"] == "query"]
            decisions = [e for t in calibrated for e in t["events"] if e["event"] != "query"]
            support[scenario] = dict(
                latest_collection_request=latest_request, latest_collection_arrival=latest_arrival,
                queries=len(queries), requests_beyond_collection_max=sum(e["t"] > latest_request for e in queries),
                arrivals_beyond_collection_max=sum(e["ready_at"] > latest_arrival for e in queries),
                calibrated_accepts=sum(e["event"] == "accept" for e in decisions),
                same_checkpoint_protocol=False,
                deadline_clock_in_features=False,
                note="Coverage differences cannot be attributed solely to policy-induced shift: checkpoint rules differ; full deadline features omit remaining time")
        entry["checkpoint_support_audit"] = support
        entry["offline_test"] = analysis["offline_test"]
        entry["input_sha256"] = {p: hashlib.sha256((run/p).read_bytes()).hexdigest()
                                 for p in ("run.json", "analysis.json", "protocol.json", "selection.json")}
        results.append(entry)
    (out/"comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    lines = ["# 跨后端测量与成本敏感性", "", "仅为当前两个reach后端的pilot；不代表独立任务家族或发表验收。",
             "成本重计保持动作与validation选定阈值不变，适用于当前固定请求、到达门控；不是每个成本重新调参的最优曲线。", ""]
    for r in results:
        h = r["horizon_audit"]["all"]
        lines += [f"## {r['backend']}", "", f"运行：`{r['run']}`；耗时{r['seconds']:.2f}秒。",
                  f"1200标签：短窗口正收益{h['short_positive']}，截止正收益{h['deadline_positive']}；短正但截止非正{h['short_positive_deadline_nonpositive']}。",
                  "上行的‘非正’按epsilon=1e-6判定；严格符号审计：`"+json.dumps(r['strict_horizon_audit'])+"`。",
                  f"local-only候选的乐观单次oracle平均截止收益{h['oracle_mean_deadline_gain']:.8f}，扣0.01后{h['oracle_net_after_one_forced_query_cost']:.8f}；不是顺序策略上界。",
                  f"最强简单基线仅由validation选择：{r['strongest_simple_baseline']}。", "",
                  "检查点支持范围审计：`"+json.dumps(r['checkpoint_support_audit'])+"`。", "",
                  "|场景|delay|方法|对照|效用差|95%配对CI|", "|---|---:|---|---|---:|---|"]
        for scenario, rows in r["main_comparisons"].items():
            for x in rows:
                lines.append(f"|{scenario}|{x['delay']}|{x['method']}|{x['reference']}|{x['paired_utility_delta']:.7f}|{x['ci95']}|")
        for scenario, rows in r["cost_curves"].items():
            lines += ["", f"### {scenario}：零查询成本，仍保留相同动作和阈值", "",
                      "|delay|方法|相对local效用差|95%配对CI|", "|---:|---|---:|---|"]
            for x in rows:
                if x["cost"] == 0:
                    lines.append(f"|{x['delay']}|{x['method']}|{x['paired_utility_delta']:.7f}|{x['ci95']}|")
    lines += ["", "完整成本网格和输入文件哈希见comparison.json。CI仅重采样12个配对episode，不含训练seed方差；不做事后挑选显著性。"]
    (out/"comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps([dict(backend=r['backend'], seconds=r['seconds'], strongest_simple_baseline=r['strongest_simple_baseline']) for r in results], indent=2))


if __name__ == "__main__":
    main()
