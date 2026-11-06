"""Fresh-data learning test for the repair/source opportunity; offline only."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from advice_audit.protocol import canonical_schema, canonical_hash
from advice_audit.repair import MODES
from advice_audit.repair_selector import OPTIONS, RepairSelector, features_for, fit_selector, load_data, option_mask
from advice_audit.report import crossed_summary
from server_run import ROOT, fingerprint, now, save


def read(path):
    return json.loads(path.read_text())


def fit(root):
    c = read(root/"protocol.json")["config"]
    if (root/"test.npz").exists():
        raise ValueError("Fit and freeze models before test collection")
    data = load_data(root/"validation.npz")
    means = data["gain"].mean(axis=0)
    source, mode = np.unravel_index(int(np.argmax(means)), means.shape)
    if means.max() <= c["epsilon"]:
        source, mode = 0, 0
    save(root/"fixed_selection.json", dict(source=int(source), mode=int(mode), selected_utc=now(),
        criterion="Maximum validation-only mean, reject when gain <= epsilon"))
    for seed in c["model_seeds"]:
        fit_selector(root, root/"models"/str(seed), seed, c)
        print("FITTED", seed, flush=True)


def values_from_scores(scores, gain):
    best = np.argmax(scores, axis=1)
    take = scores[np.arange(len(best)), best] > 0
    source = np.array([OPTIONS[j][0] for j in best])
    mode = np.array([OPTIONS[j][1] for j in best])
    value = np.where(take, gain[np.arange(len(best)), source, mode], 0.)
    return value, take, source, mode


def analyze(root):
    c = read(root/"protocol.json")["config"]
    state = read(root/"run.json")
    assert state["config"] == c
    assert state["phases"]["fit"]["ended_utc"] <= state["phases"]["collect_test"]["started_utc"]
    data = load_data(root/"test.npz")
    ids = sorted(set(data["episode"].tolist()))
    assert ids == list(range(c["test_seed"], c["test_seed"]+c["test_episodes"]))
    assert len(data["gain"]) == 11*c["test_episodes"]
    inputs = features_for(data, c["backend"], c["chunk"], c["horizon"])
    target = np.stack([data["gain"][:, source, mode] for source, mode in OPTIONS], axis=1)
    allowed = option_mask(data["delay"], c["chunk"])
    fixed = read(root/"fixed_selection.json")
    fixed_values = data["gain"][:, fixed["source"], fixed["mode"]]
    def per_episode(values):
        return [float(np.asarray(values)[data["episode"] == ep].mean()) for ep in ids]
    results, matrices = {}, {}
    for calibrated, variant in [(False, "point"), (True, "calibrated")]:
        for cheap_only, scope in [(False, "full"), (True, "cheap_only")]:
            name = variant+"_"+scope
            gains, coverage, counts, harm, expert_counts = [], [], [], [], []
            for seed in c["model_seeds"]:
                selector = RepairSelector(root/"models"/str(seed))
                scores = selector.scores(inputs, data["delay"], calibrated, cheap_only)
                value, take, sources, modes = values_from_scores(scores, data["gain"])
                oracle = data["gain"].max(axis=(1, 2))
                assert np.all(value <= oracle+1e-12)
                gains.append(per_episode(value))
                mask = allowed.copy()
                if cheap_only:
                    mask[:, :4] = False
                covered = (scores <= target) | ~mask
                coverage.append([float(np.all(covered[data["episode"] == ep])) for ep in ids])
                counts.append(int(take.sum()))
                harm.append(int((take & (value < -c["epsilon"])).sum()))
                expert_counts.append(int((take & (sources == 0)).sum()))
            matrix = np.asarray(gains)
            matrices[name] = matrix
            results[name] = dict(value=crossed_summary(matrix), vs_fixed=crossed_summary(matrix-np.asarray(per_episode(fixed_values))),
                episode_lower_coverage=crossed_summary(coverage), selected_counts=counts, harmful_counts=harm,
                expert_selected_counts=expert_counts,
                note="Offline one-intervention gain per candidate, not closed-loop success or deployment safety")
    contrasts = {name: crossed_summary(matrices[name+"_full"]-matrices[name+"_cheap_only"]) for name in ("point", "calibrated")}
    result = dict(backend=c["backend"], config=c, results=results, contrasts=contrasts, fixed_selection=fixed,
        fixed_value=crossed_summary([per_episode(fixed_values)]),
        oracle_value=crossed_summary([per_episode(data["gain"].max(axis=(1, 2)))]),
        limitation="Local-only behavior, one shared data draw, no actual adaptive closed-loop evaluation")
    save(root/"analysis.json", result)
    save(root/"checks.json", dict(checks="passed", backend=c["backend"], train_episodes=c["train_episodes"],
        validation_episodes=c["validation_episodes"], calibration_episodes=c["calibration_episodes"],
        test_episodes=c["test_episodes"], model_seeds=c["model_seeds"], options=len(OPTIONS),
        note="Integrity and held-out offline evaluation, not a required positive model result"))
    lines = ["# 修复/来源选择器：真实模型的独立离线测试", "", result["limitation"], "",
        "|选择器|平均候选收益|95% crossed CI|相对验证集固定控制|每seed选择数|每seed有害数|每seed专家使用数|", "|---|---:|---|---:|---|---|---|"]
    for name, r in results.items():
        lines.append(f"|{name}|{r['value']['mean']:.8f}|{r['value']['ci95']}|{r['vs_fixed']['mean']:.8f}|{r['selected_counts']}|{r['harmful_counts']}|{r['expert_selected_counts']}|")
    lines += ["", "full减cheap_only：`"+json.dumps(contrasts)+"`。", "",
        "固定控制只用validation选择；各repair残差尺度只用validation，joint-max offset只用calibration。cheap-only使用自身较小选项集合的独立offset，不故意使用full的更宽界削弱对照。",
        "点估计选择器按预测均值-0.5ensemble std选择，不读未来。校准选择器在此基础上扣除joint group-max标准化残差界。都是既有回归/校准工具的组合原型，不宣称新架构或定理。",
        "若候选收益为正，仍不能直接称闭环任务提升；所有数字是同一local-only候选池上的离线单次结果。"]
    (root/"analysis.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(results={k: v["value"] for k, v in results.items()}, contrasts=contrasts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["toy", "fetch"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/repair_selector_study_v1.json")
    parser.add_argument("--phase", choices=["fit", "analyze"])
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if args.phase:
        (fit if args.phase == "fit" else analyze)(root)
        return
    if not args.backend or (root.exists() and any(root.iterdir())):
        parser.error("Need backend and a new/empty directory")
    c = dict(read(Path(args.config)), backend=args.backend)
    if canonical_schema(c["schema"]) != "advice_audit.repair_selector.v1" or c["model_seeds"] != [0, 1, 2, 3, 4] or (c["horizon"], c["period"], c["chunk"], c["delays"]) != (80, 24, 12, [0, 4, 12]):
        parser.error("Unsupported frozen selector protocol")
    if c["sources"] != ["expert", "local_nominal", "local_nominal_fast"] or c["cheap_source_gains"] != [6., 12.]:
        parser.error("Need expert and both predeclared cheap controls")
    pools = [set(range(c[role+"_seed"], c[role+"_seed"]+c[role+"_episodes"])) for role in ("train", "validation", "calibration", "test")]
    if any(not x for x in pools) or len(set.union(*pools)) != sum(map(len, pools)):
        parser.error("Need four disjoint nonempty episode pools")
    root.mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir()
    save(root/"protocol.json", dict(config=c, sha256=canonical_hash(c), frozen_utc=now()))
    save(root/"environment.json", dict(python=sys.version, packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}))
    for file in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.json"), ROOT/"pyproject.toml"]):
        target_file = root/"source_snapshot"/file.relative_to(ROOT)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(file.read_bytes())
    state = dict(status="running", created_utc=now(), config=c, source_fingerprint=fingerprint(), phases={},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    phases = [("collect_"+role, ["scripts/repair_value_pilot.py", "--run-dir", str(root), "--phase", "collect_"+role]) for role in ("train", "validation", "calibration")]
    phases += [("fit", [__file__, "--run-dir", str(root), "--phase", "fit"]),
        ("collect_test", ["scripts/repair_value_pilot.py", "--run-dir", str(root), "--phase", "collect_test"]),
        ("analyze", [__file__, "--run-dir", str(root), "--phase", "analyze"])]
    try:
        for name, args in phases:
            command = [sys.executable, "-u", *args]
            entry = dict(status="running", command=command, started_utc=now(), log=str(root/"logs"/(name+".log")))
            state["phases"][name] = entry
            save(root/"run.json", state)
            print("START", name, flush=True)
            with Path(entry["log"]).open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry.update(exit_code=completed.returncode, ended_utc=now())
            if completed.returncode:
                entry["status"] = "failed"
                raise RuntimeError(f"{name} failed; see {entry['log']}")
            entry["status"] = "complete"
            save(root/"run.json", state)
            print("DONE", name, flush=True)
        if fingerprint() != state["source_fingerprint"]:
            raise RuntimeError("Code changed during study")
        state.update(status="complete", completed_utc=now())
    except BaseException as error:
        state.update(status="failed", ended_utc=now(), error=str(error))
        save(root/"run.json", state)
        raise
    save(root/"run.json", state)
    print("COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
