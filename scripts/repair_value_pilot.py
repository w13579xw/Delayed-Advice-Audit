"""Source-attributed repair opportunity pilot; no oracle is deployed as a gate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from advice_audit.envs import make_env
from advice_audit.policies import expert_plan, local_action
from advice_audit.protocol import canonical_hash
from advice_audit.repair import MODES, audited_return, local_nominal_plan
from advice_audit.report import paired_ci
from server_run import ROOT, fingerprint, now, save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_phase(root, phase):
    c = read(root/"protocol.json")["config"]
    output = root/(phase+".npz")
    if output.exists():
        raise ValueError("Refusing to overwrite phase output")
    records = []
    for seed in range(c[phase+"_seed"], c[phase+"_seed"]+c[phase+"_episodes"]):
        timing = np.random.default_rng(seed+900001)
        shift_at = int(timing.integers(c["horizon"]//4, 3*c["horizon"]//4))
        env = make_env(c["backend"], seed=seed, horizon=c["horizon"], shift_at=shift_at, shift_scale=c["shift_scale"])
        try:
            obs = env.reset(seed=seed)
            while env.t < c["horizon"]:
                if env.t % c["period"] == 0:
                    requested = obs.copy()
                    plans = [expert_plan(requested, horizon=c["chunk"], backend=c["backend"]),
                             local_nominal_plan(requested, horizon=c["chunk"], backend=c["backend"])]
                    if "local_nominal_fast" in c["sources"]:
                        plans.append(local_nominal_plan(requested, horizon=c["chunk"], backend=c["backend"], gain=12.))
                    hashes = [hashlib.sha256(np.ascontiguousarray(p, dtype="<f4").tobytes()).hexdigest() for p in plans]
                    for delay in c["delays"]:
                        if env.t+delay >= env.horizon:
                            continue
                        arrival = env.clone()
                        try:
                            for _ in range(delay):
                                arrival.step(local_action(arrival.observe(), c["backend"]))
                            control = audited_return(arrival, requested, plans[0], delay, "reject", c["feedback_gain"])
                            gains = np.zeros((len(plans), len(MODES)))
                            successes = np.zeros_like(gains, dtype=int)
                            outcomes = {}
                            for source, plan in enumerate(plans):
                                for index, mode in enumerate(MODES):
                                    if mode == "reject" or (mode in ("suffix", "track") and delay >= len(plan)):
                                        outcome = control
                                    elif mode == "suffix" and delay == 0:
                                        outcome = outcomes[(source, "raw")]
                                    else:
                                        outcome = audited_return(arrival, requested, plan, delay, mode, c["feedback_gain"])
                                    outcomes[(source, mode)] = outcome
                                    gains[source, index] = outcome["contribution"]-control["contribution"]
                                    successes[source, index] = outcome["terminal_success"]-control["terminal_success"]
                            records.append(dict(episode=seed, request_step=env.t, arrival_step=arrival.t, delay=delay,
                                requested=requested.copy(), current=arrival.observe(), plan_id=hashes,
                                gain=gains, success_gain=successes))
                        finally:
                            arrival.close()
                obs, _, done, _ = env.step(local_action(obs, c["backend"]))
                if done:
                    break
        finally:
            env.close()
        print("COLLECTED", phase, seed, len(records), flush=True)
    arrays = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    assert np.isfinite(arrays["gain"]).all()
    assert not arrays["gain"][:, :, 0].any()
    metadata = dict(config=c, phase=phase, sources=c["sources"], modes=c["modes"],
        behavior="local-only, one treatment then local", query_cost="sunk, identical forced requests; not deducted again",
        limitation="Operational value relative to declared cheap source/repair family, not mutual information")
    np.savez_compressed(output, **arrays, metadata=json.dumps(metadata))
    print(json.dumps(dict(rows=len(records), cells=int(arrays["gain"].size), phase=phase)))


def select(root):
    if (root/"test.npz").exists():
        raise ValueError("Selection must precede test collection")
    c = read(root/"protocol.json")["config"]
    with np.load(root/"validation.npz", allow_pickle=False) as data:
        gains = data["gain"]
    choices = {}
    for i, source in enumerate(c["sources"]):
        means = gains[:, i, :].mean(axis=0)
        selected = int(np.argmax(means)) if means.max() > c["epsilon"] else 0
        choices[source] = dict(mode=MODES[selected], index=selected, validation_means=dict(zip(MODES, means.tolist())))
    save(root/"selection.json", dict(choices=choices, selected_utc=now(),
        note="One fixed mode per source, validation only, reject preferred at epsilon-level ties"))


def analyze(root):
    protocol = read(root/"protocol.json")
    c, selection = protocol["config"], read(root/"selection.json")
    assert canonical_hash(c) == protocol["sha256"]
    if (root/"run.json").exists():
        state = read(root/"run.json")
        assert state["config"] == c
        assert selection["selected_utc"] <= state["phases"]["collect_test"]["started_utc"]
    with np.load(root/"test.npz", allow_pickle=False) as data:
        gains, groups = data["gain"], data["episode"]
        assert len(gains) == c["test_episodes"]*11
        assert set(groups.tolist()) == set(range(c["test_seed"], c["test_seed"]+c["test_episodes"]))
        for ep, t in set(zip(groups.tolist(), data["request_step"].tolist())):
            mask = (groups == ep) & (data["request_step"] == t)
            for source in range(2):
                assert len(set(data["plan_id"][mask, source])) == 1
        success_gains = data["success_gain"]
    assert gains.shape[1:] == (2, len(MODES)) and not gains[:, :, 0].any()
    assert np.isfinite(gains).all()
    expert, cheap = gains[:, 0], gains[:, 1]
    raw = np.maximum(expert[:, MODES.index("raw")], 0)
    repaired, null = expert.max(axis=1), cheap.max(axis=1)
    ei = selection["choices"]["expert"]["index"]
    fi = selection["choices"]["local_nominal"]["index"]
    quantities = dict(raw_accept_reject_oracle=raw, repaired_expert_oracle=repaired,
        repaired_cheap_oracle=null, repair_opportunity=repaired-raw,
        source_specific_oracle_difference=repaired-null,
        contextual_opportunity=repaired-expert[:, ei],
        expert_fixed=expert[:, ei], cheap_fixed=cheap[:, fi],
        expert_fixed_vs_cheap_fixed=expert[:, ei]-cheap[:, fi],
        same_operator_source_difference=expert[:, ei]-cheap[:, ei])
    def summarize(values):
        episode_means = [float(values[groups == ep].mean()) for ep in sorted(set(groups.tolist()))]
        return dict(mean=float(np.mean(episode_means)), ci95=paired_ci(episode_means), episode_means=episode_means)
    metrics = {k: summarize(v) for k, v in quantities.items()}
    source_value = metrics["source_specific_oracle_difference"]
    context = metrics["contextual_opportunity"]
    go = source_value["ci95"][0] > 0 and source_value["mean"] >= c["minimum_mean_value"] and context["ci95"][0] > 0
    fixed = {source: {mode: summarize(gains[:, i, j]) for j, mode in enumerate(MODES)} for i, source in enumerate(c["sources"])}
    result = dict(backend=c["backend"], config=c, selection=selection, metrics=metrics, fixed_modes=fixed,
        decision="go_opportunity_only" if go else "no_go_for_learned_selector_on_this_channel",
        note="Oracle uses future labels: necessary opportunity check, not a deployable method result or universal source-information measure",
        terminal_harm_counts={source: {mode: int((success_gains[:, i, j] < 0).sum()) for j, mode in enumerate(MODES)} for i, source in enumerate(c["sources"])})
    save(root/"analysis.json", result)
    save(root/"checks.json", dict(checks="passed", backend=c["backend"], test_rows=len(gains),
        test_episodes=c["test_episodes"], sources=c["sources"], modes=c["modes"],
        note="Integrity, source controls and shared candidate identity; no required positive result"))
    lines = ["# 修复后的专家增量价值：先行否证", "", result["note"], "", "## 固定操作（未扣重复查询费用）", "",
        "|来源|操作|平均单次增益|按episode配对95% CI|", "|---|---|---:|---|"]
    for source, rows in fixed.items():
        for mode, value in rows.items():
            lines.append(f"|{source}|{mode}|{value['mean']:.8f}|{value['ci95']}|")
    lines += ["", "## 分离修复收益和来源收益", "", "|量|平均值|按episode配对95% CI|", "|---|---:|---|"]
    for key, value in metrics.items():
        lines.append(f"|{key}|{value['mean']:.8f}|{value['ci95']}|")
    lines += ["", "验证集选择：`"+json.dumps(selection["choices"])+"`。", "", "Decision: **"+result["decision"]+"**。",
        "", "所有结果为单次干预后local继续的离线标签；oracle不是闭环性能，不将其写成模型提升。",
        "廉价source是同一local控制器的名义模型展开，不是新学到的专家或独立任务。修复操作是已知思路的简单对照，不宣称RTC/FutureRTC复现。",
        "12个test episode仅作pilot；若必要机会条件不成立，不在该test上调模型/选操作找正结果。"]
    (root/"analysis.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(decision=result["decision"], metrics={k: v for k, v in metrics.items() if k in ("repair_opportunity", "source_specific_oracle_difference", "contextual_opportunity")})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["toy", "fetch"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/repair_value_pilot_v1.json")
    parser.add_argument("--phase", choices=["collect_train", "collect_validation", "collect_calibration", "select", "collect_test", "analyze"])
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if args.phase:
        if args.phase.startswith("collect_"):
            collect_phase(root, args.phase.removeprefix("collect_"))
        elif args.phase == "select":
            select(root)
        else:
            analyze(root)
        return
    if not args.backend:
        parser.error("--backend is required for a new run")
    if root.exists() and any(root.iterdir()):
        parser.error("Use a new/empty directory")
    c = dict(read(Path(args.config)), backend=args.backend)
    if c["modes"] != list(MODES) or c["sources"] != ["expert", "local_nominal"] or c["horizon"] != 80 or c["period"] != 24 or c["delays"] != [0, 4, 12]:
        parser.error("Unsupported frozen design")
    val = set(range(c["validation_seed"], c["validation_seed"]+c["validation_episodes"]))
    test = set(range(c["test_seed"], c["test_seed"]+c["test_episodes"]))
    if not val or not test or val & test:
        parser.error("Validation/test must be disjoint and nonempty")
    root.mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir()
    save(root/"protocol.json", dict(config=c, sha256=canonical_hash(c), frozen_utc=now()))
    save(root/"environment.json", dict(python=sys.version, packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}))
    for file in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.json"), ROOT/"pyproject.toml"]):
        destination = root/"source_snapshot"/file.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file.read_bytes())
    state = dict(status="running", created_utc=now(), config=c, source_fingerprint=fingerprint(), phases={},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        for phase in ("collect_validation", "select", "collect_test", "analyze"):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-dir", str(root), "--phase", phase]
            entry = dict(status="running", command=command, started_utc=now(), log=str(root/"logs"/(phase+".log")))
            state["phases"][phase] = entry
            save(root/"run.json", state)
            print("START", phase, flush=True)
            with Path(entry["log"]).open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry.update(exit_code=completed.returncode, ended_utc=now())
            if completed.returncode:
                entry["status"] = "failed"
                raise RuntimeError(f"{phase} failed; see {entry['log']}")
            entry["status"] = "complete"
            save(root/"run.json", state)
            print("DONE", phase, flush=True)
        if fingerprint() != state["source_fingerprint"]:
            raise RuntimeError("Source changed during run")
        state.update(status="complete", completed_utc=now())
    except BaseException as error:
        state.update(status="failed", ended_utc=now(), error=str(error))
        save(root/"run.json", state)
        raise
    save(root/"run.json", state)
    print("COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
