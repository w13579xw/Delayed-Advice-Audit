"""Checkpoint protocol x public clock, with frozen seeds and serial execution."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from advice_audit.protocol import canonical_hash, make_split
from server_run import ROOT, fingerprint, now, save


def validate(config):
    if config["schema"] != "advice_audit.alignment_study.v1":
        raise ValueError("Unknown alignment protocol")
    if (config["horizon"], config["period"], config["interval"], config["delays"]) != (80, 24, 10, [0, 4, 12]):
        raise ValueError("This protocol freezes horizon80, period24, early interval10 and delays0/4/12")
    if config["collection_protocols"] != ["early", "aligned"] or config["feature_sets"] != ["no_clock", "with_clock"]:
        raise ValueError("Both predeclared factors are required")
    if config["model_seeds"] != [0, 1, 2, 3, 4]:
        raise ValueError("Five predeclared model seeds required")
    if config["period"] < max(config["delays"])+config["chunk"]:
        raise ValueError("Overlapping fixed request slots")
    if not 0 < config["closed_loop_episodes"] <= config["test_episodes"] or config["test_episodes"] < 5:
        raise ValueError("Invalid closed-loop subset")
    train = set(range(config["collection_seed"], config["collection_seed"]+config["episodes"]))
    test = set(range(config["test_seed"], config["test_seed"]+config["test_episodes"]))
    if train & test or config["episodes"] < 20:
        raise ValueError("Episode overlap or insufficient split groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["toy", "fetch"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/alignment_study_v1.json")
    args = parser.parse_args()
    c = dict(json.loads(Path(args.config).read_text()), backend=args.backend)
    validate(c)
    root = Path(args.run_dir).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Use a new/empty run directory; do not overwrite or resume changed studies")
    root.mkdir(parents=True, exist_ok=True)
    (root/"logs").mkdir()
    split = make_split(list(range(c["collection_seed"], c["collection_seed"]+c["episodes"])), c["split_seed"])
    save(root/"split_manifest.json", split)
    save(root/"protocol.json", dict(config=c, sha256=canonical_hash(c), frozen_utc=now()))
    save(root/"environment.json", dict(python=sys.version, executable=sys.executable,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}))
    (root/"tracked_source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"),
                        *ROOT.glob("configs/*.json"), ROOT/"pyproject.toml"]):
        target = root/"source_snapshot"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    state = dict(status="running", created_utc=now(), config=c, protocol_sha256=canonical_hash(c),
                 split_sha256=split["sha256"], source_fingerprint=fingerprint(), phases={},
                 commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    def run(name, arguments):
        entry = dict(status="running", command=[sys.executable, "-u", *map(str, arguments)],
                     started_utc=now(), log=str(root/"logs"/(name+".log")))
        state["phases"][name] = entry
        save(root/"run.json", state)
        print("START", name, flush=True)
        with Path(entry["log"]).open("w") as log:
            completed = subprocess.run(entry["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry.update(exit_code=completed.returncode, ended_utc=now())
        if completed.returncode:
            entry["status"] = "failed"
            raise RuntimeError(f"{name}: exit {completed.returncode}; see {entry['log']}")
        entry["status"] = "complete"
        save(root/"run.json", state)
        print("DONE", name, flush=True)

    cli = ["-m", "advice_audit.cli"]
    collect_common = ["--backend", c["backend"], "--horizon", c["horizon"], "--chunk", c["chunk"],
        "--delays", *c["delays"], "--rollout-horizon", c["rollout_horizon"], "--audit-deadline", "--include-clock"]
    evaluate_common = ["--backend", c["backend"], "--horizon", c["horizon"], "--chunk", c["chunk"],
        "--delays", *c["delays"], "--seed", c["test_seed"], "--episodes", c["closed_loop_episodes"],
        "--period", c["period"], "--fixed-schedule", "--cost", c["cost"],
        "--audit", "--audit-deadline", "--label-horizon", c["rollout_horizon"], "--audit-epsilon", c["epsilon"]]
    try:
        run("tests", ["-m", "pytest", "-q"])
        run("profile", ["scripts/profile_server.py", "--backend", c["backend"], "--out", root/"profile.json"])
        for protocol in c["collection_protocols"]:
            extra = ["--request-period", c["period"]] if protocol == "aligned" else ["--interval", c["interval"]]
            run("collect_"+protocol, cli+["collect", "--out", root/"data"/(protocol+".npz"),
                "--episodes", c["episodes"], "--seed", c["collection_seed"], "--shift-scale", c["shift_scale"]]+collect_common+extra)
        for protocol in c["collection_protocols"]:
            for feature in c["feature_sets"]:
                for seed in c["model_seeds"]:
                    name = f"{protocol}_{feature}_{seed}"
                    run("train_"+name, cli+["train", "--data", root/"data"/(protocol+".npz"),
                        "--out", root/"models"/name, "--split-manifest", root/"split_manifest.json",
                        "--target", "gain_deadline", "--feature-set", feature, "--seed", seed,
                        "--cost", c["cost"], "--alpha", c["alpha"]])
        # No test-label-dependent choices follow: all heads/contrasts were frozen.
        for scenario, shift in c["test_scenarios"].items():
            run("test_labels_"+scenario, cli+["collect", "--out", root/"data"/("test_"+scenario+".npz"),
                "--episodes", c["test_episodes"], "--seed", c["test_seed"], "--shift-scale", shift,
                "--request-period", c["period"]]+collect_common)
            run("baseline_"+scenario, cli+["evaluate", "--out", root/"baselines"/scenario,
                "--methods", *c["baseline_methods"], "--ttl", c["baseline_ttl"],
                "--drift-threshold", c["baseline_drift"], "--shift-scale", shift]+evaluate_common)
            for feature in c["feature_sets"]:
                for seed in c["model_seeds"]:
                    name = f"aligned_{feature}_{seed}"
                    run("closed_"+scenario+"_"+name, cli+["evaluate", "--out", root/"closed"/scenario/name,
                        "--methods", "veto_only", "veto_calibrated", "--model-dir", root/"models"/name,
                        "--shift-scale", shift]+evaluate_common)
        run("analyze", ["scripts/analyze_alignment.py", "--run-dir", root])
        if fingerprint() != state["source_fingerprint"]:
            raise RuntimeError("Code changed during run; keep separate source configurations")
        state.update(status="complete", completed_utc=now())
    except BaseException as error:
        state.update(status="failed", ended_utc=now(), error=str(error))
        save(root/"run.json", state)
        raise
    save(root/"run.json", state)
    print("COMPLETE", root, "— controlled state study, not a visual benchmark", flush=True)


if __name__ == "__main__":
    main()
