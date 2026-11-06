"""Frozen, serial state-diagnostic study for horizon and channel falsification.

New directories only. No automatic long sweep, no model downloads, no claim of
publication readiness. Raw command logs and failed stages are retained.
"""
import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from advice_audit.protocol import canonical_schema, canonical_hash, make_split
from server_run import ROOT, fingerprint, now, save


def validate_config(c):
    if canonical_schema(c.get("schema")) != "advice_audit.state_pilot.v2" or c.get("target") != "gain_deadline":
        raise ValueError("Unsupported protocol schema/target")
    if c["episodes"] < 20 or min(c["validation_episodes"], c["test_episodes"]) < 1:
        raise ValueError("Insufficient episode groups")
    if len(c["delays"]) != len(set(c["delays"])) or min(c["delays"]) < 0:
        raise ValueError("Delays must be distinct and nonnegative")
    if c["period"] < max(c["delays"]) + c["chunk"]:
        raise ValueError("Fixed schedule overlap")
    pools = [set(range(c[k], c[k]+c[n])) for k, n in [
        ("collection_seed", "episodes"), ("validation_seed", "validation_episodes"), ("test_seed", "test_episodes")]]
    if any(pools[i] & pools[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Collection/closed-loop validation/test seeds overlap")
    if c["feature_models"] != ["full", "linear", "no_age", "no_candidate", "current_only"]:
        raise ValueError("Use the predeclared feature models")


def select_threshold(candidates):
    """Only supplied validation rows are read; ties prefer fewer accepts."""
    scored = []
    for value, csv_path, method in candidates:
        rows = [r for r in csv.DictReader(Path(csv_path).open()) if r["method"] == method]
        if not rows:
            raise ValueError("Missing validation method")
        scored.append(dict(value=value, utility=sum(float(r["utility"]) for r in rows)/len(rows),
                           accepts=sum(int(r["accepts"]) for r in rows), rows=len(rows),
                           csv=str(csv_path)))
    selected = max(scored, key=lambda r: (r["utility"], -r["accepts"], -scored.index(r)))
    return dict(selected=selected, candidates=scored,
                criterion="mean closed-loop validation utility; ties prefer fewer accepts, then grid order")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["toy", "fetch"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/state_pilot_v2.json")
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Run directory must be new/empty; no overwrite or automatic resume")
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    config = dict(config, backend=args.backend)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir()
    split = make_split(list(range(config["collection_seed"], config["collection_seed"]+config["episodes"])), config["split_seed"])
    save(directory / "split_manifest.json", split)
    save(directory / "protocol.json", dict(config=config, sha256=canonical_hash(config), frozen_utc=now(),
                                           note="Frozen before data collection; development pilot"))
    deps = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    save(directory / "environment.json", dict(python=sys.version, executable=sys.executable, packages=deps))
    (directory / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    # Include untracked research source in the run: git diff alone omits it.
    source_dir = directory / "source_snapshot"
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"),
                        *ROOT.glob("configs/*.json"), ROOT / "pyproject.toml"]):
        target = source_dir / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    state = dict(status="running", created_utc=now(), config=config,
                 protocol_sha256=canonical_hash(config), split_sha256=split["sha256"],
                 commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 source_fingerprint=fingerprint(), phases={})
    state_file = directory / "run.json"
    save(state_file, state)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    def run(name, arguments):
        command = [sys.executable, "-u", *map(str, arguments)]
        log = directory / "logs" / (name+".log")
        entry = dict(status="running", command=command, started_utc=now(), log=str(log))
        state["phases"][name] = entry
        save(state_file, state)
        print("START", name, flush=True)
        try:
            with log.open("w") as handle:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            entry.update(exit_code=result.returncode, ended_utc=now())
            if result.returncode:
                raise RuntimeError(f"{name} exit {result.returncode}; see {log}")
        except BaseException as error:
            entry.update(status="failed", ended_utc=now(), error=str(error))
            state.update(status="failed")
            save(state_file, state)
            raise
        entry["status"] = "complete"
        save(state_file, state)
        print("DONE", name, flush=True)

    cli = ["-m", "advice_audit.cli"]
    c = config
    run("tests", ["-m", "pytest", "-q"])
    run("profile", ["scripts/profile_server.py", "--backend", args.backend, "--out", directory/"profile.json"])
    run("collect", cli + ["collect", "--backend", args.backend, "--episodes", c["episodes"],
        "--seed", c["collection_seed"], "--horizon", c["horizon"], "--delays", *c["delays"],
        "--chunk", c["chunk"], "--rollout-horizon", c["rollout_horizon"], "--interval", c["interval"],
        "--shift-scale", c["shift_scale"], "--audit-deadline", "--out", directory/"pairs.npz"])
    for mode in c["feature_models"]:
        arguments = cli + ["train", "--data", directory/"pairs.npz", "--out", directory/"models"/mode,
            "--split-manifest", directory/"split_manifest.json", "--target", c["target"],
            "--feature-set", "full" if mode == "linear" else mode,
            "--seed", c["model_seed"], "--cost", c["cost"], "--alpha", c["alpha"]]
        run("train_"+mode, arguments + (["--linear"] if mode == "linear" else []))
    common = ["--backend", args.backend, "--horizon", c["horizon"], "--chunk", c["chunk"],
              "--cost", c["cost"], "--shift-scale", c["shift_scale"], "--delays", *c["delays"],
              "--fixed-schedule", "--period", c["period"]]
    selected = {}
    for name, flag in [("ttl", "--ttl"), ("drift", "--drift-threshold")]:
        candidates = []
        for index, value in enumerate(c[name+"_grid"]):
            output = directory / "validation" / (name+"_"+str(index))
            run(f"validation_{name}_{index}", cli + ["evaluate", "--out", output,
                "--seed", c["validation_seed"], "--episodes", c["validation_episodes"],
                "--methods", "local", name, flag, value] + common)
            candidates.append((value, output/"episodes.csv", name))
        selected[name] = select_threshold(candidates)
    save(directory / "selection.json", dict(selections=selected, selected_utc=now(),
        validation_seed=c["validation_seed"], validation_episodes=c["validation_episodes"],
        note="Only closed-loop validation CSVs read; no calibration/test tuning"))
    for scenario, shift in c["test_scenarios"].items():
        scenario_common = common.copy()
        scenario_common[scenario_common.index("--shift-scale")+1] = shift
        run(scenario, cli + ["evaluate", "--out", directory/scenario, "--seed", c["test_seed"],
            "--episodes", c["test_episodes"], "--model-dir", directory/"models/full",
            "--methods", *c["test_methods"], "--ttl", selected["ttl"]["selected"]["value"],
            "--drift-threshold", selected["drift"]["selected"]["value"],
            "--audit", "--audit-deadline", "--audit-epsilon", c["epsilon"],
            "--label-horizon", c["rollout_horizon"]] + scenario_common)
        run("report_"+scenario, cli + ["report", "--dir", directory/scenario, "--plot"])
    run("analyze", ["scripts/analyze_research.py", "--run-dir", directory])
    if fingerprint() != state["source_fingerprint"]:
        state["status"] = "failed"
        save(state_file, state)
        raise RuntimeError("Source changed during study; do not mix results")
    state.update(status="complete", completed_utc=now())
    save(state_file, state)
    print("COMPLETE", directory, "— state diagnostic development study", flush=True)


if __name__ == "__main__":
    main()
