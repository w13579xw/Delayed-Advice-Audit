"""Logged, resumable server bootstrap. Run this only from the installed venv.

Every run has its own directory. A resume requires identical configuration and
source fingerprint. Completed phases are skipped only if their outputs exist.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint():
    digest = hashlib.sha256()
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"),
                    *ROOT.glob("tests/*.py"), ROOT / "pyproject.toml"])
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def phases(config, directory):
    py = [sys.executable, "-u"]
    cli = py + ["-m", "advice_audit.cli"]
    model, data = directory / "model", directory / "pairs.npz"
    common = ["--backend", config["backend"], "--model-dir", str(model),
              "--episodes", str(config["test_episodes"]), "--horizon", "80",
              "--seed", "10000", "--delays", "0", "4", "12"]
    if config["audit"]:
        common.append("--audit")
    return [
        ("profile", py + ["scripts/profile_server.py", "--backend", config["backend"],
                          "--out", str(directory / "profile.json")], ["profile.json"]),
        ("tests", py + ["-m", "pytest", "-q"], []),
        ("collect", cli + ["collect", "--backend", config["backend"], "--episodes", str(config["train_episodes"]),
                           "--seed", "100", "--horizon", "80", "--delays", "0", "4", "12", "--out", str(data)], ["pairs.npz"]),
        ("train", cli + ["train", "--data", str(data), "--out", str(model)],
         ["model/training.json", "model/query.npz", "model/arrival.npz"]),
        ("evaluate", cli + ["evaluate", "--out", str(directory / "eval")] + common,
         ["eval/episodes.csv", "eval/manifest.json", "eval/traces.jsonl"]),
        ("fixed", cli + ["evaluate", "--out", str(directory / "fixed")] + common +
         ["--fixed-schedule", "--period", "24", "--methods", "local", "always", "ttl", "drift", "veto_only", "veto_calibrated"],
         ["fixed/episodes.csv", "fixed/manifest.json", "fixed/traces.jsonl"]),
        ("report", cli + ["report", "--dir", str(directory / "eval"), "--plot"],
         ["eval/summary.json", "eval/summary.md", "eval/diagnostic.png"]),
        ("fixed_report", cli + ["report", "--dir", str(directory / "fixed"), "--plot"],
         ["fixed/summary.json", "fixed/summary.md", "fixed/diagnostic.png"]),
        ("curves", cli + ["curves", "--data", str(data), "--out", str(directory / "temporal_curves.png")], ["temporal_curves.png"]),
        ("validity", py + ["scripts/analyze_validity.py", "--data", str(data), "--out", str(directory / "validity_analysis.json")],
         ["validity_analysis.json"]),
        ("verify", py + ["scripts/check_run.py", "--run-dir", str(directory)], ["checks.json"]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["toy", "fetch"], default="toy")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.train_episodes < 5 or args.test_episodes < 1:
        parser.error("train-episodes must be >=5; test-episodes must be >=1")
    directory = Path(args.run_dir).resolve()
    config = dict(backend=args.backend, train_episodes=args.train_episodes,
                  test_episodes=args.test_episodes, audit=args.audit)
    state_file = directory / "run.json"
    if args.resume:
        if not state_file.exists():
            parser.error("--resume requires an existing run.json")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["config"] != config or state["source_fingerprint"] != fingerprint():
            parser.error("Configuration or code changed; use a NEW --run-dir")
    else:
        if directory.exists() and any(directory.iterdir()):
            parser.error("Run directory is not empty; use a new path or --resume")
        directory.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        state = dict(created_utc=now(), config=config, source_fingerprint=fingerprint(),
                     python=sys.version, platform=platform.platform(),
                     commit=commit.stdout.strip() if commit.returncode == 0 else None,
                     status="running", phases={})
    (directory / "logs").mkdir(exist_ok=True)
    state["status"] = "running"
    save(state_file, state)
    for name, command, expected in phases(config, directory):
        entry = state["phases"].get(name, {})
        if entry.get("status") == "complete":
            if any(not (directory / p).exists() for p in expected):
                raise RuntimeError("Completed phase has missing outputs: " + name + "; use a new run directory")
            print("SKIP", name, flush=True)
            continue
        attempt = entry.get("attempt", 0) + 1
        logfile = directory / "logs" / ("%s_attempt_%02d.log" % (name, attempt))
        entry = dict(status="running", attempt=attempt, started_utc=now(), command=command, log=str(logfile))
        state["phases"][name] = entry
        save(state_file, state)
        print("START", name, "log:", logfile, flush=True)
        try:
            with logfile.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1"))
            if completed.returncode:
                raise RuntimeError("Phase %s failed with exit %s; see %s" % (name, completed.returncode, logfile))
            if any(not (directory / p).exists() for p in expected):
                raise RuntimeError("Phase %s returned zero but has missing outputs" % name)
        except (Exception, KeyboardInterrupt) as exc:
            entry.update(status="failed", ended_utc=now(), error=str(exc))
            state["status"] = "failed"
            save(state_file, state)
            raise
        entry.update(status="complete", ended_utc=now())
        save(state_file, state)
        print("DONE", name, flush=True)
    state.update(status="complete", completed_utc=now())
    save(state_file, state)
    print("COMPLETE", directory, "— diagnostic reproduction only, not publication results", flush=True)


if __name__ == "__main__":
    main()
