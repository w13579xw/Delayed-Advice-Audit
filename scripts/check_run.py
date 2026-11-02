"""Fail if a server bootstrap is incomplete or contains inconsistent episode counts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
args = parser.parse_args()
root = Path(args.run_dir)
state = json.loads((root / "run.json").read_text(encoding="utf-8"))
config = state["config"]
training = json.loads((root / "model/training.json").read_text(encoding="utf-8"))
ids = training["group_ids"]
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not set(ids[left]).intersection(ids[right]), "Data split leakage"
with np.load(root / "pairs.npz", allow_pickle=False) as data:
    assert np.isfinite(data["gain"]).all(), "Nonfinite gains"
    assert len(np.unique(data["episode"])) == config["train_episodes"]
    pair_count = len(data["gain"])
counts = {}
for folder in ("eval", "fixed"):
    manifest = json.loads((root / folder / "manifest.json").read_text(encoding="utf-8"))
    with (root / folder / "episodes.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = len(manifest["methods"]) * len(manifest["delays"]) * config["test_episodes"]
    assert len(rows) == expected, (folder, len(rows), expected)
    assert len(set((r["method"], r["seed"], r["delay"]) for r in rows)) == expected
    assert all(np.isfinite(float(r["utility"])) for r in rows)
    assert all(int(r["seed"]) >= 10000 for r in rows), "Train/eval seed overlap"
    for artifact in ("summary.json", "summary.md", "diagnostic.png", "traces.jsonl"):
        assert (root / folder / artifact).stat().st_size > 0, artifact
    if folder == "fixed":
        for delay in manifest["delays"]:
            for seed in range(10000, 10000+config["test_episodes"]):
                queries = {int(r["requests"]) for r in rows if int(r["delay"])==delay and
                           int(r["seed"])==seed and r["method"]!="local"}
                assert len(queries) == 1, "Fixed schedule query counts differ"
    counts[folder] = len(rows)
report = dict(checks="passed", backend=config["backend"], pairs=pair_count, episode_rows=counts,
              calibration_groups=training["calibration_groups"],
              finite_conformal_offset=training["conformal_offset"] is not None,
              note="Integrity checks only. No required positive research result.")
(root / "checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
