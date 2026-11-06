"""Crossed paired bootstrap over training seeds and shared environment seeds.

Only for one backend/setting/delay at a time. No duplicate training seeds.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", nargs="+", required=True)
parser.add_argument("--method", default="advice_audit")
parser.add_argument("--reference", default="local")
parser.add_argument("--delay", type=int, default=12)
parser.add_argument("--out", required=True)
args = parser.parse_args()
matrices, env_seeds, train_seeds, backend_settings = [], None, [], []
for directory in args.dirs:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    training = json.loads((Path(manifest["model_dir"]) / "training.json").read_text(encoding="utf-8"))
    train_seeds.append(training["seed"])
    backend_settings.append((manifest["backend"], manifest["config"].get("shift_scale"),
                             manifest["config"].get("cost")))
    with (directory / "episodes.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["delay"]) == args.delay]
    a = {int(r["seed"]): float(r["utility"]) for r in rows if r["method"] == args.method}
    b = {int(r["seed"]): float(r["utility"]) for r in rows if r["method"] == args.reference}
    if set(a) != set(b) or not a:
        raise ValueError("Unpaired episode seeds")
    if env_seeds is None:
        env_seeds = sorted(a)
    if env_seeds != sorted(a):
        raise ValueError("All training seeds must share evaluation episode seeds")
    matrices.append([a[s]-b[s] for s in env_seeds])
if len(set(train_seeds)) != len(train_seeds) or len(set(backend_settings)) != 1:
    raise ValueError("Duplicate training seeds or mixed backend/settings")
values = np.array(matrices)
rng = np.random.default_rng(0)
boot = []
for _ in range(5000):
    i = rng.integers(len(values), size=len(values))
    j = rng.integers(len(env_seeds), size=len(env_seeds))
    boot.append(float(values[np.ix_(i, j)].mean()))
result = dict(method=args.method, reference=args.reference, delay=args.delay,
              n_train_seeds=len(values), n_env_seeds=len(env_seeds),
              mean_paired_utility_difference=float(values.mean()),
              crossed_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
              limitation="Diagnostic policies only; no task-family population coverage")
Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
