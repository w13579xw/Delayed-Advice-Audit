"""Sequential low-memory diagnostic sweep; not a ready-made publication experiment."""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["toy", "fetch"], default="toy")
parser.add_argument("--train-episodes", type=int, default=100)
parser.add_argument("--test-episodes", type=int, default=30)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--out", default="outputs/sweep")
args = parser.parse_args()
root = Path(args.out) / args.backend
root.mkdir(parents=True, exist_ok=True)


def run(*command):
    subprocess.run([sys.executable, "-m", "advice_audit.cli", *map(str, command)], check=True)


run("collect", "--backend", args.backend, "--episodes", args.train_episodes,
    "--seed", 100, "--out", root / "pairs.npz")
for seed in args.seeds:
    model = root / ("model_%d" % seed)
    run("train", "--data", root / "pairs.npz", "--seed", seed, "--out", model)
    for setting, shift in [("id", .15), ("ood_shift", .30)]:
        out = root / ("seed_%d_%s" % (seed, setting))
        run("evaluate", "--backend", args.backend, "--episodes", args.test_episodes,
            "--seed", 10000, "--shift-scale", shift, "--model-dir", model,
            "--out", out, "--delays", 0, 2, 6, 12, 24)
        run("report", "--dir", out, "--plot")
print("24-step delays and .30 shifts are held-out shifts, not covered by the offline calibration statement.")
