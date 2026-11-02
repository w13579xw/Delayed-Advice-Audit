"""Portable end-to-end run using the current Python interpreter."""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["toy", "fetch"], default="toy")
parser.add_argument("--episodes", type=int, default=12)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / "outputs" / (args.backend + "_smoke")
out.mkdir(parents=True, exist_ok=True)


def run(*command):
    subprocess.run([sys.executable, "-m", "daci.cli", *map(str, command)], cwd=root, check=True)


run("collect", "--backend", args.backend, "--episodes", args.episodes,
    "--horizon", 80, "--delays", 0, 4, 12, "--out", out / "pairs.npz")
run("train", "--data", out / "pairs.npz", "--out", out / "model")
run("evaluate", "--backend", args.backend, "--model-dir", out / "model", "--out", out / "eval",
    "--episodes", 4, "--horizon", 80, "--delays", 0, 4, 12,
    "--methods", "local", "always", "risk", "ttl", "daci", "daci_calibrated", "--audit")
run("report", "--dir", out / "eval", "--plot")
run("curves", "--data", out / "pairs.npz", "--out", out / "temporal_curves.png")
print("Smoke pipeline complete:", out)
