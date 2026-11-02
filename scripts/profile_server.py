"""Measure state simulator throughput; no model/GPU inference claim is made."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from daci.envs import make_env
from daci.policies import local_action, expert_plan
from daci.counterfactual import paired_gain

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["toy", "fetch"], default="fetch")
parser.add_argument("--out", default="outputs/server_profile.json")
args = parser.parse_args()
result = {"python": sys.version, "platform": platform.platform(), "backend": args.backend,
          "note": "State-simulation CPU profile; NOT 3090 VLA latency or training measurement"}
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    result["gpu_inventory"] = gpu.stdout.strip() if gpu.returncode == 0 else "nvidia-smi failed"
except (FileNotFoundError, subprocess.TimeoutExpired):
    result["gpu_inventory"] = "unavailable"
env = make_env(args.backend, horizon=1500)
try:
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        env.step(local_action(env.observe(), backend=args.backend))
        times.append(time.perf_counter()-start)
    result["step_seconds_p50_p95"] = np.quantile(times, [.5, .95]).tolist()
    start = time.perf_counter()
    paired_gain(env, expert_plan(env.observe(), backend=args.backend), horizon=24)
    result["one_24step_paired_label_seconds"] = time.perf_counter()-start
finally:
    env.close()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
