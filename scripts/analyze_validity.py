"""Count sign changes over the pre-specified delay grid, at fixed request/plan."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--epsilon", type=float, default=1e-6)
args = parser.parse_args()
with np.load(args.data, allow_pickle=False) as data:
    gains, groups, timestamps = [data[k] for k in ("gain", "episode", "timestamp")]
    metadata = json.loads(str(data["metadata"]))
keys = set(zip(groups.tolist(), timestamps.tolist()))
flips = positives = 0
for group, t in keys:
    y = gains[(groups==group)&(timestamps==t)]
    positives += int((y>args.epsilon).any())
    flips += int((y>args.epsilon).any() and (y < -args.epsilon).any())
result = dict(backend=metadata["backend"], rows=len(gains), request_groups=len(keys),
              epsilon=args.epsilon, positive_fraction=float((gains>0).mean()),
              sign_flip_groups=flips, groups_with_positive_gain=positives,
              note="Finite-horizon diagnostic; one future realization, not conditional expected effect")
Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
