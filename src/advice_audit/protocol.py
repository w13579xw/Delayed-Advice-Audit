"""Frozen episode partitions for development studies; never row-wise splitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

PARTITIONS = ("train", "validation", "calibration", "test")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_split(episodes, seed=0):
    ids = np.asarray(episodes)
    if ids.ndim != 1 or len(ids) < 20 or len(np.unique(ids)) != len(ids):
        raise ValueError("Need at least 20 unique episode IDs for four-way splitting")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Episode IDs must be integers")
    ids = np.sort(ids).copy()
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    cuts = [int(.60*n), int(.75*n), int(.90*n)]
    groups = dict(zip(PARTITIONS, [p.tolist() for p in np.split(ids, cuts)]))
    manifest = dict(schema="advice_audit.episode_split.v1", split_seed=seed, group_ids=groups,
                    unit="complete episode, including all requests/delays/branches")
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def load_split(path, episodes):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    content = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("sha256") != canonical_hash(content):
        raise ValueError("Split manifest checksum mismatch")
    if manifest.get("schema") != "advice_audit.episode_split.v1":
        raise ValueError("Unsupported split schema")
    groups = manifest["group_ids"]
    if set(groups) != set(PARTITIONS) or any(not groups[k] for k in PARTITIONS):
        raise ValueError("Need four nonempty partitions")
    flat = [v for k in PARTITIONS for v in groups[k]]
    if len(flat) != len(set(flat)) or set(flat) != set(np.unique(episodes).tolist()):
        raise ValueError("Split IDs must be disjoint and exactly match dataset episodes")
    return {k: np.isin(episodes, groups[k]) for k in PARTITIONS}, manifest
