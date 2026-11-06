"""Small bootstrapped random-feature ridge model; CPU runnable, no pretrained weights.

Ensemble standard deviation is a heuristic, NOT a calibrated confidence bound.
Scaling and random features are fitted only on the training split.
"""
import json
import hashlib
import numpy as np


class GainModel:
    def __init__(self, seed=0, width=64, members=5, ridge=1.0, bootstrap=True):
        self.seed, self.width, self.members, self.ridge = seed, width, members, ridge
        self.bootstrap = bootstrap

    def _features(self, x):
        z = (np.asarray(x) - self.mean) / self.scale
        return np.concatenate([np.ones((len(z), 1)), z,
                               np.tanh(z @ self.proj + self.bias)], axis=1)

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Need >=2 finite training rows")
        rng = np.random.default_rng(self.seed)
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 0.02)
        self.proj = rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], self.width))
        self.bias = rng.uniform(-1, 1, self.width)
        phi = self._features(x)
        reg = np.eye(phi.shape[1]) * self.ridge
        reg[0, 0] = 1e-8
        self.weights = []
        for _ in range(self.members):
            ix = rng.integers(len(x), size=len(x)) if self.bootstrap else np.arange(len(x))
            a = phi[ix]
            self.weights.append(np.linalg.solve(a.T @ a + reg, a.T @ y[ix]))
        self.weights = np.asarray(self.weights)
        return self

    def predict(self, x):
        predictions = self._features(np.atleast_2d(x)) @ self.weights.T
        return predictions.mean(1), predictions.std(1)

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale, proj=self.proj,
                            bias=self.bias, weights=self.weights)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in ("mean", "scale", "proj", "bias", "weights"):
                setattr(model, key, data[key])
        return model


def group_split(groups, seed=0):
    """Split whole episode IDs, never individual counterfactual branches."""
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("Use at least 5 episodes for train/validation/test groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = min(len(unique) - 2, max(1, int(.6 * len(unique))))
    n_val = min(len(unique) - n_train - 1, max(1, int(.2 * len(unique))))
    parts = (unique[:n_train], unique[n_train:n_train+n_val], unique[n_train+n_val:])
    return [np.isin(groups, part) for part in parts], [p.tolist() for p in parts]


def fit_models(dataset, outdir, seed=0, cost=0.01, alpha=.2,
               split_manifest=None, target="gain", feature_set="full", linear=False):
    from pathlib import Path
    if cost < 0:
        raise ValueError("cost must be nonnegative")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as data:
        if target not in ("gain", "gain_deadline") or target not in data:
            raise ValueError("Requested target unavailable; collect with --audit-deadline for deadline labels")
        q, a, y, groups = [data[k] for k in ("query", "arrival", target, "episode")]
        metadata = json.loads(str(data["metadata"]))
    from .features import arrival_columns
    columns = arrival_columns(feature_set)
    if max(columns) >= a.shape[1] or (feature_set in ("with_clock", "no_clock") and not metadata.get("clock_features")):
        raise ValueError("Requested clock features unavailable in dataset")
    a = a[:, columns]
    if feature_set == "no_clock":
        # Keep dimensions, random projections and bootstrap draws matched.
        a[:, 34:] = 0
    if not np.isfinite(q).all() or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite dataset")
    split_info = None
    if split_manifest:
        from .protocol import load_split
        partitions, split_info = load_split(split_manifest, groups)
        ids = split_info["group_ids"]
        train, cal = partitions["train"], partitions["calibration"]
    else:
        if target != "gain" or feature_set != "full" or linear:
            raise ValueError("Research targets/ablations require a frozen four-way --split-manifest")
        masks, split_ids = group_split(groups, seed)
        ids = dict(zip(("train", "validation", "test"), split_ids))
        partitions = dict(zip(("train", "validation", "test"), masks))
        train, cal = partitions["train"], partitions["validation"]
    # Query value is E[max(accept gain, 0)] - query cost. Arrival model approximates
    # the option to reject; this optimistic target needs a policy-consistent update
    # for a final paper (documented in research plan), not a causal guarantee.
    qm = GainModel(seed=seed).fit(q[train], np.maximum(y[train], 0))
    am = GainModel(seed=seed+1, width=0 if linear else 64,
                   members=1 if linear else 5, bootstrap=not linear).fit(a[train], y[train])
    qm.save(outdir / "query.npz")
    am.save(outdir / "arrival.npz")
    mv, sv = am.predict(a[cal])
    from .calibration import group_lower_offset, group_coverage
    # Fixed before calibration: never select beta using the calibration labels.
    penalty = .5
    offset, calibration_scores = group_lower_offset(mv-penalty*sv, y[cal], groups[cal], alpha)
    report = {"seed": seed, "cost": cost, "uncertainty_penalty": penalty,
              "group_ids": ids,
              "data_metadata": metadata, "alpha": alpha,
              "conformal_offset": offset if np.isfinite(offset) else None,
              "calibration_groups": len(calibration_scores),
              "query_target": "positive_part_oracle_gain; optimistic proxy",
              "implementation": "state diagnostic; linear ridge" if linear else "state diagnostic; random-feature ridge ensemble",
              "target": target, "feature_set": feature_set, "arrival_feature_indices": columns,
              "uses_clock": feature_set != "no_clock" and any(i >= 34 for i in columns),
              "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
              "split_manifest_sha256": split_info["sha256"] if split_info else None,
              "split_roles": "four-way independent validation/calibration" if split_info else "legacy three-way; validation is calibration"}
    for split, mask in partitions.items():
        if split == "train":
            continue
        m, s = am.predict(a[mask])
        accepted = m - penalty*s > 0
        report[split] = {
            "rows": int(mask.sum()), "gain_mae": float(np.abs(m-y[mask]).mean()),
            "accept_fraction": float(accepted.mean()),
            "negative_gain_among_accepted": float((y[mask][accepted] < 0).mean()) if accepted.any() else None,
            "paired_label_utility": float(np.where(accepted, y[mask], 0).mean()),
            "offline_episode_lower_coverage": group_coverage(m-penalty*s-offset, y[mask], groups[mask]),
        }
    (outdir / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
