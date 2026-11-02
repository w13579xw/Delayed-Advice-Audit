"""Small bootstrapped random-feature ridge model; CPU runnable, no pretrained weights.

Ensemble standard deviation is a heuristic, NOT a calibrated confidence bound.
Scaling and random features are fitted only on the training split.
"""
import json
import numpy as np


class GainModel:
    def __init__(self, seed=0, width=64, members=5, ridge=1.0):
        self.seed, self.width, self.members, self.ridge = seed, width, members, ridge

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
            ix = rng.integers(len(x), size=len(x))
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


def fit_models(dataset, outdir, seed=0, cost=0.01, alpha=.2):
    from pathlib import Path
    if cost < 0:
        raise ValueError("cost must be nonnegative")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as data:
        q, a, y, groups = [data[k] for k in ("query", "arrival", "gain", "episode")]
        metadata = json.loads(str(data["metadata"]))
    masks, ids = group_split(groups, seed)
    train, val, test = masks
    # Query value is E[max(accept gain, 0)] - query cost. Arrival model approximates
    # the option to reject; this optimistic target needs a policy-consistent update
    # for a final paper (documented in research plan), not a causal guarantee.
    qm = GainModel(seed=seed).fit(q[train], np.maximum(y[train], 0))
    am = GainModel(seed=seed+1).fit(a[train], y[train])
    qm.save(outdir / "query.npz")
    am.save(outdir / "arrival.npz")
    mv, sv = am.predict(a[val])
    from .calibration import group_lower_offset, group_coverage
    # Fixed before calibration: never select beta using the calibration labels.
    penalty = .5
    offset, calibration_scores = group_lower_offset(mv-penalty*sv, y[val], groups[val], alpha)
    report = {"seed": seed, "cost": cost, "uncertainty_penalty": penalty,
              "group_ids": dict(zip(("train", "validation", "test"), ids)),
              "data_metadata": metadata, "alpha": alpha,
              "conformal_offset": offset if np.isfinite(offset) else None,
              "calibration_groups": len(calibration_scores),
              "query_target": "positive_part_oracle_gain; optimistic proxy",
              "implementation": "state diagnostic; random-feature ridge ensemble"}
    for split, mask in (("validation", val), ("test", test)):
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
