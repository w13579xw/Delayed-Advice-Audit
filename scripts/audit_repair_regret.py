"""Posthoc accounting on frozen tests; no refit or test-based model choice."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from advice_audit.repair_selector import RepairSelector, features_for, load_data
from advice_audit.report import crossed_summary
from repair_selector_study import values_from_scores
from server_run import now, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.parent).resolve(), Path(args.out).resolve()
    if out.exists():
        parser.error("Use a new output directory")
    state = json.loads((root/"run.json").read_text())
    assert state["status"] == "complete"
    c = state["config"]
    data = load_data(root/"test.npz")
    inputs = features_for(data, c["backend"], c["chunk"], c["horizon"])
    expert = data["gain"][:, 0].max(axis=1)
    cheap = data["gain"][:, 1:].max(axis=(1, 2))
    total = np.maximum(expert, cheap)
    increment = total-cheap
    assert np.all(increment >= 0)
    ids = sorted(set(data["episode"].tolist()))

    def grouped(values):
        return np.array([values[data["episode"] == ep].mean() for ep in ids])

    result = dict(backend=c["backend"], created_utc=now(), parent=str(root),
        test_sha256=hashlib.sha256((root/"test.npz").read_bytes()).hexdigest(),
        expert_oracle=crossed_summary([grouped(expert)]),
        cheap_oracle=crossed_summary([grouped(cheap)]),
        portfolio_increment=crossed_summary([grouped(increment)]), variants={},
        note="Descriptive posthoc. Realized future oracle; exact algebra, not new theorem or deployment guarantee.")
    for calibrated, name in ((False, "point"), (True, "calibrated")):
        rows = {key: [] for key in ("full_regret", "cheap_regret", "learned_difference", "accounted_difference")}
        error = 0.
        for seed in c["model_seeds"]:
            model = RepairSelector(root/"models"/str(seed))
            full = values_from_scores(model.scores(inputs, data["delay"], calibrated, False), data["gain"])[0]
            null = values_from_scores(model.scores(inputs, data["delay"], calibrated, True), data["gain"])[0]
            full_regret, cheap_regret = total-full, cheap-null
            account = increment-full_regret+cheap_regret
            error = max(error, float(np.max(np.abs(full-null-account))))
            assert np.allclose(full-null, account, atol=1e-12, rtol=0)
            for key, value in zip(rows, (full_regret, cheap_regret, full-null, account)):
                rows[key].append(grouped(value))
        result["variants"][name] = {key: crossed_summary(value) for key, value in rows.items()}
        result["variants"][name]["identity_max_abs_error"] = error
    out.mkdir(parents=True)
    save(out/"analysis.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
