"""Paired episode bootstrap; multi-training-seed hierarchy is a paper requirement."""
import csv
import json
from pathlib import Path
import numpy as np


def paired_ci(values, seed=0, repeats=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("No paired values")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


def crossed_summary(values, seed=0, repeats=5000):
    """Crossed model-seed x paired-environment bootstrap, conditional on data.

    The collection/calibration dataset is shared and is not resampled here.
    Paired contrasts must be computed BEFORE calling this function.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Need a finite model-seed by episode matrix")
    rng = np.random.default_rng(seed)
    i = rng.integers(values.shape[0], size=(repeats, values.shape[0]))
    j = rng.integers(values.shape[1], size=(repeats, values.shape[1]))
    boot = values[i[:, :, None], j[:, None, :]].mean(axis=(1, 2))
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                model_seeds=values.shape[0], episodes=values.shape[1],
                per_seed_mean=values.mean(axis=1).tolist(),
                limitation="Conditional on shared collection/calibration data; no task-population coverage")


def summarize(directory, reference="local"):
    directory = Path(directory)
    with (directory / "episodes.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = []
    for delay in sorted({int(r["delay"]) for r in rows}):
        subset = [r for r in rows if int(r["delay"])==delay]
        base = {int(r["seed"]): r for r in subset if r["method"]==reference}
        if not base:
            raise ValueError("Reference method missing: " + reference)
        for method in sorted({r["method"] for r in subset}):
            samples = [r for r in subset if r["method"]==method]
            if len({r["seed"] for r in samples}) != len(samples):
                raise ValueError("Duplicate episode seeds; aggregate training seeds separately")
            if set(base) != {int(r["seed"]) for r in samples}:
                raise ValueError("Methods do not have identical paired episode seeds")
            diffs = [float(r["utility"])-float(base[int(r["seed"])]["utility"]) for r in samples]
            success_diff = [float(r["success"])-float(base[int(r["seed"])]["success"]) for r in samples]
            audited = sum(int(r["audited_accepts"]) for r in samples)
            def total(key):
                return sum(int(r.get(key, 0)) for r in samples)
            positives, positives_deadline = total("positive_candidates"), total("deadline_positive_candidates")
            deadline_audit = all(int(r.get("audit_deadline_enabled", 0)) for r in samples)
            lower_n = total("audited_lower_count")
            summary.append(dict(delay=delay, method=method, episodes=len(samples),
                success=float(np.mean([float(r["success"]) for r in samples])),
                utility=float(np.mean([float(r["utility"]) for r in samples])),
                mean_queries=float(np.mean([float(r["requests"]) for r in samples])),
                mean_accepts=float(np.mean([float(r["accepts"]) for r in samples])),
                mean_rejects=float(np.mean([float(r["rejects"]) for r in samples])),
                paired_utility_delta=float(np.mean(diffs)), utility_delta_ci95=paired_ci(diffs),
                paired_success_delta=float(np.mean(success_diff)), success_delta_ci95=paired_ci(success_diff),
                harmful_accept_fraction=sum(int(r["harmful_accepts"]) for r in samples)/audited if audited else None,
                harmful_accept_fraction_epsilon=total("harmful_accepts_epsilon")/audited if audited and "harmful_accepts_epsilon" in samples[0] else None,
                missed_positive_fraction=total("missed_positive")/positives if positives else None,
                audited_accepts=audited, audited_candidates=total("audited_candidates"),
                deadline_harmful_accept_fraction=total("deadline_harmful_accepts")/audited if deadline_audit and audited else None,
                deadline_missed_positive_fraction=total("deadline_missed_positive")/positives_deadline if deadline_audit and positives_deadline else None,
                arrival_lower_point_coverage=total("audited_lower_covered")/lower_n if lower_n else None,
                host_runtime_p95=float(np.quantile([float(r["host_runtime_seconds"]) for r in samples], .95))))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Diagnostic results — not publication evidence", "",
             "Intervals resample paired environment seeds for ONE trained gate. They do not include model training variation.",
             "Host runtime includes optional audit clones and is NOT model latency. Delay is simulated environment steps.", "",
             "|Delay|Method|N|Final success|Queries|Utility delta vs local|95% paired CI|",
             "|---:|---|---:|---:|---:|---:|---|"]
    for r in summary:
        lo, hi = r["utility_delta_ci95"]
        lines.append("|{delay}|{method}|{episodes}|{success:.3f}|{mean_queries:.2f}|{paired_utility_delta:.4f}|[{lo:.4f}, {hi:.4f}]|".format(lo=lo, hi=hi, **r))
    (directory / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in sorted({r["method"] for r in rows}):
        subset = sorted([r for r in rows if r["method"]==method], key=lambda r:r["delay"])
        x = [r["delay"] for r in subset]
        axes[0].plot(x, [r["success"] for r in subset], marker="o", label=method)
        axes[1].plot(x, [r["utility"] for r in subset], marker="o", label=method)
    axes[0].set_ylabel("Final success fraction")
    axes[1].set_ylabel("Mean reward + success - query costs")
    for ax in axes:
        ax.set_xlabel("Simulated delay (environment steps)")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.suptitle("State diagnostic only — analytic controller and snapshot adviser")
    fig.savefig(directory / "diagnostic.png", dpi=160)
    plt.close(fig)


def plot_envelopes(dataset, output, count=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(dataset, allow_pickle=False) as data:
        groups, timestamps, q, gain = [data[k] for k in ("episode", "timestamp", "query", "gain")]
    keys = sorted(set(zip(groups.tolist(), timestamps.tolist())))[:count]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for group, t in keys:
        ix = (groups==group)&(timestamps==t)
        delay = q[ix, -2]*20
        order = np.argsort(delay)
        ax.plot(delay[order], gain[ix][order], marker="o", label="seed %s, t=%s" % (group, t))
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Delay (steps)", ylabel="Paired intervention gain",
           title="Diagnostic temporal validity curves (no monotonicity assumption)")
    ax.legend(fontsize=7)
    fig.savefig(output, dpi=160)
    plt.close(fig)
