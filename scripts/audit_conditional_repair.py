"""Exploratory fixed-arrival future resampling; no learned policy is retrained."""
import argparse
import copy
import hashlib
import json
import sys

import numpy as np

from advice_audit.envs import make_env
from advice_audit.policies import expert_plan, local_action
from advice_audit.protocol import canonical_hash
from advice_audit.repair import MODES, audited_return, local_nominal_plan
from advice_audit.report import paired_ci
from server_run import ROOT, fingerprint, now, save
from pathlib import Path


def future_snapshot(snapshot, rng):
    """Resample only an event not yet revealed, conditional on full history.

    Conditions on latent physical state/history, MORE information than gate
    features. Not a claim of a Bayes oracle on gate observations.
    """
    result = copy.deepcopy(snapshot)
    if result["t"] < result["shift_at"]:
        lower = max(result["horizon"]//4, result["t"]+1)
        upper = 3*result["horizon"]//4
        if lower >= upper:
            raise ValueError("No legal unseen future shift")
        result["shift_at"] = int(rng.integers(lower, upper))
        direction = rng.normal(size=3)
        result["goal_shift"] = direction/max(np.linalg.norm(direction), 1e-12)*result["shift_scale"]
    return result


def option_values(arrival, requested, plans, delay):
    control = audited_return(arrival, requested, plans[0], delay, "reject")["contribution"]
    values = np.zeros((len(plans), len(MODES)))
    for s, plan in enumerate(plans):
        for m, mode in enumerate(MODES[1:], 1):
            if mode in ("suffix", "track") and delay >= len(plan):
                continue
            if mode == "suffix" and delay == 0:
                values[s, m] = values[s, 1]
            else:
                values[s, m] = audited_return(arrival, requested, plan, delay, mode)["contribution"]-control
    return values


def summarize_worlds(gain):
    """Each row is one fixed arrival; worlds are independent conditional draws."""
    if gain.ndim != 4 or gain.shape[1] % 2:
        raise ValueError("Need arrivals x even worlds x sources x modes")
    half = gain.shape[1]//2
    result = {}
    for scope, subset in (("full", gain), ("cheap", gain[:, :, 1:, :])):
        flat = subset.reshape(*subset.shape[:2], -1)
        realized = flat.max(axis=2).mean(axis=1)
        pooled = flat.mean(axis=1).max(axis=1)
        a, b = flat[:, :half].mean(axis=1), flat[:, half:].mean(axis=1)
        rows = np.arange(len(flat))
        crossfit = .5*(b[rows, a.argmax(axis=1)]+a[rows, b.argmax(axis=1)])
        assert np.all(realized >= pooled-1e-12)
        result.update({scope+"_realized": realized, scope+"_sample_conditional_max": pooled,
                       scope+"_crossfit": crossfit})
    result["realized_portfolio_increment"] = result["full_realized"]-result["cheap_realized"]
    result["crossfit_source_increment"] = result["full_crossfit"]-result["cheap_crossfit"]
    result["realized_minus_crossfit"] = result["full_realized"]-result["full_crossfit"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=["toy", "fetch"])
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=74000)
    p.add_argument("--worlds", type=int, default=16)
    args = p.parse_args()
    root = Path(args.out).resolve()
    if root.exists() or args.worlds < 2 or args.worlds % 2 or args.episodes < 1:
        p.error("Need new output, positive episodes, even worlds >=2")
    root.mkdir(parents=True)
    c = dict(vars(args), horizon=80, requests=[0, 24], delays=[0, 4, 12], shift_scale=.15,
        sources=["expert", "local_P6", "local_P12"], modes=list(MODES),
        note="Prospective exploratory pilot; full physical state/history conditioned, not gate observations; finite-MC error remains")
    save(root/"protocol.json", dict(config=c, sha256=canonical_hash(c), frozen_utc=now()))
    code_hash = fingerprint()
    for path in [*ROOT.glob("src/**/*.py"), Path(__file__).resolve()]:
        dest = root/"source_snapshot"/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    state = dict(status="running", created_utc=now(), source_fingerprint=code_hash, config=c)
    save(root/"run.json", state)
    records = []
    try:
        for seed in range(args.seed, args.seed+args.episodes):
            timing = np.random.default_rng(seed+900001)
            env = make_env(args.backend, seed=seed, horizon=80, shift_at=int(timing.integers(20, 60)), shift_scale=.15)
            try:
                obs = env.reset(seed=seed)
                while env.t <= 24:
                    if env.t in c["requests"]:
                        requested = obs.copy()
                        plans = [expert_plan(requested, 12, args.backend),
                            local_nominal_plan(requested, 12, args.backend),
                            local_nominal_plan(requested, 12, args.backend, gain=12.)]
                        hashes = [hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest() for x in plans]
                        for delay in c["delays"]:
                            arrival = env.clone()
                            try:
                                for _ in range(delay):
                                    arrival.step(local_action(arrival.observe(), args.backend))
                                snapshot, visible = arrival.snapshot(), arrival.observe().copy()
                                unseen = snapshot["t"] < snapshot["shift_at"]
                                rng = np.random.default_rng(np.random.SeedSequence([seed, env.t, delay, 20261102]))
                                values, times = [], []
                                for world in range(args.worlds):
                                    sampled = future_snapshot(snapshot, rng)
                                    arrival.restore(sampled)
                                    assert np.array_equal(visible, arrival.observe())
                                    # Paired alternatives receive exactly this same world.
                                    value = option_values(arrival, requested, plans, delay) if unseen or world == 0 else values[0].copy()
                                    assert np.array_equal(visible, arrival.observe())
                                    values.append(value)
                                    times.append(sampled["shift_at"])
                                arrival.restore(snapshot)
                                assert np.array_equal(visible, arrival.observe())
                                records.append(dict(episode=seed, request_step=env.t, delay=delay,
                                    current=visible, requested=requested, plan_id=hashes,
                                    unseen=unseen, sampled_shift_at=times, gain=values))
                            finally:
                                arrival.close()
                    obs, _, _, _ = env.step(local_action(obs, args.backend))
            finally:
                env.close()
            print("EPISODE", seed, "arrivals", len(records), flush=True)
        arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
        assert arrays["gain"].shape == (6*args.episodes, args.worlds, 3, 5)
        assert np.isfinite(arrays["gain"]).all() and not arrays["gain"][:, :, :, 0].any()
        np.savez_compressed(root/"worlds.npz", **arrays)
        quantities = summarize_worlds(arrays["gain"])
        results = {}
        for key, values in quantities.items():
            means = [float(values[arrays["episode"] == ep].mean()) for ep in sorted(set(arrays["episode"].tolist()))]
            results[key] = dict(mean=float(np.mean(means)), ci95=paired_ci(means), episode_means=means)
        save(root/"analysis.json", dict(backend=args.backend, metrics=results,
            arrivals=len(records), unseen_arrivals=int(arrays["unseen"].sum()), config=c,
            limitation="Four independent physical episodes; future draws are not independent episodes. Realized-vs-crossfit gap includes finite-MC selection error, not proof of irreducible uncertainty."))
        assert fingerprint() == code_hash
        save(root/"checks.json", dict(checks="passed", observed_state_invariant=True, reject_zero=True,
            paired_common_future=True, rows=len(records), worlds=args.worlds))
        state.update(status="complete", completed_utc=now(), exit_code=0)
    except BaseException as error:
        state.update(status="failed", ended_utc=now(), error=str(error))
        save(root/"run.json", state)
        raise
    save(root/"run.json", state)
    print("COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
