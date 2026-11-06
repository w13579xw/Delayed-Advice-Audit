"""Deterministic simulated-time scheduler; model compute does not pause WORLD ticks.

The analytic expert is computed immediately and released after D environment
steps. This is delay emulation, NOT measured asynchronous LLM/VLA serving.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
from .compat import canonical_method
from .envs import make_env
from .policies import local_action, expert_plan
from .features import query_features, arrival_features
from .model import GainModel
from .counterfactual import paired_gain, paired_outcomes

METHODS = ("local", "always", "periodic", "random", "risk", "ttl", "drift", "query_only",
           "veto_only", "veto_calibrated", "advice_audit", "advice_audit_calibrated")
EXTRA_METHODS = ("reject",)


class Gates:
    def __init__(self, path):
        path = Path(path)
        self.query = GainModel.load(path / "query.npz")
        self.arrival = GainModel.load(path / "arrival.npz")
        self.config = json.loads((path / "training.json").read_text(encoding="utf-8"))

    def ask(self, obs, delay, chunk, cost):
        m, s = self.query.predict(query_features(obs, delay, chunk))
        return bool(m[0] - .5*s[0] > cost)

    def lower(self, obs, requested, plan, delay, calibrated=False, step=None, horizon=None):
        if self.config.get("uses_clock") and (step is None or horizon is None):
            raise ValueError("This gate requires public step and horizon clock inputs")
        clock = dict(step=step, horizon=horizon) if self.config.get("uses_clock") else {}
        features = arrival_features(obs, requested, plan, delay, **clock)
        if self.config.get("feature_set") == "no_clock":
            features = np.concatenate([features, np.zeros(2)])
        columns = self.config.get("arrival_feature_indices")
        m, s = self.arrival.predict(features[columns] if columns is not None else features)
        lower = m[0] - self.config["uncertainty_penalty"]*s[0]
        if calibrated:
            q = self.config.get("conformal_offset")
            lower -= q if q is not None else float("inf")
        # Query cost is sunk here; do not charge it twice.
        return float(lower)

    def accept(self, obs, requested, plan, delay, calibrated=False, step=None, horizon=None):
        return self.lower(obs, requested, plan, delay, calibrated, step, horizon) > 0


def episode(backend, method, seed, delay, gates=None, horizon=120, chunk=12,
            cost=.01, shift_scale=.15, budget=12, period=12, risk_threshold=.10,
            random_probability=.25, ttl=6, audit=False, label_horizon=24,
            fixed_schedule=False, drift_threshold=.05, audit_deadline=False,
            audit_epsilon=1e-6):
    reported_method = method
    method = canonical_method(method)
    if method not in METHODS + EXTRA_METHODS:
        raise ValueError("Unknown method: " + method)
    if delay < 0 or budget < 0 or period < 1:
        raise ValueError("Invalid scheduler parameter")
    if chunk < 1 or label_horizon < 1 or cost < 0 or risk_threshold < 0 or ttl < -1:
        raise ValueError("Invalid horizon/cost/threshold")
    if not 0 <= random_probability <= 1:
        raise ValueError("random_probability must be in [0, 1]")
    if fixed_schedule and period < delay+chunk:
        raise ValueError("Fixed schedule period must be >= delay + chunk to avoid overlap")
    if drift_threshold < 0 and drift_threshold != -1:
        raise ValueError("drift_threshold must be nonnegative or -1 (reject all)")
    if audit_deadline and not audit:
        raise ValueError("--audit-deadline requires --audit during evaluation")
    if not np.isfinite(audit_epsilon) or audit_epsilon < 0:
        raise ValueError("audit_epsilon must be finite and nonnegative")
    rng = np.random.default_rng(seed+800001)
    timing_rng = np.random.default_rng(seed+900001)
    shift_at = int(timing_rng.integers(horizon//4, max(horizon//4+1, 3*horizon//4)))
    env = make_env(backend, seed=seed, horizon=horizon, shift_scale=shift_scale, shift_at=shift_at)
    start = time.perf_counter()
    pending, queue, trace, rewards = None, [], [], []
    requests = accepts = rejects = harmful = audited_accepts = 0
    simulated_wait_steps = 0
    audited_candidates = positive_candidates = missed_positive = harmful_epsilon = 0
    deadline_positive = deadline_missed = deadline_harmful = deadline_harmful_zero = 0
    lower_count = lower_covered = 0
    deadline_regret = 0.0
    try:
        obs = env.reset(seed=seed)
        for _ in range(horizon):
            step = env.t
            scheduled = not fixed_schedule or step % period == 0
            if scheduled and pending is None and not queue and requests < budget and step+delay < horizon:
                if method == "local":
                    ask = False
                elif method == "periodic":
                    ask = step % period == 0
                elif method == "random":
                    ask = rng.random() < random_probability
                elif method == "risk":
                    ask = np.linalg.norm(obs[6:9]-obs[:3]) > risk_threshold
                elif method in ("advice_audit", "advice_audit_calibrated", "query_only"):
                    ask = gates.ask(obs, delay, chunk, cost)
                else:
                    ask = True
                if ask:
                    plan = expert_plan(obs, horizon=chunk, backend=backend)
                    pending = (step+delay, obs.copy(), plan)
                    requests += 1
                    trace.append(dict(t=step, event="query", ready_at=step+delay))
            if pending is not None and pending[0] <= step:
                ready, requested, plan = pending
                accept = True
                if method == "reject":
                    accept = False
                elif method == "ttl":
                    accept = delay <= ttl
                elif method == "drift":
                    accept = np.linalg.norm(obs[:9]-requested[:9]) <= drift_threshold
                elif method in ("veto_only", "veto_calibrated", "advice_audit", "advice_audit_calibrated"):
                    accept = gates.accept(obs, requested, plan, delay, method in ("advice_audit_calibrated", "veto_calibrated"), step=step, horizon=horizon)
                # Offline audit is diagnostic only; its result cannot change routing.
                outcome = paired_outcomes(env, plan, label_horizon) if audit_deadline else None
                gain = outcome["gain"] if outcome else paired_gain(env, plan, label_horizon) if audit else None
                deadline_gain = outcome["gain_deadline"] if outcome else None
                bound = None
                if gain is not None:
                    audited_candidates += 1
                    positive_candidates += int(gain > audit_epsilon)
                    missed_positive += int(not accept and gain > audit_epsilon)
                    harmful_epsilon += int(accept and gain < -audit_epsilon)
                if deadline_gain is not None:
                    deadline_positive += int(deadline_gain > audit_epsilon)
                    deadline_missed += int(not accept and deadline_gain > audit_epsilon)
                    deadline_harmful += int(accept and deadline_gain < -audit_epsilon)
                    deadline_harmful_zero += int(accept and deadline_gain < 0)
                    deadline_regret += max(deadline_gain, 0) - (deadline_gain if accept else 0)
                if audit and method in ("veto_only", "veto_calibrated", "advice_audit", "advice_audit_calibrated"):
                    bound = gates.lower(obs, requested, plan, delay, method.endswith("calibrated"), step=step, horizon=horizon)
                    label = deadline_gain if gates.config.get("target") == "gain_deadline" else gain
                    if label is not None:
                        lower_count += 1
                        lower_covered += int(bound <= label)
                if accept:
                    accepts += 1
                    queue = list(plan.copy())
                    if gain is not None:
                        audited_accepts += 1
                        harmful += int(gain < 0)
                else:
                    rejects += 1
                trace.append(dict(t=step, event="accept" if accept else "reject", audit_gain=gain,
                                  audit_gain_deadline=deadline_gain,
                                  gate_lower=bound if bound is not None and np.isfinite(bound) else None,
                                  gate_lower_infinite=bool(bound is not None and not np.isfinite(bound))))
                pending = None
            if pending is not None:
                simulated_wait_steps += 1
            action = queue.pop(0) if queue else local_action(obs, backend=backend)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        result = dict(backend=backend, method=reported_method, seed=seed, delay=delay,
                      shift_scale=shift_scale, success=int(env.success), steps=env.t,
                      final_distance=float(info["distance"]), requests=requests,
                      accepts=accepts, rejects=rejects, harmful_accepts=harmful,
                      audited_accepts=audited_accepts,
                      audited_candidates=audited_candidates,
                      positive_candidates=positive_candidates, missed_positive=missed_positive,
                      harmful_accepts_epsilon=harmful_epsilon,
                      deadline_positive_candidates=deadline_positive,
                      deadline_missed_positive=deadline_missed,
                      deadline_harmful_accepts=deadline_harmful,
                      deadline_harmful_accepts_zero=deadline_harmful_zero,
                      deadline_local_continuation_regret_sum=deadline_regret,
                      audited_lower_count=lower_count, audited_lower_covered=lower_covered,
                      mean_reward=float(np.mean(rewards)),
                      utility=float(np.mean(rewards))+float(env.success)-cost*requests,
                      query_cost=cost, delay_seconds=delay*env.dt,
                      simulated_duration=env.t*env.dt, simulated_wait_steps=simulated_wait_steps,
                      host_runtime_seconds=time.perf_counter()-start,
                      audit_enabled=int(audit), audit_deadline_enabled=int(audit_deadline),
                      audit_epsilon=audit_epsilon)
        return result, trace
    finally:
        env.close()


def evaluate(outdir, model_dir=None, backend="toy", methods=METHODS, episodes=20,
             seed=10000, delays=(0, 6, 12, 24), **kwargs):
    if episodes < 1 or not methods or not delays or len(set(methods)) != len(methods):
        raise ValueError("Need positive episodes, nonempty delays, unique nonempty methods")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gates = Gates(model_dir) if model_dir else None
    learned = {"advice_audit", "advice_audit_calibrated", "veto_only", "veto_calibrated", "query_only"}
    if learned.intersection(canonical_method(m) for m in methods) and gates is None:
        raise ValueError("Learned methods require --model-dir")
    if gates:
        meta = gates.config["data_metadata"]
        if backend != meta["backend"]:
            raise ValueError("Train/evaluate backends differ: train a gate per backend")
        if any(meta["seed_start"] <= s <= meta["seed_end"] for s in range(seed, seed+episodes)):
            raise ValueError("Evaluation seeds overlap training data")
        if kwargs.get("chunk", 12) != meta["chunk"]:
            raise ValueError("Chunk differs from training; explicit transfer support not implemented")
    rows = []
    with (outdir / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for delay in delays:
            for episode_seed in range(seed, seed+episodes):
                for method in methods:
                    row, trace = episode(backend, method, episode_seed, delay, gates, **kwargs)
                    rows.append(row)
                    traces.write(json.dumps({"method": method, "seed": episode_seed,
                                             "delay": delay, "events": trace})+"\n")
    with (outdir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(backend=backend, methods=list(methods), episodes=episodes,
                    seed=seed, delays=list(delays), config=kwargs,
                    track="state_diagnostic", delay_mode="simulated_steps",
                    model_dir=str(model_dir), success_definition="final consecutive hold",
                    limitation="Not VLA; not actual asynchronous serving; not paper evidence")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows
