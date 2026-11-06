"""Paired finite-horizon outcomes for a single intervention, not global causal claims."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .features import query_features, arrival_features
from .policies import local_action, expert_plan
from .envs import make_env


def utility_rollout(env, plan=None, horizon=24):
    rewards = []
    for k in range(min(horizon, env.horizon-env.t)):
        obs = env.observe()
        action = plan[k] if plan is not None and k < len(plan) else local_action(obs, backend=env.backend)
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    return (float(np.mean(rewards)) if rewards else 0.) + float(env.success)


def paired_gain(env, plan, horizon=24):
    """Clone both branches from the SAME arrival state, including future RNG.

    The teacher can use simulator clones offline. No gate sees these snapshots.
    A fixed local continuation is used after the candidate chunk in both branches.
    """
    control, treatment = env.clone(), env.clone()
    try:
        base = utility_rollout(control, None, horizon)
        treated = utility_rollout(treatment, plan, horizon)
        return treated-base
    finally:
        control.close()
        treatment.close()


def paired_outcomes(env, plan, horizon=24):
    """Offline single-intervention audit through the ORIGINAL episode deadline.

    Deadline gain uses sum(rewards)/episode_horizon + terminal hold success.
    The common past cancels, so this equals the full-episode utility difference
    from exactly one intervention followed by local-only continuation. It is
    not an oracle for future interventions by an adaptive policy.
    """
    if horizon < 1 or env.t >= env.horizon:
        raise ValueError("Need a positive window and a live arrival state")
    branches = []
    for candidate in (None, plan):
        branch = env.clone()
        try:
            rewards, short_success = [], None
            steps = min(horizon, branch.horizon - branch.t)
            for k in range(branch.horizon - branch.t):
                action = candidate[k] if candidate is not None and k < len(candidate) else local_action(branch.observe(), backend=branch.backend)
                _, reward, done, _ = branch.step(action)
                rewards.append(reward)
                if len(rewards) == steps:
                    short_success = float(branch.success)
                if done:
                    break
            branches.append(dict(short_utility=float(np.mean(rewards[:steps])) + short_success,
                                 deadline_contribution=float(np.sum(rewards))/env.horizon + float(branch.success),
                                 terminal_success=int(branch.success), steps=len(rewards)))
        finally:
            branch.close()
    control, treatment = branches
    return dict(gain=treatment["short_utility"]-control["short_utility"],
                gain_deadline=treatment["deadline_contribution"]-control["deadline_contribution"],
                success_gain=treatment["terminal_success"]-control["terminal_success"],
                control=control, treatment=treatment)


def collect(path, backend="toy", episodes=40, seed=100, horizon=120,
            delays=(0, 2, 6, 12), chunk=12, rollout_horizon=24,
            shift_scale=.15, interval=10, audit_deadline=False,
            request_period=None, include_clock=False):
    if episodes < 5 or not delays or any(d < 0 for d in delays):
        raise ValueError("Need >=5 episodes and nonnegative delays")
    if interval < 1 or chunk < 1 or rollout_horizon < 1:
        raise ValueError("interval, chunk, rollout_horizon must be positive")
    if request_period is not None and (not audit_deadline or request_period < max(delays)+chunk):
        raise ValueError("Aligned collection requires deadline audit and period >= max(delay)+chunk")
    queries, arrivals, gains, groups, timestamps = [], [], [], [], []
    audit_rows, ages, drifts, plan_ids = [], [], [], []
    for episode in range(episodes):
        episode_seed = seed + episode
        # Disturbance timing randomized BEFORE rolling out, invisible to gate.
        timing_rng = np.random.default_rng(episode_seed+900001)
        shift_at = int(timing_rng.integers(horizon//4, max(horizon//4+1, 3*horizon//4)))
        env = make_env(backend, seed=episode_seed, horizon=horizon,
                       shift_scale=shift_scale, shift_at=shift_at)
        try:
            obs = env.reset(seed=episode_seed)
            collection_end = horizon if request_period is not None else horizon-max(delays)-rollout_horizon
            while env.t < collection_end:
                if env.t % (request_period if request_period is not None else interval) == 0:
                    requested = obs.copy()
                    plan = expert_plan(requested, horizon=chunk, backend=backend)
                    for delay in delays:
                        if env.t + delay >= horizon:
                            continue
                        arrival = env.clone()
                        try:
                            for _ in range(delay):
                                _, _, done, _ = arrival.step(local_action(arrival.observe(), backend=backend))
                                if done:
                                    break
                            if arrival.t >= arrival.horizon:
                                continue
                            queries.append(query_features(requested, delay, chunk))
                            clock = dict(step=arrival.t, horizon=horizon) if include_clock else {}
                            arrivals.append(arrival_features(arrival.observe(), requested, plan, delay, **clock))
                            if audit_deadline:
                                outcome = paired_outcomes(arrival, plan, rollout_horizon)
                                gains.append(outcome["gain"])
                                audit_rows.append(outcome)
                            else:
                                gains.append(paired_gain(arrival, plan, rollout_horizon))
                            groups.append(episode_seed)
                            timestamps.append(env.t)
                            ages.append(delay)
                            drifts.append(float(np.linalg.norm(arrival.observe()[:9]-requested[:9])))
                            plan_ids.append(hashlib.sha256(np.ascontiguousarray(plan, dtype="<f4").tobytes()).hexdigest())
                        finally:
                            arrival.close()
                obs, _, done, _ = env.step(local_action(obs, backend=backend))
                if done:
                    break
        finally:
            env.close()
    if not gains:
        raise ValueError("No rows collected: increase horizon or reduce delay/rollout horizon")
    metadata = dict(backend=backend, episodes=episodes, seed_start=seed,
                    seed_end=seed+episodes-1, horizon=horizon, delays=list(delays),
                    chunk=chunk, rollout_horizon=rollout_horizon, shift_scale=shift_scale,
                    interval=interval, track="privileged_state_diagnostic",
                    expert="analytic_snapshot_open_loop", delay_mode="simulated_steps",
                    behavior_policy="local_only", utility="mean_dense_reward + final_success",
                    schema="advice_audit.pairs.v2", audit_deadline=bool(audit_deadline),
                    request_protocol="fixed_schedule" if request_period is not None else "legacy_early_interval",
                    request_period=request_period, clock_features=bool(include_clock),
                    deadline_utility="sum_remaining_rewards / original_episode_horizon + terminal_success; common past cancels",
                    continuation="one intervention then local-only, no future expert calls")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if audit_deadline:
        extra = {key: np.array([r[key] for r in audit_rows]) for key in ("gain_deadline", "success_gain")}
        for branch in ("control", "treatment"):
            for key in ("short_utility", "deadline_contribution", "terminal_success"):
                extra[branch+"_"+key] = np.array([r[branch][key] for r in audit_rows])
    np.savez_compressed(path, query=np.array(queries), arrival=np.array(arrivals),
                        gain=np.array(gains), episode=np.array(groups),
                        timestamp=np.array(timestamps), delay=np.array(ages),
                        arrival_step=np.array(timestamps)+np.array(ages),
                        short_window_steps=np.minimum(rollout_horizon, horizon-np.array(timestamps)-np.array(ages)),
                        drift=np.array(drifts), plan_id=np.array(plan_ids),
                        metadata=json.dumps(metadata), **extra)
    return {"rows": len(gains), "positive_fraction": float(np.mean(np.array(gains)>0)),
            "negative_fraction": float(np.mean(np.array(gains)<0)), "metadata": metadata}
