"""Paired finite-horizon outcomes for a single intervention, not global causal claims."""
import json
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


def collect(path, backend="toy", episodes=40, seed=100, horizon=120,
            delays=(0, 2, 6, 12), chunk=12, rollout_horizon=24,
            shift_scale=.15, interval=10):
    if episodes < 5 or not delays or any(d < 0 for d in delays):
        raise ValueError("Need >=5 episodes and nonnegative delays")
    if interval < 1 or chunk < 1 or rollout_horizon < 1:
        raise ValueError("interval, chunk, rollout_horizon must be positive")
    queries, arrivals, gains, groups, timestamps = [], [], [], [], []
    for episode in range(episodes):
        episode_seed = seed + episode
        # Disturbance timing randomized BEFORE rolling out, invisible to gate.
        timing_rng = np.random.default_rng(episode_seed+900001)
        shift_at = int(timing_rng.integers(horizon//4, max(horizon//4+1, 3*horizon//4)))
        env = make_env(backend, seed=episode_seed, horizon=horizon,
                       shift_scale=shift_scale, shift_at=shift_at)
        try:
            obs = env.reset(seed=episode_seed)
            while env.t < horizon-max(delays)-rollout_horizon:
                if env.t % interval == 0:
                    requested = obs.copy()
                    plan = expert_plan(requested, horizon=chunk, backend=backend)
                    for delay in delays:
                        arrival = env.clone()
                        try:
                            for _ in range(delay):
                                _, _, done, _ = arrival.step(local_action(arrival.observe(), backend=backend))
                                if done:
                                    break
                            if arrival.t >= arrival.horizon:
                                continue
                            queries.append(query_features(requested, delay, chunk))
                            arrivals.append(arrival_features(arrival.observe(), requested, plan, delay))
                            gains.append(paired_gain(arrival, plan, rollout_horizon))
                            groups.append(episode_seed)
                            timestamps.append(env.t)
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
                    behavior_policy="local_only", utility="mean_dense_reward + final_success")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, query=np.array(queries), arrival=np.array(arrivals),
                        gain=np.array(gains), episode=np.array(groups),
                        timestamp=np.array(timestamps), metadata=json.dumps(metadata))
    return {"rows": len(gains), "positive_fraction": float(np.mean(np.array(gains)>0)),
            "negative_fraction": float(np.mean(np.array(gains)<0)), "metadata": metadata}
