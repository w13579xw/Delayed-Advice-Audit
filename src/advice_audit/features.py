"""Observable-only features. No simulator snapshot or future disturbance enters gates."""
import numpy as np


def query_features(obs, delay, chunk):
    obs = np.asarray(obs, dtype=float)
    rel = obs[6:9] - obs[:3]
    return np.concatenate([obs, rel, [np.linalg.norm(rel), delay / 20., chunk / 20.]])


def arrival_features(current, requested, plan, delay, step=None, horizon=None):
    current, requested, plan = map(np.asarray, (current, requested, plan))
    features = np.concatenate([
        query_features(current, delay, len(plan)),
        current[:3] - requested[:3], current[6:9] - requested[6:9],
        plan[0], plan[-1], plan.mean(axis=0),
        [np.linalg.norm(current[6:9] - requested[6:9])],
    ])
    if step is None and horizon is None:
        return features
    if step is None or horizon is None or not 0 <= step < horizon:
        raise ValueError("Clock features require a live step and original episode horizon")
    # Both are public scheduler quantities, not future task/branch outcomes.
    return np.concatenate([features, [(horizon-step)/horizon, horizon/80.]])


def arrival_columns(mode="full"):
    """Predeclared ablations; no label-dependent feature selection.

    no_age removes only explicit age (motion/history can remain age proxies).
    no_candidate removes the plan summaries; current_only removes age, history
    differences and candidate information, retaining current obs and chunk size.
    """
    modes = {"full": list(range(34)), "with_clock": list(range(36)),
             "no_clock": list(range(36)),
             "no_age": [i for i in range(34) if i != 16],
             "no_candidate": [i for i in range(34) if not 24 <= i < 33],
             "current_only": list(range(16)) + [17]}
    if mode not in modes:
        raise ValueError("Unknown arrival feature set")
    return modes[mode]
