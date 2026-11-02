"""Observable-only features. No simulator snapshot or future disturbance enters gates."""
import numpy as np


def query_features(obs, delay, chunk):
    obs = np.asarray(obs, dtype=float)
    rel = obs[6:9] - obs[:3]
    return np.concatenate([obs, rel, [np.linalg.norm(rel), delay / 20., chunk / 20.]])


def arrival_features(current, requested, plan, delay):
    current, requested, plan = map(np.asarray, (current, requested, plan))
    return np.concatenate([
        query_features(current, delay, len(plan)),
        current[:3] - requested[:3], current[6:9] - requested[6:9],
        plan[0], plan[-1], plan.mean(axis=0),
        [np.linalg.norm(current[6:9] - requested[6:9])],
    ])
