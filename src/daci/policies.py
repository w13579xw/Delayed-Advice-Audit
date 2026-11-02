"""Small deterministic controllers, not trained VLA or language models.

The local policy is a feedback proportional servo. The nominal 'expert' plans
an open-loop sequence against its request-time goal and approximate dynamics.
It is deliberately NOT an oracle: inertia, unobserved velocity, goal changes,
and MuJoCo/model mismatch can make its advice harmful. A higher nominal gain
can improve short-horizon progress, but no superiority is guaranteed.

Replace these diagnostic policies with learned policies for publication-scale
experiments. Both policies receive the same observable 12D state at query time.
"""

import numpy as np


def _observation(obs):
    value = np.asarray(obs, dtype=np.float64)
    if value.shape != (12,) or not np.all(np.isfinite(value)):
        raise ValueError("expected a finite 12D observation")
    return value


def local_action(obs, backend="toy"):
    """Reactive position servo; the backend argument preserves adapter symmetry."""
    value = _observation(obs)
    if backend not in ("toy", "fetch"):
        raise ValueError("backend must be 'toy' or 'fetch'")
    return np.clip(6.0 * (value[6:9] - value[:3]), -1.0, 1.0).astype(np.float32)


def expert_plan(obs, horizon=12, backend="toy"):
    """Return Hx3 commands planned from an old observation and frozen goal.

    Position prediction uses nominal scale 0.05m/step. Toy inertia is known;
    initial velocity is unavailable and starts at zero. Fetch prediction is a
    kinematic approximation and does not access the simulator or future goal.
    This is a controlled staleness probe, not a claim about VLM performance.
    """
    value = _observation(obs)
    if int(horizon) < 1:
        raise ValueError("plan horizon must be positive")
    if backend not in ("toy", "fetch"):
        raise ValueError("backend must be 'toy' or 'fetch'")
    position, goal = value[:3].copy(), value[6:9].copy()
    velocity = np.zeros(3, dtype=np.float64)
    inertia = 0.60 if backend == "toy" else 0.0
    actions = []
    for _ in range(int(horizon)):
        action = np.clip(12.0 * (goal - position) - 8.0 * velocity, -1.0, 1.0)
        actions.append(action)
        velocity = inertia * velocity + (1.0 - inertia) * 0.05 * action
        position += velocity
    return np.asarray(actions, dtype=np.float32)
