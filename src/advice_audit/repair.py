"""Observable-only repair controls, used to audit advice-specific value.

These are explicit simple baselines, not reimplementations of RTC/FutureRTC.
No adapter reads simulator state, future goals, branch outcomes or new expert
outputs. Source attribution is relative to the declared cheap source/family.
"""
import numpy as np

from .policies import local_action

MODES = ("reject", "raw", "prefix3", "suffix", "track")


def nominal_reference(requested, plan, backend):
    """Request-time nominal positions BEFORE each action, from visible inputs."""
    if backend not in ("toy", "fetch"):
        raise ValueError("Unknown backend")
    plan = np.asarray(plan, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if requested.shape != (12,) or plan.ndim != 2 or plan.shape[1] != 3 or not len(plan):
        raise ValueError("Need a 12D request and nonempty Hx3 plan")
    if not np.isfinite(requested).all() or not np.isfinite(plan).all():
        raise ValueError("Nonfinite plan/request")
    inertia = .6 if backend == "toy" else 0.
    position, velocity = requested[:3].copy(), np.zeros(3)
    reference = []
    for action in plan:
        reference.append(position.copy())
        velocity = inertia*velocity+(1-inertia)*.05*np.clip(action, -1, 1)
        position += velocity
    return np.asarray(reference)


def local_nominal_plan(requested, horizon=12, backend="toy", gain=6.):
    """Cheap control: roll out the unchanged local servo in a nominal model.

    This comparator has the same request observation and model constants. It
    never calls expert_plan and is not claimed to be an independent learned
    expert. Its purpose is to test whether repair needs the expert content.
    """
    if backend not in ("toy", "fetch") or horizon < 1:
        raise ValueError("Invalid backend/horizon")
    obs = np.asarray(requested, dtype=float).copy()
    velocity = np.zeros(3)
    inertia = .6 if backend == "toy" else 0.
    plan = []
    for _ in range(horizon):
        action = local_action(obs, backend) if gain == 6. else np.clip(gain*(obs[6:9]-obs[:3]), -1, 1).astype(np.float32)
        plan.append(action)
        velocity = inertia*velocity+(1-inertia)*.05*action
        obs[:3] += velocity
        obs[3:6] = obs[:3]
    return np.asarray(plan)


def repair_action(current, plan, reference, delay, elapsed, mode, feedback_gain=6.):
    """Return a candidate action or None to use the SAME local controller."""
    if mode not in MODES or delay < 0 or elapsed < 0 or feedback_gain < 0:
        raise ValueError("Invalid repair control")
    if mode == "reject":
        return None
    index = elapsed+delay if mode in ("suffix", "track") else elapsed
    limit = min(3, len(plan)) if mode == "prefix3" else len(plan)
    if index >= limit:
        return None
    action = np.asarray(plan[index], dtype=float).copy()
    if mode == "track":
        action += feedback_gain*(reference[index]-np.asarray(current)[:3])
    return np.clip(action, -1, 1).astype(np.float32)


def audited_return(env, requested, plan, delay, mode, feedback_gain=6.):
    """Offline SINGLE-intervention contribution through original deadline."""
    reference = nominal_reference(requested, plan, env.backend)
    branch = env.clone()
    try:
        rewards = []
        for elapsed in range(branch.horizon-branch.t):
            obs = branch.observe()
            action = repair_action(obs, plan, reference, delay, elapsed, mode, feedback_gain)
            if action is None:
                action = local_action(obs, branch.backend)
            _, reward, done, _ = branch.step(action)
            rewards.append(reward)
            if done:
                break
        return dict(contribution=float(np.sum(rewards))/env.horizon+int(branch.success),
                    terminal_success=int(branch.success), final_distance=float(np.linalg.norm(branch.observe()[3:6]-branch.observe()[6:9])))
    finally:
        branch.close()
