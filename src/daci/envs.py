"""Deterministic state-observation environments for intervention diagnostics.

This module is NOT a visual robotics benchmark. ``toy`` is a three-dimensional
point servo with inertia. ``fetch`` uses the real MuJoCo FetchReach-v4 model,
but exposes privileged simulator state (not pixels). The task's goal is an
observable user command. A pre-sampled change to that command occurs at one
absolute episode step; no future command is included in the observation.

Observation: [gripper_xyz, achieved_xyz, goal_xyz, observed_goal_velocity_xyz].
Success means within tolerance for the last five steps, measured at the episode
deadline. Episodes never end early just because the initial goal was reached.
Snapshots include future disturbance schedules so paired counterfactual branches
receive identical exogenous events. Snapshots are for offline training only.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np


def _action(action):
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("action must be a finite three-element vector")
    return np.clip(value, -1.0, 1.0)


class ToyReachEnv:
    """A CPU-only diagnostic: position servo with inertial command execution."""

    backend = "toy"
    dt = 0.04
    action_scale = 0.05
    inertia = 0.60
    tolerance = 0.035
    hold_steps = 5

    def __init__(self, seed=0, horizon=120, shift_scale=0.15, shift_at=40):
        if int(horizon) < self.hold_steps:
            raise ValueError("horizon must be at least five control steps")
        if not np.isfinite(shift_scale) or shift_scale < 0:
            raise ValueError("shift_scale must be finite and non-negative")
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.shift_scale = float(shift_scale)
        self.shift_at = int(shift_at)
        self._rng = np.random.RandomState(self.seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.RandomState(self.seed)
        self.position = self._rng.uniform(-0.15, 0.15, 3)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.goal = self._rng.uniform(-0.35, 0.35, 3)
        direction = self._rng.normal(size=3)
        self._goal_shift = direction / max(np.linalg.norm(direction), 1e-12) * self.shift_scale
        self.goal_velocity = np.zeros(3, dtype=np.float64)
        self.t = 0
        self._hold_count = 0
        self.last_action = np.zeros(3, dtype=np.float64)
        return self.observe()

    def observe(self):
        return np.concatenate((self.position, self.position, self.goal, self.goal_velocity)).astype(np.float32)

    @property
    def success(self):
        return bool(self._hold_count >= self.hold_steps)

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError("episode ended; call reset before stepping again")
        action = _action(action)
        self.last_action = action.copy()
        self.velocity = self.inertia * self.velocity + (1.0 - self.inertia) * self.action_scale * action
        self.position = self.position + self.velocity
        self.t += 1
        self.goal_velocity[:] = 0.0
        if self.t == self.shift_at:
            self.goal = self.goal + self._goal_shift
            self.goal_velocity = self._goal_shift / self.dt
        distance = float(np.linalg.norm(self.position - self.goal))
        self._hold_count = self._hold_count + 1 if distance < self.tolerance else 0
        info = {"distance": distance, "success": self.success, "is_success": self.success,
                "within_tolerance": bool(distance < self.tolerance), "step": self.t}
        return self.observe(), -distance, bool(self.t >= self.horizon), info

    def snapshot(self):
        return copy.deepcopy({"backend": self.backend, "seed": self.seed,
            "horizon": self.horizon, "shift_scale": self.shift_scale, "shift_at": self.shift_at,
            "position": self.position, "velocity": self.velocity, "goal": self.goal,
            "goal_shift": self._goal_shift, "goal_velocity": self.goal_velocity,
            "t": self.t, "hold_count": self._hold_count, "last_action": self.last_action,
            "rng": self._rng.get_state()})

    def restore(self, snapshot):
        state = copy.deepcopy(snapshot)
        if state["backend"] != self.backend:
            raise ValueError("snapshot backend does not match environment")
        self.seed = state["seed"]
        self.horizon, self.shift_scale, self.shift_at = state["horizon"], state["shift_scale"], state["shift_at"]
        self.position, self.velocity, self.goal = state["position"], state["velocity"], state["goal"]
        self._goal_shift, self.goal_velocity = state["goal_shift"], state["goal_velocity"]
        self.t, self._hold_count, self.last_action = state["t"], state["hold_count"], state["last_action"]
        self._rng.set_state(state["rng"])
        return self.observe()

    def clone(self):
        other = type(self)(self.seed, self.horizon, self.shift_scale, self.shift_at)
        other.restore(self.snapshot())
        return other

    def close(self):
        pass


class FetchReachEnv:
    """MuJoCo Fetch arm with an observable, externally changed reach command.

    Requires gymnasium-robotics and mujoco. No rendering/OpenGL is needed for
    state-based runs. The native task reward is replaced by negative Euclidean
    goal distance to support low-variance diagnostic return comparisons.
    """

    backend = "fetch"
    action_scale = 0.05
    tolerance = 0.05
    hold_steps = 5

    def __init__(self, seed=0, horizon=120, shift_scale=0.15, shift_at=40):
        if int(horizon) < self.hold_steps:
            raise ValueError("horizon must be at least five control steps")
        if not np.isfinite(shift_scale) or shift_scale < 0:
            raise ValueError("shift_scale must be finite and non-negative")
        try:
            import gymnasium as gym
            import gymnasium_robotics
            import mujoco
        except ImportError as exc:
            raise ImportError("Fetch backend requires Python >=3.10 and `pip install -e '.[sim]'`.") from exc
        gym.register_envs(gymnasium_robotics)
        self._mujoco = mujoco
        self.seed, self.horizon = int(seed), int(horizon)
        self.shift_scale, self.shift_at = float(shift_scale), int(shift_at)
        self._rng = np.random.RandomState(self.seed)
        self._env = gym.make("FetchReach-v4", max_episode_steps=self.horizon)
        self._base = self._env.unwrapped
        self.dt = float(self._base.dt)
        self.reset(seed=self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.RandomState(self.seed)
        # Explicitly derive a simulator seed, so wrapper RNG state alone defines
        # future resets and simulator randomization is reproducible.
        native_seed = int(self._rng.randint(0, 2 ** 31 - 1))
        self._env.reset(seed=native_seed)
        direction = self._rng.normal(size=3)
        self._goal_shift = direction / max(np.linalg.norm(direction), 1e-12) * self.shift_scale
        self.goal_velocity = np.zeros(3, dtype=np.float64)
        self.t, self._hold_count = 0, 0
        self.last_action = np.zeros(3, dtype=np.float64)
        return self.observe()

    @property
    def goal(self):
        return self._base.goal.copy()

    def observe(self):
        native = self._base._get_obs()
        return np.concatenate((native["observation"][:3], native["achieved_goal"],
                               self._base.goal, self.goal_velocity)).astype(np.float32)

    @property
    def success(self):
        return bool(self._hold_count >= self.hold_steps)

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError("episode ended; call reset before stepping again")
        action = _action(action)
        self.last_action = action.copy()
        native_action = np.concatenate((action, [0.0])).astype(np.float32)
        self._env.step(native_action)
        self.t += 1
        self.goal_velocity[:] = 0.0
        if self.t == self.shift_at:
            previous = self._base.goal.copy()
            # Restrict command changes to a conservative reach workspace.
            center = self._base.initial_gripper_xpos
            shifted = np.clip(previous + self._goal_shift, center - 0.25, center + 0.25)
            shifted[2] = max(shifted[2], 0.45)
            self._base.goal = shifted
            self.goal_velocity = (shifted - previous) / self.dt
        obs = self.observe()
        distance = float(np.linalg.norm(obs[3:6] - obs[6:9]))
        self._hold_count = self._hold_count + 1 if distance < self.tolerance else 0
        info = {"distance": distance, "success": self.success, "is_success": self.success,
                "within_tolerance": bool(distance < self.tolerance), "step": self.t}
        return obs, -distance, bool(self.t >= self.horizon), info

    def snapshot(self):
        # INTEGRATION contains time, qpos/qvel, act, warmstart, ctrl,
        # applied forces, mocap, equality activation, userdata, and plugins.
        state_spec = self._mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(self._mujoco.mj_stateSize(self._base.model, state_spec))
        self._mujoco.mj_getState(self._base.model, self._base.data, state, state_spec)
        wrapper_steps = []
        wrapper = self._env
        while wrapper is not self._base:
            wrapper_steps.append(getattr(wrapper, "_elapsed_steps", None))
            wrapper = wrapper.env
        return copy.deepcopy({"backend": self.backend, "seed": self.seed,
            "horizon": self.horizon, "shift_scale": self.shift_scale, "shift_at": self.shift_at,
            "physics": state, "goal": self._base.goal, "goal_shift": self._goal_shift,
            "goal_velocity": self.goal_velocity, "t": self.t, "hold_count": self._hold_count,
            "last_action": self.last_action, "rng": self._rng.get_state(),
            "native_rng": self._base.np_random.bit_generator.state,
            "wrapper_steps": wrapper_steps, "site_pos": self._base.model.site_pos.copy()})

    def restore(self, snapshot):
        state = copy.deepcopy(snapshot)
        if state["backend"] != self.backend or state["horizon"] != self.horizon:
            raise ValueError("snapshot backend and horizon must match environment")
        self.seed, self.shift_scale, self.shift_at = state["seed"], state["shift_scale"], state["shift_at"]
        self._base.goal = state["goal"]
        self._goal_shift, self.goal_velocity = state["goal_shift"], state["goal_velocity"]
        self.t, self._hold_count, self.last_action = state["t"], state["hold_count"], state["last_action"]
        self._rng.set_state(state["rng"])
        self._base.np_random.bit_generator.state = state["native_rng"]
        self._base.model.site_pos[:] = state["site_pos"]
        self._mujoco.mj_setState(self._base.model, self._base.data, state["physics"],
                                 self._mujoco.mjtState.mjSTATE_INTEGRATION)
        self._mujoco.mj_forward(self._base.model, self._base.data)
        wrapper = self._env
        for elapsed in state["wrapper_steps"]:
            if elapsed is not None:
                wrapper._elapsed_steps = elapsed
            wrapper = wrapper.env
        return self.observe()

    def clone(self):
        other = type(self)(self.seed, self.horizon, self.shift_scale, self.shift_at)
        other.restore(self.snapshot())
        return other

    def close(self):
        self._env.close()


def make_env(backend="toy", seed=0, horizon=120, shift_scale=0.15, shift_at=40):
    """Construct a state-diagnostic environment; no graphics or API needed."""
    classes = {"toy": ToyReachEnv, "fetch": FetchReachEnv}
    if backend not in classes:
        raise ValueError("backend must be 'toy' or 'fetch'")
    return classes[backend](seed=seed, horizon=horizon, shift_scale=shift_scale, shift_at=shift_at)
