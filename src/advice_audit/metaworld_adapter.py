"""Optional contact-task diagnostics with public expert and learned BC local.

Only loaded in .venv-metaworld, whose MuJoCo version differs from old Fetch.
Native reward, native horizon and task success are not weakened.
"""
import copy
import numpy as np

TASKS = ("push-v3", "drawer-open-v3")


def _array_paths(value, path=()):
    """Arrays inside the environment's ordinary dict/list/tuple state."""
    if isinstance(value, np.ndarray):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _array_paths(child, path+(key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _array_paths(child, path+(index,))


def _native_aliases(python_state, model, data):
    # Some native tasks store views (e.g. init_left_pad/right_pad) into data.
    # Detaching those views changes their actual reward semantics. Preserve
    # aliases instead of "fixing" upstream rewards to use constant copies.
    arrays = []
    for owner_name, owner in (("model", model), ("data", data)):
        for field in dir(owner):
            if field.startswith("_"):
                continue
            value = getattr(owner, field)
            if isinstance(value, np.ndarray) and value.flags.c_contiguous and value.nbytes:
                arrays.append((owner_name, field, value))
    aliases = []
    for path, value in _array_paths(python_state):
        for owner, field, native in arrays:
            if np.shares_memory(value, native):
                offset = value.ctypes.data-native.ctypes.data
                aliases.append(dict(path=path, owner=owner, field=field,
                                    offset=offset, shape=value.shape,
                                    strides=value.strides, dtype=value.dtype.str))
                break
    return aliases


def expert_for(task):
    from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy
    from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
    return {"push-v3": SawyerPushV3Policy, "drawer-open-v3": SawyerDrawerOpenV3Policy}[task]()


def expert_action(policy, obs):
    # Some upstream policies mutate views of their input; copy at API boundary.
    return np.clip(policy.get_action(np.asarray(obs).copy()), -1., 1.)


def bc_features(obs):
    obs = np.asarray(obs)
    if obs.shape[-1] != 39 or not np.isfinite(obs).all():
        raise ValueError("Need finite native 39D goal-observable state")
    return np.concatenate((obs[..., :3]-obs[..., 4:7], obs[..., -3:]-obs[..., 4:7],
                           obs[..., 4:7], .1*obs[..., 3:4]), axis=-1)


class BCLocal:
    def __init__(self, obs, actions, k=1):
        from scipy.spatial import cKDTree
        self.tree = cKDTree(bc_features(obs))
        self.actions = np.asarray(actions)
        self.k = k
        if self.actions.shape != (len(obs), 4) or not 1 <= k <= len(obs):
            raise ValueError("Invalid BC data/k")

    def action(self, obs):
        distances, indices = self.tree.query(bc_features(obs), k=self.k)
        distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
        weights = 1./np.maximum(distances, 1e-8)
        return np.clip(np.average(self.actions[indices], axis=0, weights=weights), -1, 1)


class ContactEnv:
    """State API with explicit full integration state and Python-state rollback.

    Branches restore into this same environment. Snapshot never enters policy
    inputs. Python state is also rolled back, including observation history and
    any task reward bookkeeping. No artificial goal disturbance is introduced.
    """
    _keep = {"model", "data", "mujoco_renderer"}

    def __init__(self, task, seed):
        import mujoco
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
        self.mujoco = mujoco
        self.task, self.seed = task, seed
        self.env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[task+"-goal-observable"](seed=seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.horizon = self.env.max_path_length
        assert self.horizon == 500
        self.seen_success = False
        self.last_success = False

    @property
    def t(self):
        return self.env.curr_path_length

    def observe(self):
        # Calling native _get_obs would modify previous-frame history.
        return self.obs.copy()

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Need finite 4D action")
        self.obs, reward, terminated, truncated, info = self.env.step(np.clip(action, -1, 1))
        self.last_success = bool(info["success"])
        self.seen_success |= self.last_success
        assert not terminated and truncated == (self.t == self.horizon)
        if self.env._did_see_sim_exception:
            raise RuntimeError("MuJoCo instability; do not hide as ordinary failure")
        return float(reward)

    def snapshot(self):
        spec = self.mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(self.mujoco.mj_stateSize(self.env.model, spec))
        self.mujoco.mj_getState(self.env.model, self.env.data, physics, spec)
        # mj_forward after integration-state restoration changes warm starts
        # and refreshes derived quantities. Native environment feedback can
        # read those quantities before the next physics step. Their tiny
        # differences can grow through contact and policy feedback, so copy
        # the complete data, including its derived quantities, for exact replay.
        python_state = {k: v for k, v in self.env.__dict__.items() if k not in self._keep}
        return dict(task=self.task, physics=physics, data=copy.copy(self.env.data),
            state_version=3,
            python=copy.deepcopy(python_state),
            native_aliases=_native_aliases(python_state, self.env.model, self.env.data),
            model_arrays={k: getattr(self.env.model, k).copy() for k in ("site_pos", "body_pos", "geom_pos")},
            obs=self.obs.copy(), seen_success=self.seen_success, last_success=self.last_success,
            global_rng=copy.deepcopy(np.random.get_state()))

    def restore(self, state):
        if state["task"] != self.task:
            raise ValueError("Task mismatch")
        if state.get("state_version") != 3:
            raise ValueError("Need alias-preserving full-MjData version3 snapshot")
        retained = {k: v for k, v in self.env.__dict__.items() if k in self._keep}
        for key, value in state["model_arrays"].items():
            getattr(self.env.model, key)[:] = value
        retained["data"] = copy.copy(state["data"])
        memo = {}
        for alias in state["native_aliases"]:
            saved = state["python"]
            for key in alias["path"]:
                saved = saved[key]
            native = getattr(retained[alias["owner"]], alias["field"])
            memo[id(saved)] = np.ndarray(alias["shape"], dtype=alias["dtype"],
                buffer=native, offset=alias["offset"], strides=alias["strides"])
        self.env.__dict__.clear()
        self.env.__dict__.update(retained)
        self.env.__dict__.update(copy.deepcopy(state["python"], memo))
        # Gym's lazy renderer stores a data reference. ACTCamera instead
        # receives the current data explicitly on every update_scene call.
        self.env.mujoco_renderer.data = self.env.data
        for viewer in self.env.mujoco_renderer._viewers.values():
            viewer.data = self.env.data
        self.obs = state["obs"].copy()
        self.seen_success, self.last_success = state["seen_success"], state["last_success"]
        np.random.set_state(copy.deepcopy(state["global_rng"]))
        return self.observe()

    def close(self):
        self.env.close()


def rollout(env, policy, held_action=None, hold=0):
    rewards = []
    for elapsed in range(env.horizon-env.t):
        action = held_action if elapsed < hold else policy.action(env.observe())
        rewards.append(env.step(action))
    return dict(success=int(env.seen_success), terminal_success=int(env.last_success),
                native_return=float(np.sum(rewards))/env.horizon)
