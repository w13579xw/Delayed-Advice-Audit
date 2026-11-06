"""Qualify public expert and a non-weakened learned local before delay audits."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
import numpy as np

from advice_audit.metaworld_adapter import BCLocal, ContactEnv, TASKS, expert_action, expert_for, rollout
from server_run import ROOT, fingerprint, now, save


class Expert:
    def __init__(self, task):
        self.policy = expert_for(task)

    def action(self, obs):
        return expert_action(self.policy, obs)


def evaluate(task, policy, seeds):
    rows = []
    for seed in seeds:
        env = ContactEnv(task, seed)
        try:
            rows.append(dict(seed=seed, **rollout(env, policy)))
        finally:
            env.close()
    return rows


def fit_task(root, task, offset):
    observations, actions, episodes = [], [], []
    teacher = Expert(task)
    for seed in range(81000+offset, 81200+offset):
        env = ContactEnv(task, seed)
        try:
            while env.t < env.horizon:
                obs = env.observe()
                action = teacher.action(obs)
                observations.append(obs)
                actions.append(action)
                episodes.append(seed)
                env.step(action)
        finally:
            env.close()
        if (seed-81000-offset+1) % 40 == 0:
            print(task, "DEMOS", seed-81000-offset+1, flush=True)
    observations, actions, episodes = map(np.asarray, (observations, actions, episodes))
    np.savez_compressed(root/(task+"_demos.npz"), obs=observations, action=actions, episode=episodes)
    scores, choice = [], None
    for n in (40, 100, 200):
        mask = episodes < 81000+offset+n
        for k in (1, 3, 5):
            model = BCLocal(observations[mask], actions[mask], k)
            result = evaluate(task, model, range(82000+offset, 82008+offset))
            score = dict(episodes=n, k=k, success=float(np.mean([x["success"] for x in result])), rows=result)
            scores.append(score)
        qualified = [r for r in scores if r["episodes"] == n and r["success"] >= .875]
        if qualified:
            choice = max(qualified, key=lambda r: (r["success"], -r["k"]))
            break
    save(root/(task+"_validation.json"), dict(results=scores, selected=choice))
    if choice is None:
        return dict(task=task, qualified=False, reason="BC validation below 7/8; no independent test opened")
    mask = episodes < 81000+offset+choice["episodes"]
    np.savez_compressed(root/(task+"_local.npz"), obs=observations[mask], action=actions[mask], k=choice["k"], episode=episodes[mask])
    save(root/(task+"_freeze.json"), dict(selected=choice, frozen_utc=now(),
        model_sha256=hashlib.sha256((root/(task+"_local.npz")).read_bytes()).hexdigest()))
    local = BCLocal(observations[mask], actions[mask], choice["k"])
    test_seeds = range(83000+offset, 83020+offset)
    results = {"public_expert": evaluate(task, teacher, test_seeds), "bc_local": evaluate(task, local, test_seeds)}
    rates = {k: float(np.mean([r["success"] for r in rows])) for k, rows in results.items()}
    return dict(task=task, selected_train_episodes=choice["episodes"], k=choice["k"], test=results,
        success_rates=rates, qualified=all(value >= .8 for value in rates.values()),
        note="20 held-out native 500-step episodes; report all tasks, no failed-task deletion")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out).resolve()
    if root.exists():
        p.error("Need new output directory")
    root.mkdir(parents=True)
    save(root/"protocol.json", dict(tasks=TASKS, demo_seed=81000, validation_seed=82000, test_seed=83000,
        task_offsets=[0, 10000], max_demos=200, ladder=[40, 100, 200], k=[1, 3, 5],
        validation_episodes=8, test_episodes=20, validation_threshold=.875, qualification=.8,
        horizon=500, frozen_utc=now(), package_versions={x: importlib.metadata.version(x) for x in ("metaworld", "mujoco", "numpy", "scipy", "gymnasium")}))
    code_hash = fingerprint()
    state = dict(status="running", created_utc=now(), source_fingerprint=code_hash)
    save(root/"run.json", state)
    for file in [*ROOT.glob("src/**/*.py"), Path(__file__).resolve(), ROOT/"requirements-metaworld.txt"]:
        dest = root/"source_snapshot"/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file.read_bytes())
    for task in TASKS:
        path = Path(inspect.getfile(type(expert_for(task))))
        dest = root/"public_expert_source"/path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    try:
        results = [fit_task(root, task, offset) for task, offset in zip(TASKS, (0, 10000))]
        save(root/"analysis.json", dict(results=results, note="Capability is prerequisite, not evidence for advice or novelty"))
        assert fingerprint() == code_hash
        save(root/"checks.json", dict(checks="passed", all_tasks_recorded=True, test_after_model_freeze=True))
        state.update(status="complete", completed_utc=now(), exit_code=0)
    except BaseException as error:
        state.update(status="failed", error=str(error), ended_utc=now())
        save(root/"run.json", state)
        raise
    save(root/"run.json", state)
    print("COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
