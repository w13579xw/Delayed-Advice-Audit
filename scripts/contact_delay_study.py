"""External contact-task attribution; frozen public actions and BC controls."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
from advice_audit.metaworld_adapter import BCLocal, ContactEnv, expert_for, expert_action, rollout
from advice_audit.report import paired_ci
from server_run import ROOT, fingerprint, now, save

SOURCES = ('public_requested', 'bc_requested', 'bc_arrival')
HOLDS = (0, 1, 3, 12)


def read(path):
    return json.loads(path.read_text())


def local_model(parent, task):
    path = parent/(task+'_local.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == read(parent/(task+'_freeze.json'))['model_sha256']
    with np.load(path, allow_pickle=False) as d:
        return BCLocal(d['obs'], d['action'], int(d['k']))


def collect(task, local, seeds, queries):
    records = []
    teacher = expert_for(task)
    for seed in seeds:
        env = ContactEnv(task, seed)
        try:
            while env.t <= max(queries):
                if env.t in queries:
                    step, requested = env.t, env.observe()
                    public = expert_action(teacher, requested)
                    cheap = local.action(requested)
                    original = env.snapshot()
                    for delay in (0, 4, 12):
                        env.restore(original)
                        for _ in range(delay):
                            env.step(local.action(env.observe()))
                        current = env.observe()
                        actions = np.array([public, cheap, local.action(current)])
                        hashes = [hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest() for a in actions]
                        arrival = env.snapshot()
                        control = rollout(env, local)
                        values = {key: np.zeros((3, 4)) for key in ('success', 'terminal_success', 'native_return')}
                        for source in range(3):
                            for m, hold in enumerate(HOLDS[1:], 1):
                                env.restore(arrival)
                                value = rollout(env, local, actions[source], hold)
                                for key in values:
                                    values[key][source, m] = value[key]-control[key]
                        records.append(dict(episode=seed, request_step=step, delay=delay, requested=requested,
                            current=current, action=actions, plan_id=hashes,
                            success_gain=values['success'], terminal_gain=values['terminal_success'],
                            return_gain=values['native_return'], baseline_success=control['success']))
                    env.restore(original)
                env.step(local.action(env.observe()))
        finally:
            env.close()
        print(task, 'COLLECTED', seed, len(records), flush=True)
    result = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    assert np.isfinite(result['return_gain']).all()
    assert not result['success_gain'][:, :, 0].any() and not result['return_gain'][:, :, 0].any()
    for ep, step in set(zip(result['episode'].tolist(), result['request_step'].tolist())):
        mask = (result['episode'] == ep) & (result['request_step'] == step)
        for source in (0, 1):
            assert len(set(result['plan_id'][mask, source])) == 1
    return result


def best_fixed(data, sources):
    success, reward = data['success_gain'].mean(axis=0), data['return_gain'].mean(axis=0)
    best = max(((float(success[s, m]), float(reward[s, m]), -m, -s, s, m)
                for s in sources for m in range(4)))
    return dict(source=int(best[-2]), mode=int(best[-1]), hold=HOLDS[best[-1]],
                validation_success_gain=best[0], validation_return_gain=best[1])


def episode_summary(values, groups):
    means = [float(values[groups == ep].mean()) for ep in sorted(set(groups.tolist()))]
    return dict(mean=float(np.mean(means)), ci95=paired_ci(means), episode_means=means)


def offline_analysis(data, choices):
    metrics = {}
    for target in ('success_gain', 'terminal_gain', 'return_gain'):
        g = data[target]
        full, cheap = g.max(axis=(1, 2)), g[:, 1:].max(axis=(1, 2))
        values = dict(full_realized_oracle=full, cheap_realized_oracle=cheap,
            expert_portfolio_increment=full-cheap,
            public_raw12_accept_reject_oracle=np.maximum(g[:, 0, 3], 0.),
            public_duration_oracle=g[:, 0].max(axis=1))
        for name, choice in choices.items():
            values[name] = g[:, choice['source'], choice['mode']]
        values['full_fixed_minus_cheap_fixed'] = values['full_fixed']-values['cheap_fixed']
        metrics[target] = {key: episode_summary(value, data['episode']) for key, value in values.items()}
    return metrics


def closed_episode(task, local, seed, delay, choice, queries):
    env, teacher = ContactEnv(task, seed), expert_for(task)
    pending, held, left, rewards, requests = None, None, 0, [], 0
    try:
        while env.t < env.horizon:
            if env.t in queries:
                assert pending is None and left == 0
                obs = env.observe()
                pending = (env.t+delay, expert_action(teacher, obs), local.action(obs))
                requests += 1
            if pending is not None and pending[0] == env.t:
                options = (pending[1], pending[2], local.action(env.observe()))
                held, left = options[choice['source']], choice['hold']
                pending = None
            action = held if left else local.action(env.observe())
            rewards.append(env.step(action))
            left = max(0, left-1)
        assert requests == len(queries)
        return dict(seed=seed, delay=delay, requests=requests, success=int(env.seen_success),
            terminal_success=int(env.last_success), native_return=float(np.sum(rewards))/env.horizon)
    finally:
        env.close()


def closed_analysis(rows):
    results = {}
    for delay in (0, 4, 12):
        subset = [r for r in rows if r['delay'] == delay]
        base = {r['seed']: r for r in subset if r['method'] == 'reject'}
        results[str(delay)] = {}
        for method in ('reject', 'full_fixed', 'cheap_fixed'):
            chosen = sorted([r for r in subset if r['method'] == method], key=lambda r: r['seed'])
            assert {r['seed'] for r in chosen} == set(base)
            result = {}
            for key in ('success', 'terminal_success', 'native_return'):
                differences = np.array([r[key]-base[r['seed']][key] for r in chosen])
                result[key] = dict(mean=float(np.mean([r[key] for r in chosen])),
                    paired_delta=float(differences.mean()), ci95=paired_ci(differences))
            results[str(delay)][method] = result
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    parent, root = Path(a.parent).resolve(), Path(a.out).resolve()
    if root.exists():
        p.error('Need new directory')
    assert read(parent/'run.json')['status'] == 'complete'
    qualifications = read(parent/'analysis.json')['results']
    tasks = [r['task'] for r in qualifications if r['qualified']]
    assert tasks, 'No qualified tasks; do not run weak baseline experiment'
    root.mkdir(parents=True)
    c = dict(parent=str(parent), tasks=tasks, sources=SOURCES, holds=HOLDS, horizon=500,
        queries=[24] if a.smoke else [24, 96, 240], delays=[0, 4, 12], validation_episodes=1 if a.smoke else 8,
        test_episodes=1 if a.smoke else 20, validation_seed=1085000 if a.smoke else 85000,
        test_seed=1086000 if a.smoke else 86000, task_offsets={'push-v3':0, 'drawer-open-v3':10000},
        primary='native any-success by step500', secondary=['native cutoff success', 'native dense sum/500'],
        note='State observations, public single action held fixed, synthetic delay, shared demonstration source; not VLA chunks')
    save(root/'protocol.json', dict(config=c, frozen_utc=now()))
    save(root/'environment.json', dict(python=sys.version, versions={x:importlib.metadata.version(x) for x in ('metaworld','mujoco','numpy','scipy','gymnasium')}))
    for path in [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests_metaworld/*.py'), ROOT/'requirements-metaworld.txt']:
        dest = root/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    code_hash = fingerprint()
    state = dict(status='running', created_utc=now(), source_fingerprint=code_hash, phases={})
    save(root/'run.json', state)
    report = dict(tasks={}, qualification=qualifications)
    try:
        for task in tasks:
            local = local_model(parent, task)
            offset = c['task_offsets'][task]
            start = c['validation_seed']+offset
            val = collect(task, local, range(start, start+c['validation_episodes']), c['queries'])
            np.savez_compressed(root/(task+'_validation.npz'), **val)
            choices = dict(full_fixed=best_fixed(val, (0, 1, 2)), cheap_fixed=best_fixed(val, (1, 2)), public_fixed=best_fixed(val, (0,)))
            assert not (root/(task+'_test.npz')).exists()
            save(root/(task+'_selection.json'), dict(choices=choices, frozen_utc=now()))
            print(task, 'FROZEN', choices, flush=True)
            start = c['test_seed']+offset
            test_started = now()
            test = collect(task, local, range(start, start+c['test_episodes']), c['queries'])
            np.savez_compressed(root/(task+'_test.npz'), **test)
            closed = []
            for seed in range(start, start+c['test_episodes']):
                for delay in c['delays']:
                    for method in ('reject', 'full_fixed', 'cheap_fixed'):
                        choice = dict(source=0, hold=0) if method == 'reject' else choices[method]
                        closed.append(dict(method=method, **closed_episode(task, local, seed, delay, choice, c['queries'])))
                print(task, 'CLOSED', seed, flush=True)
            save(root/(task+'_closed.json'), closed)
            assert read(root/(task+'_selection.json'))['frozen_utc'] <= test_started
            report['tasks'][task] = dict(choices=choices, offline=offline_analysis(test, choices), closed=closed_analysis(closed),
                validation_rows=len(val['episode']), test_rows=len(test['episode']), closed_rows=len(closed))
            state['phases'][task] = dict(status='complete', ended_utc=now(), exit_code=0)
            save(root/'run.json', state)
            save(root/'analysis.json', report)
        assert fingerprint() == code_hash
        save(root/'checks.json', dict(checks='passed', selection_before_test=True, fixed_requested_hashes=True,
            native_horizon=500, shared_arrival_and_deadline=True, all_qualification_results_retained=True))
        state.update(status='complete', completed_utc=now(), exit_code=0)
    except BaseException as error:
        state.update(status='failed', error=str(error), ended_utc=now())
        save(root/'run.json', state)
        raise
    save(root/'run.json', state)
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
