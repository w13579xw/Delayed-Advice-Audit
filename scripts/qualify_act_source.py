"""Development-only competence and latency of a pinned visual ACT source."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np

from advice_audit.act_bridge import ACTCamera, ACTClient, source_action_to_native
from advice_audit.metaworld_adapter import ContactEnv
from server_run import ROOT, fingerprint, now, save
from strong_local_baseline import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', default='outputs/learned_source_intake/act_checkpoint')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=4200000)
    parser.add_argument('--slices', type=int, nargs='+', default=[20, 100])
    parser.add_argument('--tasks', nargs='+', default=['push-v3', 'drawer-open-v3'])
    args = parser.parse_args()
    root, checkpoint = Path(args.out).resolve(), Path(args.checkpoint).resolve()
    if root.exists():
        parser.error('New run directory required')
    if not all(1 <= n <= 100 for n in args.slices):
        parser.error('Slices must be1..100')
    expected = '7f2472be5ecf68cf99b031583c79bbd2bf8981145a4fbcba9ba3c4f69cb70d04'
    assert hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest() == expected
    root.mkdir(parents=True)
    save(root/'protocol.json', dict(config=vars(args), tier='auxiliary/development',
         frozen_utc=now(), weight_sha256=expected, native_horizon=500,
         observation='corner4 vertically flipped + black wrist placeholder + native robot4D',
         action='negate x, clip to native[-1,1]', device='cpu', chunk_size=100))
    for file in [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py')]:
        dest = root/'source_snapshot'/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file.read_bytes())
    code = fingerprint()
    state = dict(status='running', created_utc=now(), source_fingerprint=code)
    save(root/'run.json', state)
    client = None
    rows, latency = [], []
    t0 = time.monotonic()
    try:
        client = ACTClient(ROOT/'.venv-act/bin/python', ROOT/'scripts/act_chunk_worker.py', checkpoint, root/'ipc')
        save(root/'worker_ready.json', client.ready)
        for ti, task in enumerate(args.tasks):
            for execution_slice in args.slices:
                for ep in range(args.episodes):
                    seed = args.seed+ti*100000+ep
                    env = ContactEnv(task, seed)
                    camera = ACTCamera(env)
                    rewards, actions, chunks, requests = [], [], [], []
                    first = None
                    try:
                        while env.t < 500:
                            step = env.t
                            render_start = time.monotonic()
                            image = camera.image(env)
                            render_seconds = time.monotonic()-render_start
                            obs = env.observe()[:4].astype('float32')
                            wall_start = time.monotonic()
                            prediction, timing = client.predict(obs, image)
                            wall_seconds = time.monotonic()-wall_start
                            latency.append(dict(task=task, seed=seed, slice=execution_slice, step=step,
                                                infer_seconds=timing['seconds'], render_seconds=render_seconds,
                                                ipc_wall_seconds=wall_seconds))
                            if not rows and not requests:
                                repeat, _ = client.predict(obs, image)
                                assert np.array_equal(prediction, repeat), 'Non-deterministic ACT inference'
                            chunks.append(prediction[0])
                            requests.append(dict(step=step, state=obs.tolist(),
                                                 image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                                 chunk_sha256=hashlib.sha256(prediction.tobytes()).hexdigest()))
                            native = source_action_to_native(prediction[0])
                            for action in native[:min(execution_slice, 500-step)]:
                                actions.append(action)
                                rewards.append(env.step(action))
                                if env.seen_success and first is None:
                                    first = env.t
                        row = dict(task=task, seed=seed, slice=execution_slice,
                                   success=int(env.seen_success), terminal_success=int(env.last_success),
                                   native_return=float(np.mean(rewards)), first_success_step=first,
                                   requests=len(requests))
                        stem = f'{task}_{execution_slice}_{seed}'
                        np.savez_compressed(root/(stem+'.npz'), action=actions, reward=rewards, chunks=chunks)
                        save(root/(stem+'_requests.json'), requests)
                        rows.append(row)
                        save(root/'episodes.json', rows)
                        save(root/'latency.json', latency)
                        print('EPISODE', row, flush=True)
                    except BaseException as episode_error:
                        np.savez_compressed(root/'failed_episode_trace.npz', action=actions, reward=rewards, chunks=chunks)
                        save(root/'failed_episode.json', dict(task=task, seed=seed,
                             slice=execution_slice, step=env.t, error=str(episode_error),
                             status='failed_not_filtered'))
                        raise
                    finally:
                        camera.close()
                        env.close()
        client.close()
        client = None
        result = {task: {str(n): summary([r for r in rows if r['task']==task and r['slice']==n])
                        for n in args.slices} for task in args.tasks}
        save(root/'analysis.json', dict(metrics=result, infer_seconds={
             key: float(np.quantile([x['infer_seconds'] for x in latency], q))
             for key, q in [('median', .5), ('p95', .95)]},
             limitation='Development only; no inference delay, no advice selector, no new-method claim'))
        assert code == fingerprint()
        save(root/'checks.json', dict(checks='passed', weights_verified=True,
             deterministic_first_request=True, all_episodes_full500=True))
        state.update(status='complete', completed_utc=now(), exit_code=0, seconds=time.monotonic()-t0)
    except BaseException as error:
        state.update(status='failed', error=str(error), ended_utc=now())
        save(root/'run.json', state)
        if client is not None:
            client.close()
        raise
    save(root/'run.json', state)
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
