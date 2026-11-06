"""Clean native qualification of a frozen standalone ACT source; no gate."""
import argparse,hashlib,json,tempfile,shutil,time
from pathlib import Path
import numpy as np
from advice_audit.act_bridge import ACTClient,ACTCamera,source_action_to_native
from advice_audit.metaworld_adapter import ContactEnv
from server_run import ROOT,fingerprint,save,now
from strong_local_baseline import summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--task',required=True);p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,required=True);p.add_argument('--slices',type=int,nargs='+',default=[1,20,100]);p.add_argument('--offsamples',type=int,choices=[0,4],default=None);a=p.parse_args();model=Path(a.model).resolve();root=Path(a.out).resolve();metadata=json.loads((model/'metadata.json').read_text())
 if root.exists():p.error('New run directory required')
 if metadata.get('smoke_only',False):p.error('Cannot use a smoke model for competence claims')
 if (model.parent/'checks.json').exists() and json.loads((model.parent/'checks.json').read_text()).get('smoke_only',False):p.error('Cannot use a smoke training run')
 assert all(1<=n<=100 for n in a.slices);assert metadata['task'] in (a.task,'mt50');root.mkdir(parents=True);code=fingerprint();start=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state);save(root/'protocol.json',dict(config=vars(a),metadata=metadata,frozen_utc=now(),native_horizon=500,inputs=['RGB+robot4only'],experiment='synchronous clean source qualification; not asynchronous controller efficacy',slices1='fresh-feedback clean reference; actual latency reported, no real-time claim'))
 for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  d=root/'source_snapshot'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
 rows=[];latency=[];client=None;temporary=tempfile.TemporaryDirectory(prefix='advice_audit_act_',dir='/dev/shm')
 try:
  client=ACTClient(ROOT/'.venv-vision/bin/python',ROOT/'scripts/matched_act_worker.py',model,Path(temporary.name)/'ipc');save(root/'worker_ready.json',client.ready)
  for execution in a.slices:
   for seed in range(a.seed,a.seed+a.episodes):
    env=ContactEnv(a.task,seed);env.env.model.vis.quality.offsamples=a.offsamples if a.offsamples is not None else env.env.model.vis.quality.offsamples;camera=ACTCamera(env);actions=[];rewards=[];chunks=[];first=None
    try:
     while env.t<500:
      step=env.t;robot=env.observe()[:4].astype('float32');image=camera.image(env);begin=time.monotonic();chunk,timing=client.predict(robot,image);latency.append(dict(seed=seed,slice=execution,step=step,infer_seconds=timing['seconds'],rpc_seconds=time.monotonic()-begin));chunks.append(chunk[0]);native=source_action_to_native(chunk[0])
      for action in native[:min(execution,500-step)]:
       actions.append(action);rewards.append(env.step(action))
       if env.seen_success and first is None:first=env.t
     row=dict(seed=seed,slice=execution,success=int(env.seen_success),terminal_success=int(env.last_success),native_return=float(np.mean(rewards)),first_success_step=first,requests=len(chunks));rows.append(row);save(root/'episodes.json',rows);save(root/'latency.json',latency);np.savez_compressed(root/f'episode_{seed}_slice{execution}.npz',actions=actions,rewards=rewards,chunks=chunks);print('EPISODE',a.task,row,flush=True)
    except BaseException as e:save(root/'failed_episode.json',dict(seed=seed,slice=execution,step=env.t,error=str(e)));np.savez_compressed(root/'failed_trace.npz',actions=actions,rewards=rewards,chunks=chunks);raise
    finally:camera.close();env.close()
  client.close();client=None;shutil.copyfile(Path(temporary.name)/'ipc/worker_stderr.log',root/'worker_stderr.log');assert code==fingerprint();save(root/'analysis.json',dict(metrics={str(n):summary([r for r in rows if r['slice']==n]) for n in a.slices},infer_seconds_median=float(np.median([r['infer_seconds'] for r in latency])),source_metadata=metadata));save(root/'checks.json',dict(checks='passed',episodes=len(rows),all_native500=True,smoke_only=False));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-start,exit_code=0)
 except BaseException as e:
  state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state)
  if client is not None:client.close();shutil.copyfile(Path(temporary.name)/'ipc/worker_stderr.log',root/'worker_stderr.log')
  raise
 finally:temporary.cleanup()
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
