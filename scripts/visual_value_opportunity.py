"""Arrival opportunity against a broad, sensory-matched cheap repair set."""
import argparse,hashlib,json,tempfile,shutil,sys,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import numpy as np
from advice_audit.act_bridge import ACTCamera
from advice_audit.metaworld_adapter import ContactEnv
from visual_bundle import BundleClient,actions,make_manifest,DeterministicCamera
from server_run import save,now,fingerprint

def metrics(rewards,env):return dict(success=int(env.seen_success),terminal_success=int(env.last_success),native_return=float(np.mean(rewards)))
def summarize(rows):
 result={}
 for task in sorted({r['task'] for r in rows}):
  rr=[r for r in rows if r['task']==task];names=list(rr[0]['outcomes']);cheap=[n for n in names if not n.startswith('act_')];expensive=[n for n in names if n.startswith('act_')];by={n:{m:float(np.mean([r['outcomes'][n][m] for r in rr])) for m in ('success','terminal_success','native_return')} for n in names};gaps={}
  for m in ('success','terminal_success','native_return'):
   c=np.array([max(r['outcomes'][n][m] for n in cheap) for r in rr]);e=np.array([max(r['outcomes'][n][m] for n in expensive) for r in rr]);gaps[m]=dict(signed_oracle_gap=float(np.mean(e-c)),nonnegative_oracle_opportunity=float(np.mean(np.maximum(e-c,0))),note='Hindsight maxima; neither is a deployed selector effect')
  result[task]=dict(rows=len(rr),individual_options=by,oracle=gaps)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args();root=Path(a.out).resolve()
 if root.exists():p.error('New directory required')
 root.mkdir();code=fingerprint();start=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 for source_file in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  destination=root/'source_snapshot'/source_file.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source_file.read_bytes())

 queries=[64] if a.profile else [32,128];delays=[0,16] if a.profile else [0,16,32];episodes=1 if a.profile else 2;tasks=[('push-v3',60400100 if a.profile else 61200000),('pick-place-v3',60500100 if a.profile else 61300000)];save(root/'protocol.json',dict(frozen_utc=now(),tasks=tasks,episodes=episodes,queries=queries,delays=delays,profile_only=a.profile,primary='native any-success',secondary=['cutoff-success','mean native reward'],native_horizon=500,input='RGB+robot4only',arrival='Local control advances; same source packet per request for every delay; response inaccessible to any action until arrival',cheap_union='offline hindsight diagnostic; not deployable baseline',source='matched ACT,source convention already converted/clipped by worker',single_intervention=True))
 rows=[];timings=[]
 try:
  for task,seedbase in tasks:
   manifest,path=make_manifest(task,root,ROOT);local=manifest['selected'];cheap_other=[i for i in range(4) if i!=local];options=['reject']+[f'cheap_head{i}_48' for i in cheap_other]+['cheap_ensemble48','cheap_request48','cheap_arrival48','act_raw16','act_raw48','act_suffix16','act_suffix48','act_blend48']
   with tempfile.TemporaryDirectory(prefix='advice_audit_value_',dir='/dev/shm') as temp:
    work=Path(temp);client=BundleClient(ROOT/'.venv-vision/bin/python',Path(__file__).parent/'visual_bundle_worker.py',path,work/'bundle');save(root/(task+'_worker_ready.json'),client.ready)
    try:
     for seed in range(seedbase,seedbase+episodes):
      env=ContactEnv(task,seed);camera=DeterministicCamera(env);baseline_rewards=[];snapshots={};packets={};observed={};baseline_actions=[]
      try:
       while env.t<500:
        t=env.t
        if t in {q+d for q in queries for d in delays}:snapshots[t]=env.snapshot()
        robot=env.observe()[:4].astype('float32');image=camera.image(env);prediction,timing=client.predict(robot,image,source=t in queries);aa=actions(task,robot,prediction,manifest);timings.append(timing['seconds'])
        if t in queries:packets[t]=dict(source=prediction['source'].copy(),request_action=aa[local].copy(),robot=robot.copy(),predictions={k:v.copy() for k,v in prediction.items() if k!='source'},image_sha256=hashlib.sha256(image.tobytes()).hexdigest());np.savez_compressed(root/f'packet_{task}_{seed}_{t}.npz',**packets[t]['predictions'],source=packets[t]['source'],robot=robot)
        if t in snapshots:observed[t]=dict(robot=robot.copy(),predictions={k:v.copy() for k,v in prediction.items() if k!='source'},actions=aa.copy())
        action=aa[local];baseline_actions.append(action);baseline_rewards.append(env.step(action))
       base=metrics(baseline_rewards,env);np.savez_compressed(root/f'baseline_{task}_{seed}.npz',actions=baseline_actions,rewards=baseline_rewards)
       for q in queries:
        packet=packets[q];source=packet['source'];packet_hash=hashlib.sha256(source.tobytes()).hexdigest()
        for delay in delays:
         arrival=q+delay;snapshot=snapshots[arrival];arrival_obs=observed[arrival];outcomes={};traces={}
         for name in options:
          if name=='reject':
           outcomes[name]=dict(base)
           # Profile performs the actual full-tail negative-control replay.
           if a.profile and delay==0:
            env.restore(snapshot);rr=list(baseline_rewards[:arrival])
            while env.t<500:
             robot=env.observe()[:4].astype('float32');prediction,_=client.predict(robot,camera.image(env));action=actions(task,robot,prediction,manifest)[local];np.testing.assert_array_equal(action,baseline_actions[env.t]);rr.append(env.step(action));assert rr[-1]==baseline_rewards[env.t-1]
            assert metrics(rr,env)==base
           continue
          if delay==0 and name.startswith('act_suffix'):
           raw=name.replace('suffix','raw');outcomes[name]=dict(outcomes[raw]);traces[name]=dict(identical_to=raw,reason='delay0:exact same actions and environment');continue
          env.restore(snapshot);reward=list(baseline_rewards[:arrival]);trace_action=[]
          hold=16 if name.endswith('16') else 48
          while env.t<500:
           elapsed=env.t-arrival;robot=env.observe()[:4].astype('float32');prediction,_=client.predict(robot,camera.image(env));aa=actions(task,robot,prediction,manifest);local_action=aa[local]
           if elapsed>=hold:action=local_action
           elif name.startswith('cheap_head'):action=aa[int(name[len('cheap_head')])]
           elif name=='cheap_ensemble48':action=np.mean(np.stack(aa),axis=0)
           elif name=='cheap_request48':action=packet['request_action']
           elif name=='cheap_arrival48':action=arrival_obs['actions'][local]
           elif name.startswith('act_raw'):action=source[elapsed]
           elif name.startswith('act_suffix'):action=source[delay+elapsed]
           elif name=='act_blend48':action=.5*source[delay+elapsed]+.5*local_action
           else:raise ValueError(name)
           trace_action.append(action);reward.append(env.step(action))
          outcomes[name]=metrics(reward,env);traces[name]=dict(actions=np.array(trace_action),rewards=np.array(reward))
          print('OPTION',task,seed,q,delay,name,outcomes[name],flush=True)
         assert outcomes['reject']==base
         row=dict(task=task,seed=seed,query=q,delay=delay,arrival=arrival,source_packet_sha256=packet_hash,outcomes=outcomes,observed_features=dict(request_robot=packet['robot'].tolist(),arrival_robot=arrival_obs['robot'].tolist(),request_predictions={k:v.tolist() for k,v in packet['predictions'].items()},arrival_predictions={k:v.tolist() for k,v in arrival_obs['predictions'].items()}));rows.append(row);save(root/'rows.json',rows);save(root/'analysis.json',summarize(rows))
         for name,trace in traces.items():
          if 'actions' in trace:np.savez_compressed(root/f'trace_{task}_{seed}_q{q}_d{delay}_{name}.npz',**trace)
         print('ROW',task,seed,q,delay,'seconds',time.monotonic()-start,flush=True)
      finally:camera.close();env.close()
    finally:client.close();shutil.copyfile(work/'bundle/worker_stderr.log',root/(task+'_worker_stderr.log'))
  assert code==fingerprint();expected=len(tasks)*episodes*len(queries)*len(delays);assert len(rows)==expected
  for task,seedbase in tasks:
   for seed in range(seedbase,seedbase+episodes):
    ss=[r for r in rows if r['task']==task and r['seed']==seed]
    assert all(r['outcomes']['reject']==ss[0]['outcomes']['reject'] for r in ss)
    for q in queries:assert len({r['source_packet_sha256'] for r in ss if r['query']==q})==1
  save(root/'checks.json',dict(checks='passed',rows=len(rows),options=12,same_packet_across_delays=True,reject_equal_uninterrupted=True,profile_only=a.profile));save(root/'timing.json',dict(seconds=time.monotonic()-start,worker_seconds_median=float(np.median(timings)),rows=len(rows)));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-start,exit_code=0)
 except BaseException as e:state.update(status='failed',ended_utc=now(),error=str(e));save(root/'run.json',state);raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
