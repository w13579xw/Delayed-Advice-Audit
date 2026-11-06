"""Matched one-request ACT/cheap repair opportunity; development data only."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from advice_audit.act_bridge import ACTCamera,ACTClient,source_action_to_native
from advice_audit.metaworld_adapter import BCLocal,ContactEnv,bc_features
from server_run import ROOT,fingerprint,now,save

OPTIONS=['reject','cheap_request16','cheap_arrival16','cheap_dwell16','cheap_ensemble48','act_raw16','act_raw48','act_suffix16','act_suffix48','act_blend48']

def features(request,arrival,chunk,local_req,local_now,q,d):
 return np.concatenate([bc_features(request),bc_features(arrival),arrival[:18],arrival[-3:],arrival[:18]-request[:18],local_req,local_now,chunk[:8].ravel(),chunk[:16].mean(0),chunk[:48].mean(0),chunk.mean(0),chunk.std(0),[q/500,d/100]]).astype('float64')

def branch(env,local,ensemble,chunk,option,request_action,arrival_action,delay,past_rewards):
 rewards=list(past_rewards);start=env.t
 while env.t<500:
  elapsed=env.t-start;action=local.action(env.observe())
  if option=='cheap_request16' and elapsed<16:action=request_action
  elif option=='cheap_arrival16' and elapsed<16:action=arrival_action
  elif option=='cheap_dwell16' and elapsed<16:action=np.array([0.,0.,0.,arrival_action[3]])
  elif option=='cheap_ensemble48' and elapsed<48:action=np.mean([p.action(env.observe()) for p in ensemble],axis=0)
  elif option.startswith('act_'):
   length=16 if option.endswith('16') else 48
   if elapsed<length:
    idx=delay+elapsed if 'suffix' in option else elapsed
    action=chunk[idx]
    if 'blend' in option:action=.5*action+.5*local.action(env.observe())
  rewards.append(env.step(action))
 return [int(env.seen_success),int(env.last_success),float(np.sum(rewards)/500)]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=8);p.add_argument('--seed',type=int,default=4700000);p.add_argument('--queries',type=int,nargs='+',default=[16,64,192]);p.add_argument('--delays',type=int,nargs='+',default=[0,16,32]);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 root=Path(a.out).resolve()
 if root.exists():p.error('new directory required')
 root.mkdir(parents=True);weights=ROOT/'outputs/learned_source_intake/act_checkpoint/model.safetensors';assert hashlib.sha256(weights.read_bytes()).hexdigest()=='7f2472be5ecf68cf99b031583c79bbd2bf8981145a4fbcba9ba3c4f69cb70d04';code=fingerprint();t0=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 sources={'push-v3':ROOT/'outputs/learned_advice/strong_local_20261103T022217Z/push-v3_local.npz','pick-place-v3':ROOT/'outputs/learned_advice/task_development_20261103_v1/pick-place-v3_local.npz'}
 save(root/'protocol.json',dict(frozen_utc=now(),config=vars(a),tier='auxiliary/development',sources={t:dict(path=str(s),sha256=hashlib.sha256(s.read_bytes()).hexdigest()) for t,s in sources.items()},options=OPTIONS,metric_order=['native_any_success','native_cutoff_success','mean_native_reward'],native_horizon=500,single_request=True,delay='synthetic, local continues, fixed request content',privilege='state-based selector; oracle outcomes never inputs'))
 for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  dest=root/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
 rows=[];xs=[];ys=[];chunks=[];client=None;times=[]
 try:
  client=ACTClient(ROOT/'.venv-act/bin/python',ROOT/'scripts/act_chunk_worker.py',ROOT/'outputs/learned_source_intake/act_checkpoint',root/'ipc')
  for ti,(task,source) in enumerate(sources.items()):
   data=np.load(source);local=BCLocal(data['obs'],data['action'],int(data['k']));ensemble=[BCLocal(data['obs'],data['action'],k) for k in (1,3,5)]
   for ep in range(a.episodes):
    seed=a.seed+ti*100000+ep;env=ContactEnv(task,seed);camera=ACTCamera(env);history=[]
    try:
     for q in a.queries:
      while env.t<q:history.append(env.step(local.action(env.observe())))
      request=env.observe();request_action=local.action(request);snapshot=env.snapshot();base_history=list(history)
      continuous=None
      if a.smoke:
       continuous=branch(env,local,ensemble,None,'reject',request_action,request_action,0,base_history);env.restore(snapshot)
      image=camera.image(env);prediction,timing=client.predict(request[:4].astype('float32'),image);chunk=source_action_to_native(prediction[0]);chunks.append(chunk);times.append(timing['seconds']);digest=hashlib.sha256(chunk.tobytes()).hexdigest()
      for delay in a.delays:
       env.restore(snapshot);past=list(base_history)
       for _ in range(delay):past.append(env.step(local.action(env.observe())))
       arrival=env.observe();arrival_action=local.action(arrival);at=env.snapshot();outcomes=[]
       for option in OPTIONS:
        env.restore(at);outcomes.append(branch(env,local,ensemble,chunk,option,request_action,arrival_action,delay,past))
       if delay==0:
        np.testing.assert_allclose(outcomes[5:7],outcomes[7:9],atol=1e-10,rtol=0)
       if a.smoke:
        np.testing.assert_array_equal(outcomes[0],continuous)
        env.restore(at);replay=branch(env,local,ensemble,chunk,'reject',request_action,arrival_action,delay,past);np.testing.assert_allclose(replay,outcomes[0],atol=1e-10,rtol=0)
       rows.append(dict(task=task,seed=seed,query=q,delay=delay,chunk_sha256=digest,chunk_index=len(chunks)-1));xs.append(features(request,arrival,chunk,request_action,arrival_action,q,delay));ys.append(outcomes)
      env.restore(snapshot);history=list(base_history)
     save(root/'rows.json',rows);np.savez_compressed(root/'labels.npz',x=xs,y=ys,chunks=chunks)
     y=np.asarray(ys);print('EPISODE',task,seed,'rows',len(rows),'act_oracle_return_gain',float(np.mean(y[:,-5:,2].max(1)-y[:,0,2])),'cheap_oracle_return_gain',float(np.mean(y[:,:5,2].max(1)-y[:,0,2])),flush=True)
    finally:camera.close();env.close()
  client.close();client=None;yy=np.asarray(ys);analysis={}
  for key in {(r['task'],r['seed']) for r in rows}:
   indices=[i for i,r in enumerate(rows) if (r['task'],r['seed'])==key]
   np.testing.assert_array_equal(yy[indices,0],np.repeat(yy[indices[:1],0],len(indices),axis=0))
  for key in {(r['task'],r['seed'],r['query']) for r in rows}:
   assert len({r['chunk_sha256'] for r in rows if (r['task'],r['seed'],r['query'])==key})==1
  for task in sources:
   mask=np.array([r['task']==task for r in rows]);v=yy[mask];analysis[task]=dict(episodes=a.episodes,rows=int(mask.sum()),method_mean={name:v[:,i].mean(0).tolist() for i,name in enumerate(OPTIONS)},source_positive_opportunity_mean=np.maximum(v[:,5:,2].max(1)-v[:,:5,2].max(1),0).mean(),source_signed_oracle_gap_mean=(v[:,5:,2].max(1)-v[:,:5,2].max(1)).mean())
  save(root/'analysis.json',dict(tasks=analysis,infer_seconds_median=float(np.median(times)),limitations='Development oracle opportunity; no learned gate, no deployed improvement claim'))
  assert code==fingerprint();save(root/'checks.json',dict(checks='passed',fixed_content_across_delay=True,zero_delay_prefix_suffix_identical=True,reject_replay_checked=a.smoke,all_options_full_native500=True,reject_identical_across_cuts=True,snapshot_version=3));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-t0,exit_code=0)
 except BaseException as e:
  state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state)
  if client is not None:client.close()
  raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
