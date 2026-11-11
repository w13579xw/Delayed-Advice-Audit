"""Paired closed-loop robustness with exact prefix-tree branch reuse.

A shared branch is reused only until its policies choose different actions.
Periodic requests then use each branch's own states and source packets.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'experiments/state')]
from server_run import save,now,fingerprint
from state_core_20261105 import *

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run_case(case,local,teacher,critics,folder,verify=False):
 e=ContactEnv(case['task'],case['seed']);noise=noise_stream(case['seed']+1000000,case['sigma']);qs=[16] if case['frequency']=='single' else [16,144,272,400];d=case['delay'];labels=list(critics)+['reject','fixed_expert32'];leaves={};steps=0;events=[];pathcounter=[0]
 def recurse(members,event_index,rewards,actions,decisions):
  nonlocal steps
  q=qs[event_index]
  while e.t<q:
   ob=sensor(e.observe(),noise[e.t]);act=local.action(ob);actions=actions+[act.copy()];rewards=rewards+[e.step(act)];steps+=1
  request_ob=sensor(e.observe(),noise[e.t]);request=visible(request_ob,local);request_action=local.action(request_ob);packet=teacher.action(request_ob)
  while e.t<q+d:
   ob=sensor(e.observe(),noise[e.t]);act=local.action(ob);actions=actions+[act.copy()];rewards=rewards+[e.step(act)];steps+=1
  arrival_ob=sensor(e.observe(),noise[e.t]);arrival=visible(arrival_ob,local);arrival_action=local.action(arrival_ob);cheap,pf=feature_groups(request,arrival,q,d,case['sigma'],packet);snap=e.snapshot();groups={};eventid=len(events)
  for label in members:
   if label in critics:
    cc=critics[label];choice=cc.decide(pf if cc.config['input']=='packet_features' else cheap)
   else:choice='reject' if label=='reject' else 'expert32'
   groups.setdefault(choice,[]).append(label)
  events.append(dict(id=eventid,q=q,arrival=q+d,packet=packet.tolist(),request={k:v.tolist() for k,v in request.items()},observation={k:v.tolist() for k,v in arrival.items()},cheap=cheap.tolist(),packet_features=pf.tolist(),groups=groups))
  for choice,group in groups.items():
   e.restore(snap);rr=list(rewards);aa=list(actions);dd=decisions+[dict(event=eventid,choice=choice)];limit=qs[event_index+1] if event_index+1<len(qs) else 500
   while e.t<limit:
    ob=sensor(e.observe(),noise[e.t]);act=plan_action(local,ob,choice,e.t-q-d,packet,request_action,arrival_action);aa.append(act.copy());rr.append(e.step(act));steps+=1
   if limit<500:recurse(group,event_index+1,rr,aa,dd)
   else:
    leaf=pathcounter[0];pathcounter[0]+=1
    np.savez_compressed(folder/f'leaf{leaf:03d}.npz',reward=np.array(rr),action=np.array(aa),noise=noise)
    result=dict(leaf=leaf,**metrics(rr,e),decisions=dd)
    for label in group:leaves[label]=result
 try:
  recurse(labels,0,[],[],[])
  if verify:
   # Independent sequential replay of every policy, not tree logic.
   for label,result in leaves.items():
    direct=ContactEnv(case['task'],case['seed']);rr=[];aa=[];active='reject';arrival_t=-1;request_action=None;arrival_action=None;packet=None;request=None;hist=[]
    try:
     while direct.t<500:
      t=direct.t;ob=sensor(direct.observe(),noise[t])
      if t in qs:request_ob=ob.copy();request=visible(ob,local);request_action=local.action(ob);packet=teacher.action(ob);q=t
      if t in [q0+d for q0 in qs]:
       arrival=visible(ob,local);arrival_action=local.action(ob);cheap,pf=feature_groups(request,arrival,q,d,case['sigma'],packet)
       active=critics[label].decide(pf if critics[label].config['input']=='packet_features' else cheap) if label in critics else ('reject' if label=='reject' else 'expert32');arrival_t=t;hist.append(active)
      act=local.action(ob) if arrival_t<0 else plan_action(local,ob,active,t-arrival_t,packet,request_action,arrival_action);aa.append(act);rr.append(direct.step(act));steps+=1
     z=np.load(folder/f"leaf{result['leaf']:03d}.npz");np.testing.assert_array_equal(z['reward'],rr);np.testing.assert_array_equal(z['action'],aa);assert hist==[x['choice'] for x in result['decisions']]
    finally:direct.close()
  save(folder/'case.json',dict(**case,results=leaves,events=events,native_steps=steps,unique_leaves=pathcounter[0],independent_sequential_replay=verify));return steps,pathcounter[0]
 finally:e.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--qualification',required=True);p.add_argument('--models',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);qual=Path(a.qualification).resolve();models=Path(a.models).resolve();schedule=[];start=time.monotonic();fp=fingerprint()
 for ti,task in enumerate(TASKS):
  cell=0
  for d in [0,16,64]:
   for f in ['single','periodic']:
    for sigma in [0.,.005]:
     for i in range(10):schedule.append(dict(task=task,cell=cell,index=i,seed=830000000+ti*10000+cell*100+i,delay=d,frequency=f,sigma=sigma))
     cell+=1
 if a.smoke:
  schedule=[dict(task=t,cell=j,index=0,seed=840000100+ti*10000+j,delay=d,frequency=f,sigma=s) for ti,t in enumerate(TASKS) for j,(d,f,s) in enumerate([(0,'single',0.),(64,'periodic',.005)])]
 save(out/'schedule.json',schedule);files=list(models.rglob('*.npz'))+list(models.rglob('config.json'))+list(qual.glob('*/model.npz'))+list(qual.glob('*/config.json'));frozen={str(p):sha(p) for p in files};save(out/'frozen_before_test.json',dict(utc=now(),files=frozen,source_fingerprint=fp));save(out/'run.json',dict(status='running',created_utc=now(),smoke=a.smoke));steps=0;tails=0
 try:
  for ti,task in enumerate(TASKS):
   local=Locals(qual/task);teacher=Teacher(task);critics={f'seed{s}/{inp}/{act}':Critic(models/task/f'seed{s}'/inp/act) for s in range(5) for inp in ['local','packet'] for act in ['cheap','full']}
   for case in [x for x in schedule if x['task']==task]:
    folder=out/task/f"cell{case['cell']:02d}_ep{case['index']:02d}";folder.mkdir(parents=True);n,l=run_case(case,local,teacher,critics,folder,verify=a.smoke);steps+=n;tails+=l;print('TEST',task,case['cell'],case['index'],'tails',l,flush=True)
  assert all(sha(Path(p))==h for p,h in frozen.items()) and fingerprint()==fp
  save(out/'checks.json',dict(checks='passed',cases=len(schedule),native_steps=steps,unique_trajectories=tails,all_model_hashes_unchanged=True,independent_sequential_replay_every_policy=a.smoke));save(out/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,completed_utc=now()))
 except BaseException as ex:save(out/'run.json',dict(status='failed',exit_code=1,error=str(ex),ended_utc=now()));raise
if __name__=='__main__':main()
