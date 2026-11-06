"""Fixed existing policy, native stale/refresh/erasure channel interventions.
No fitting, parameter replacement, or outcome-based choice is performed here.
"""
import argparse,json,hashlib,sys,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/home/ez/DACI/code');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'outputs/jaamas')]
from server_run import save,now,fingerprint
from state_core_20261105 import *

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def candidate_key(name,packet):return name+('/'+hashlib.sha256(np.asarray(packet).tobytes()).hexdigest() if name.startswith('expert') or name=='blend8' else '/local')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);begin=time.monotonic();fp=fingerprint();qual=ROOT/'outputs/jaamas/state_qualification_20261105_v1';models=ROOT/'outputs/jaamas/state_campaign_20261105_v1/models';rng=np.random.default_rng(2026110551);schedule=[]
 for ti,task in enumerate(TASKS):
  cell=0
  for d in [0,16,64]:
   for sigma in [0.,.005]:
    for i in range(10):schedule.append(dict(task=task,cell=cell,index=i,seed=850000000+ti*10000+cell*100+i,q=int(rng.choice([16,144,272,400])),delay=d,sigma=sigma))
    cell+=1
 if a.smoke:schedule=[dict(task=task,cell=j,index=0,seed=849000000+ti*10000+j,q=16,delay=d,sigma=.005) for ti,task in enumerate(TASKS) for j,d in enumerate([0,64])]
 files=[*models.rglob('model.npz'),*models.rglob('config.json'),*qual.glob('*/model.npz'),*qual.glob('*/config.json')];frozen={str(f):digest(f) for f in files};save(out/'protocol.json',dict(frozen_utc=now(),schedule=schedule,channels=dict(S='sameexpert noisyrequestobservation',F='sameexpert noisyarrivalobservation',E='absentsymbol, declared fixedwrapper->reject'),conditional_on='commonprefix/arrivalsnapshot/noisestream/native500;no resourcecost',scope='definedrefresh/availabilitychanneleffect,notprivateinformation;freshjointinputshiftacknowledged',smoke_only=a.smoke,model_hashes=frozen,source_fingerprint=fp));shutil.copy2(__file__,out/'entry_snapshot.py');save(out/'run.json',dict(status='running',started_utc=now()));rows=[];native_steps=0
 try:
  for task in TASKS:
   local=Locals(qual/task);teacher=Teacher(task);critics={f'seed{s}/{inp}/{act}':Critic(models/task/f'seed{s}'/inp/act) for s in range(5) for inp in ['local','packet'] for act in ['cheap','full']}
   for case in [c for c in schedule if c['task']==task]:
    folder=out/task/f"cell{case['cell']:02d}_ep{case['index']:02d}";folder.mkdir(parents=True);env=ContactEnv(task,case['seed']);noise=noise_stream(case['seed']+1000000,case['sigma']);rewards=[];actions0=[];states=[];lasts=[];q=case['q'];d=case['delay'];arrival=q+d
    try:
     request=None;stale=None;request_action=None
     while env.t<arrival:
      t=env.t;ob=sensor(env.observe(),noise[t]);act=local.action(ob)
      if t==q:request=visible(ob,local);stale=teacher.action(ob);request_action=act.copy()
      states.append(env.observe()[OBS_IDS]);actions0.append(act.copy());rewards.append(env.step(act));lasts.append(int(env.last_success));native_steps+=1
     ob=sensor(env.observe(),noise[env.t]);arr=visible(ob,local);arrival_action=local.action(ob);fresh=teacher.action(ob)
     if d==0:request=visible(ob,local);stale=teacher.action(ob);request_action=arrival_action.copy();np.testing.assert_array_equal(stale,fresh)
     snap=env.snapshot();packets={'S':stale,'F':fresh};cf={c:feature_groups(request,arr,q,d,case['sigma'],v) for c,v in packets.items()};decisions={}
     for label,critic in critics.items():
      vals={}
      for ch in ['S','F']:
       x=cf[ch][1 if critic.config['input']=='packet_features' else 0];vals[ch]=dict(choice=critic.decide(x),scores=critic.scores(x).tolist(),key=candidate_key(critic.decide(x),packets[ch]))
      vals['E']=dict(choice='reject',scores=None,key=candidate_key('reject',stale));decisions[label]=vals
     record=dict(**case,arrival=arrival,created_utc=now(),written_before_future=True,request={k:v.tolist() for k,v in request.items()},arrival_observation={k:v.tolist() for k,v in arr.items()},packets={k:v.tolist() for k,v in packets.items()},request_action=request_action.tolist(),arrival_action=arrival_action.tolist(),names=local.names,decisions=decisions,prefix_success=bool(env.seen_success),source_fresh_joint_residual=float(np.linalg.norm(fresh-stale)))
     choicepath=folder/'choices_before_outcomes.json';save(choicepath,record);choicehash=digest(choicepath);np.savez_compressed(folder/'observed_inputs.npz',S=cf['S'][1],F=cf['F'][1],cheap=cf['S'][0],noise=noise)
     plans={}
     for ch,packet in packets.items():
      for name in local.names:plans.setdefault(candidate_key(name,packet),dict(name=name,packet=packet,channels=[]))['channels'].append(ch)
     keys=list(plans);np.random.default_rng(case['seed']+37).shuffle(keys);branches={}
     for i,key in enumerate(keys):
      plan=plans[key];env.restore(snap);rr=list(rewards);aa=list(actions0);ss=list(states);ll=list(lasts)
      while env.t<500:
       t=env.t;noisy=sensor(env.observe(),noise[t]);act=plan_action(local,noisy,plan['name'],t-arrival,plan['packet'],request_action,arrival_action);ss.append(env.observe()[OBS_IDS]);aa.append(act.copy());rr.append(env.step(act));ll.append(int(env.last_success));native_steps+=1
      file=f'branch{i:02d}.npz';np.savez_compressed(folder/file,reward=np.asarray(rr),action=np.asarray(aa),state10=np.asarray(ss),last_success=np.asarray(ll));branches[key]=dict(file=file,name=plan['name'],channels=plan['channels'],**metrics(rr,env))
     # Always verify the same native reject suffix independently, no extra choice.
     env.restore(snap);basekey=candidate_key('reject',stale);z=np.load(folder/branches[basekey]['file'])
     while env.t<500:
      t=env.t;np.testing.assert_array_equal(env.observe()[OBS_IDS],z['state10'][t]);act=local.action(sensor(env.observe(),noise[t]));np.testing.assert_array_equal(act,z['action'][t]);assert env.step(act)==z['reward'][t];assert int(env.last_success)==z['last_success'][t];native_steps+=1
     assert digest(choicepath)==choicehash
     outcomes={lab:{ch:dict(branches[v['key']]) for ch,v in dec.items()} for lab,dec in decisions.items()};save(folder/'case.json',dict(**case,arrival=arrival,branches=branches,outcomes=outcomes,choices_sha256=choicehash,unique_branches=len(branches)));rows.append(dict(**case,folder=str(folder),unique_branches=len(branches)));save(out/'rows.json',rows);print('CHANNEL',len(rows),len(schedule),task,case['cell'],case['index'],'plans',len(plans),'seconds',round(time.monotonic()-begin,1),flush=True)
    finally:env.close()
  assert fingerprint()==fp and all(digest(Path(f))==h for f,h in frozen.items());save(out/'checks.json',dict(checks='passed',cases=len(rows),native_steps=native_steps,frozen_models_unchanged=True,choices_before_outcomes=True,reject_tail_bitwise_replayed_every_case=True,delay0_identical_packets=True,smoke_only=a.smoke));save(out/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-begin,completed_utc=now()))
 except BaseException as ex:save(out/'run.json',dict(status='failed',exit_code=1,error=str(ex),ended_utc=now()));raise
if __name__=='__main__':main()
