"""Matched one-arrival labels on independently frozen state-cohort episodes."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'outputs/jaamas')]
from server_run import save,now,fingerprint
from state_core_20261105 import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--qualification',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();fp=fingerprint();rng=np.random.default_rng(2026110521);qual=Path(a.qualification).resolve();qa=json.loads((qual/'analysis.json').read_text());schedule=[]
 for ti,task in enumerate(TASKS):
  for i in range(42):schedule.append(dict(task=task,index=i,seed=820000000+ti*10000+i,q=int(rng.choice([16,144,272,400])),delay=int(rng.choice([0,16,64])),sigma=float(rng.choice([0,.005])),split='train' if i<30 else 'validation'))
 if a.smoke:schedule=[dict(task=t,index=0,seed=840000000+ti*10000,q=16,delay=64,sigma=.005,split='engineering') for ti,t in enumerate(TASKS)]
 save(out/'schedule.json',schedule);save(out/'run.json',dict(status='running',created_utc=now(),source_fingerprint=fp,smoke=a.smoke));rows=[];steps=0
 try:
  for case in schedule:
   task=case['task'];assert qa[task]['qualified'],f'Unqualified task retained in qualification report: {task}; stop before interpreting advice efficacy'
   local=Locals(qual/task);teacher=Teacher(task);e=ContactEnv(task,case['seed']);noise=noise_stream(case['seed']+1000000,case['sigma']);rewards=[];actions=[];q=case['q'];d=case['delay']
   try:
    while e.t<q:
     ob=sensor(e.observe(),noise[e.t]);act=local.action(ob);actions.append(act);rewards.append(e.step(act));steps+=1
    request_ob=sensor(e.observe(),noise[e.t]);request=visible(request_ob,local);request_action=local.action(request_ob);packet=teacher.action(request_ob)
    while e.t<q+d:
     ob=sensor(e.observe(),noise[e.t]);act=local.action(ob);actions.append(act);rewards.append(e.step(act));steps+=1
    arrival_ob=sensor(e.observe(),noise[e.t]);arrival=visible(arrival_ob,local);arrival_action=local.action(arrival_ob);snap=e.snapshot();prefix=np.array(rewards);prefix_actions=np.array(actions);rr=[];aa=[];ss=[];oo=[]
    for name in local.names:
     e.restore(snap);tail=[];acts=[];obs=[]
     while e.t<500:
      obs.append(e.observe());noisy=sensor(e.observe(),noise[e.t]);act=plan_action(local,noisy,name,e.t-q-d,packet,request_action,arrival_action);acts.append(act);tail.append(e.step(act));steps+=1
     rr.append(np.r_[prefix,tail]);aa.append(np.vstack([prefix_actions,acts]));ss.append((int(e.seen_success),int(e.last_success)));oo.append(np.asarray(obs))
    e.restore(snap)
    for act,reward,ob in zip(aa[0][q+d:],rr[0][q+d:],oo[0]):
     np.testing.assert_array_equal(e.observe(),ob);np.testing.assert_array_equal(local.action(sensor(e.observe(),noise[e.t])),act);assert e.step(act)==reward;steps+=1
    cheap,packetfeatures=feature_groups(request,arrival,q,d,case['sigma'],packet);r=np.asarray(rr);target=np.mean(r,axis=1)-np.mean(r[0]);assert target[0]==0
    row={**case,'names':local.names,'packet':packet.tolist(),'request':{k:v.tolist() for k,v in request.items()},'arrival':{k:v.tolist() for k,v in arrival.items()},'reward_means':r.mean(axis=1).tolist(),'target':target.tolist(),'file':f"{task}_{case['index']:03d}.npz"}
    np.savez_compressed(out/row['file'],reward=r,action=np.asarray(aa),success=np.asarray(ss),cheap=cheap,packet_features=packetfeatures,target=target,noise=noise,prefix=prefix)
    rows.append(row);save(out/'rows.json',rows);print('LABELS',len(rows),len(schedule),task,flush=True)
   finally:e.close()
  assert fingerprint()==fp;save(out/'checks.json',dict(checks='passed',reject_replay='bitwise every development episode',native_steps=steps,whole_episode_groups=len(rows)));save(out/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,completed_utc=now()))
 except BaseException as ex:save(out/'run.json',dict(status='failed',exit_code=1,error=str(ex),ended_utc=now()));raise
if __name__=='__main__':main()
