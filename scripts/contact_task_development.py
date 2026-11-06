"""Development-only native-task screen with all200 demos and strong BC grid."""
import argparse,importlib,json,time
from pathlib import Path
import numpy as np
from advice_audit.metaworld_adapter import ContactEnv,BCLocal,expert_action
from strong_local_baseline import evaluate,summary
from server_run import ROOT,fingerprint,save,now

SPECS=[('assembly-v3','sawyer_assembly_v3_policy','SawyerAssemblyV3Policy'),
       ('pick-place-v3','sawyer_pick_place_v3_policy','SawyerPickPlaceV3Policy'),
       ('peg-insert-side-v3','sawyer_peg_insertion_side_v3_policy','SawyerPegInsertionSideV3Policy'),
       ('door-open-v3','sawyer_door_open_v3_policy','SawyerDoorOpenV3Policy')]
class Public:
 def __init__(self,module,name):self.policy=getattr(importlib.import_module('metaworld.policies.'+module),name)()
 def action(self,obs):return expert_action(self.policy,obs)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out)
 if root.exists():p.error('new directory required')
 root.mkdir(parents=True);code=fingerprint();t0=time.monotonic()
 save(root/'protocol.json',dict(frozen_utc=now(),tier='auxiliary/development',tasks=SPECS,demos=200,development=12,horizon=500,k=[1,3,5],ladder=[40,100,200],test_episodes=0,primary='native any-success',no_failure_filtering=True))
 for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  d=root/'source_snapshot'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
 state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state);all_results={}
 try:
  for idx,(task,module,name) in enumerate(SPECS):
   policy=Public(module,name);obs=[];actions=[];episodes=[];demo_rows=[];start=6000000+idx*100000
   for ep in range(200):
    env=ContactEnv(task,start+ep);rewards=[]
    try:
     while env.t<500:
      ob=env.observe();act=policy.action(ob);obs.append(ob);actions.append(act);episodes.append(ep);rewards.append(env.step(act))
     demo_rows.append(dict(seed=start+ep,success=int(env.seen_success),terminal_success=int(env.last_success),native_return=float(np.mean(rewards))))
    finally:env.close()
   obs,actions,episodes=map(np.asarray,(obs,actions,episodes));np.savez_compressed(root/(task+'_demos.npz'),obs=obs,action=actions,episode=episodes);save(root/(task+'_demo_outcomes.json'),demo_rows)
   print(task,'DEMOS',summary(demo_rows),flush=True);scores=[]
   seeds=range(start+50000,start+50012)
   for count in (40,100,200):
    for k in (1,3,5):
     mask=episodes<count;model=BCLocal(obs[mask],actions[mask],k);rows=evaluate(task,model,seeds);entry=dict(demos=count,k=k,metrics=summary(rows),rows=rows);scores.append(entry);save(root/(task+'_development.json'),scores);print(task,'DEV',count,k,entry['metrics'],flush=True)
   public=evaluate(task,policy,seeds);save(root/(task+'_public.json'),public)
   best=max(scores,key=lambda r:(r['metrics']['success'],r['metrics']['terminal_success'],r['metrics']['native_return'],-r['k'],r['demos']))
   mask=episodes<best['demos'];np.savez_compressed(root/(task+'_local.npz'),obs=obs[mask],action=actions[mask],episode=episodes[mask],k=best['k'])
   all_results[task]=dict(selected={k:best[k] for k in ('demos','k','metrics')},public=summary(public),headroom=summary(public)['success']-best['metrics']['success']);save(root/'analysis.json',all_results)
  assert code==fingerprint();save(root/'checks.json',dict(checks='passed',development_only=True,all_tasks_retained=True));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-t0,exit_code=0)
 except BaseException as e:state.update(status='failed',error=str(e));save(root/'run.json',state);raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
