"""Development-only public-policy reference on every native MT50 task."""
import argparse,json,time,importlib.metadata,hashlib
from pathlib import Path
from metaworld.policies import ENV_POLICY_MAP
from advice_audit.metaworld_adapter import expert_action
from strong_local_baseline import evaluate,summary
from server_run import ROOT,fingerprint,now,save

class Public:
 def __init__(self,cls):self.policy=cls()
 def action(self,obs):return expert_action(self.policy,obs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=2);p.add_argument('--seed',type=int,default=50000000);a=p.parse_args();root=Path(a.out).resolve()
 if root.exists():p.error('new directory required')
 root.mkdir(parents=True);tasks=sorted(ENV_POLICY_MAP);assert len(tasks)==50;code=fingerprint();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 save(root/'protocol.json',dict(config=vars(a),tasks=tasks,offset=100000,tier='auxiliary/development',frozen_utc=now(),horizon=500,public_policies='metaworld.policies.ENV_POLICY_MAP',native_state_feedback=True))
 for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  d=root/'source_snapshot'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
 rows={};t0=time.monotonic()
 try:
  for i,task in enumerate(tasks):
   seeds=range(a.seed+i*100000,a.seed+i*100000+a.episodes);rows[task]=evaluate(task,Public(ENV_POLICY_MAP[task]),seeds);save(root/'episodes.json',rows);print(task,summary(rows[task]),flush=True)
  save(root/'analysis.json',{t:summary(r) for t,r in rows.items()});assert code==fingerprint();save(root/'checks.json',dict(checks='passed',tasks=50,episodes=50*a.episodes,no_test_claim=True,all_failures_retained=True));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-t0,exit_code=0)
 except BaseException as e:state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state);raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
