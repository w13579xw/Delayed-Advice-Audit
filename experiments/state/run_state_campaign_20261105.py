"""Sequential, logged state campaign; unique outputs, failure preservation."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts')]
from server_run import save,now,fingerprint
out=ROOT/'outputs/jaamas/state_campaign_20261105_v1';out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();snap=out/'entry_snapshot';snap.mkdir();scripts=['state_core_20261105.py','qualify_state_tasks_20261105.py','collect_state_data_20261105.py','fit_state_critics_20261105.py','test_state_critics_20261105.py','analyze_audit_state_20261105.py','run_state_campaign_20261105.py'];hs={}
for name in scripts:
 p=ROOT/'experiments/state'/name;shutil.copy2(p,snap/name);hs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copy2(ROOT/'docs/learned_advice/jaamas_campaign_20261105/PLAN.md',snap/'PLAN.md');save(out/'frozen_entries.json',dict(utc=now(),sha256=hs,source_fingerprint=fingerprint()));start=time.monotonic();commands=[];env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH='src',MUJOCO_GL='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8');qual=ROOT/'outputs/jaamas/state_qualification_20261105_v1'
assert json.loads((qual/'run.json').read_text())['status']=='complete'
assert all(x['qualified'] for x in json.loads((qual/'analysis.json').read_text()).values())
steps=[('development_smoke','.venv-metaworld',['collect_state_data_20261105.py','--out',str(out/'development_smoke'),'--qualification',str(qual),'--smoke']),('development','.venv-metaworld',['collect_state_data_20261105.py','--out',str(out/'development'),'--qualification',str(qual)]),('fit','.venv-vision',['fit_state_critics_20261105.py','--data',str(out/'development'),'--out',str(out/'models')]),('boundary_smoke','.venv-metaworld',['test_state_critics_20261105.py','--out',str(out/'boundary_smoke'),'--qualification',str(qual),'--models',str(out/'models'),'--smoke']),('test','.venv-metaworld',['test_state_critics_20261105.py','--out',str(out/'test'),'--qualification',str(qual),'--models',str(out/'models')]),('analysis_audit','.venv-metaworld',['analyze_audit_state_20261105.py','--test',str(out/'test'),'--data',str(out/'development'),'--models',str(out/'models'),'--out',str(out/'analysis')])]
try:
 for name,venv,args in steps:
  save(out/'run.json',dict(status='running',stage=name,started_utc=now()));cmd=[str(ROOT/venv/'bin/python'),'-u',str(ROOT/'experiments/state'/args[0]),*args[1:]];s=time.monotonic();print('START',name,flush=True)
  with (out/'logs'/f'{name}.log').open('w') as log:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  commands.append(dict(stage=name,command=cmd,exit_code=p.returncode,seconds=time.monotonic()-s));save(out/'commands.json',commands)
  if p.returncode:raise RuntimeError(f'{name} failed exit{p.returncode}; see preserved log')
  print('DONE',name,commands[-1]['seconds'],flush=True)
 save(out/'checks.json',dict(checks='passed',stages=len(commands)));save(out/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,completed_utc=now()));print('COMPLETE',flush=True)
except BaseException as ex:save(out/'run.json',dict(status='failed',exit_code=1,error=str(ex),ended_utc=now()));raise
