import json,hashlib,os,sys,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/home/ez/DelayedAdviceAudit/code');sys.path.insert(0,str(ROOT/'scripts'));from server_run import save,now,fingerprint
OUT=ROOT/'outputs/jaamas/visual_full_training_20261105_v1';OUT.mkdir(parents=True,exist_ok=False);(OUT/'logs').mkdir();fp=fingerprint();start=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'),CUBLAS_WORKSPACE_CONFIG=':4096:8');data=ROOT/'outputs/vision/features_200x2_20261103_v1';commands=[]
save(OUT/'protocol.json',dict(utc=now(),source_fingerprint=fp,training_seeds=[1,2],source_updates=15000,head_targets=['action','pose'],epochs=20,data=str(data),manifest_sha256=hashlib.sha256((data/'episodes.json').read_bytes()).hexdigest(),no_test_access=True));shutil.copy2(__file__,OUT/'entry_snapshot.py')
def run(name,args):
 save(OUT/'run.json',dict(status='running',stage=name,utc=now()));cmd=[str(ROOT/'.venv-vision/bin/python'),'-u',*args];t=time.monotonic();print('START',name,flush=True)
 with (OUT/'logs'/f'{name}.log').open('w') as f:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(name=name,command=cmd,exit_code=p.returncode,seconds=time.monotonic()-t));save(OUT/'commands.json',commands);assert p.returncode==0,name;assert fingerprint()==fp;print('DONE',name,commands[-1]['seconds'],flush=True)
try:
 run('smoke',['scripts/train_matched_act.py','--data',str(data),'--out',str(OUT/'smoke'),'--task','push-v3','--updates','2','--seed','1','--smoke'])
 for task,tag in [('push-v3','push'),('pick-place-v3','pick')]:
  for seed in [1,2]:
   run(f'{tag}_heads_seed{seed}',['scripts/train_visual_heads.py','--data',str(data),'--out',str(OUT/f'{tag}_heads_seed{seed}'),'--task',task,'--seed',str(seed),'--epochs','20','--targets','action','pose'])
   run(f'{tag}_act15k_seed{seed}',['scripts/train_matched_act.py','--data',str(data),'--out',str(OUT/f'{tag}_act15k_seed{seed}'),'--task',task,'--seed',str(seed),'--updates','15000'])
 save(OUT/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,completed_utc=now()));save(OUT/'checks.json',dict(checks='passed',act15k_models=4,full_head_runs=4,no_test_access=True));print('COMPLETE',flush=True)
except BaseException as ex:save(OUT/'run.json',dict(status='failed',exit_code=1,error=str(ex),ended_utc=now()));raise
