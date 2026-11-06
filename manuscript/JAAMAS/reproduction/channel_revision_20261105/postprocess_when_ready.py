"""Queue read-only audit/analysis after all previously authorized native runs."""
import json,time,subprocess,sys,os,hashlib,shutil
from pathlib import Path
R=Path('/home/ez/DACI/code');D=R/'outputs/jaamas/revision_20261105';O=R/'outputs/jaamas/visual_revision_postprocess_20261105_v1';O.mkdir(exist_ok=False);(O/'logs').mkdir();(O/'entry_snapshot').mkdir();sys.path.insert(0,str(R/'scripts'));from server_run import save,now
files=[D/x for x in ['postprocess_when_ready.py','analyze_visual_replicates.py','audit_visual_development.py']]
for p in files:shutil.copy2(p,O/'entry_snapshot'/p.name)
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};save(O/'entries.json',dict(files=hashes,utc=now()));env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1');commands=[]
def run(name,args):
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());save(O/'run.json',dict(status='running',stage=name,utc=now()));start=time.monotonic();cmd=[str(R/'.venv-metaworld/bin/python'),*args]
 with (O/'logs'/f'{name}.log').open('w') as f:p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(stage=name,command=cmd,exit_code=p.returncode,seconds=time.monotonic()-start));save(O/'commands.json',commands);assert p.returncode==0,name
try:
 save(O/'run.json',dict(status='running',stage='waiting_for_native_pipeline',utc=now()))
 P=R/'outputs/jaamas/visual_pipeline_20261105_v1'
 while True:
  status=json.loads((P/'run.json').read_text())
  if status['status']=='failed':raise RuntimeError('native_pipeline failed; preserve all outputs')
  if status['status']=='complete':break
  time.sleep(30)
 for seed in [1,2]:run(f'development_seed{seed}',[str(D/'audit_visual_development.py'),'--data',str(P/f'seed{seed}_development'),'--models',str(P/f'seed{seed}_models'),'--out',str(O/f'seed{seed}_development_audit')])
 run('test_audit_analysis',[str(D/'analyze_visual_replicates.py')])
 save(O/'checks.json',dict(checks='passed',all_prespecified_analyses_retained=True));save(O/'run.json',dict(status='complete',exit_code=0,utc=now()));print('COMPLETE',O,flush=True)
except BaseException as ex:save(O/'run.json',dict(status='failed',exit_code=1,error=str(ex),utc=now()));raise
