import sys,os,time,json,subprocess,hashlib,shutil
from pathlib import Path
R=Path('/home/ez/DACI/code');sys.path.insert(0,str(R/'scripts'));from server_run import save,now
O=R/'outputs/jaamas/channel_revision_20261105_v1';O.mkdir(exist_ok=False);(O/'logs').mkdir();env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONPATH=str(R/'src'));start=time.monotonic();commands=[];source=R/'outputs/jaamas/revision_20261105/channel_native.py';shutil.copy2(source,O/'entry_snapshot.py');save(O/'entry_hash.json',dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),utc=now()))
try:
 for name,extra in [('smoke',['--smoke']),('main',[])]:
  save(O/'run.json',dict(status='running',stage=name,utc=now()));cmd=[str(R/'.venv-metaworld/bin/python'),'-u',str(source),'--out',str(O/name),*extra];t=time.monotonic();print('START',name,flush=True)
  with (O/'logs'/f'{name}.log').open('w') as f:p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
  commands.append(dict(stage=name,exit_code=p.returncode,command=cmd,seconds=time.monotonic()-t));save(O/'commands.json',commands);assert p.returncode==0,name;print('DONE',name,commands[-1]['seconds'],flush=True)
 save(O/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,utc=now()));print('COMPLETE',flush=True)
except BaseException as e:save(O/'run.json',dict(status='failed',exit_code=1,error=str(e),utc=now()));raise
