"""Complete three-seed system reproduction; waits for isolated GPU training."""
import json,time,subprocess,os,sys,hashlib,shutil
from pathlib import Path
R=Path('/home/ez/DACI/code');D=R/'outputs/jaamas/revision_20261105';sys.path[:0]=[str(R/'scripts'),str(D)];from server_run import save,now,fingerprint
from replication_support import source_path
O=R/'outputs/jaamas/visual_pipeline_20261105_v1';O.mkdir(exist_ok=False);(O/'logs').mkdir();(O/'entry_snapshot').mkdir();fp=fingerprint();start=time.monotonic();commands=[];schedule=D/'visual_schedule.json';tracked=[D/x for x in ['replication_support.py','collect_visual_repl.py','test_visual_repl.py','visual_schedule.json','run_visual_pipeline.py']]+[R/'outputs/vision/fit_critic_curve_v2_20261104.py']
for p in tracked:shutil.copy2(p,O/'entry_snapshot'/p.name)
save(O/'entry_hashes.json',dict(utc=now(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},source_fingerprint=fp));baseenv=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONPATH=str(R/'src'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
def run(name,cmd,seed=0):
 assert fingerprint()==fp;save(O/'run.json',dict(status='running',stage=name,utc=now()));env=dict(baseenv,DACI_REPLICATION_ACTOR_SEED=str(seed));t=time.monotonic();print('START',name,flush=True)
 with (O/'logs'/f'{name}.log').open('w') as f:p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(stage=name,seed=seed,command=cmd,exit_code=p.returncode,seconds=time.monotonic()-t));save(O/'commands.json',commands);assert p.returncode==0,name;print('DONE',name,commands[-1]['seconds'],flush=True)
try:
 save(O/'run.json',dict(status='running',stage='waiting_for_training',utc=now()))
 while True:
  status=json.loads((R/'outputs/jaamas/visual_full_training_20261105_v1/run.json').read_text())
  if status['status']=='failed':raise RuntimeError('Training failed, preserve outputs and do not open tests')
  if status['status']=='complete':break
  time.sleep(30)
 # Seed0 accepted development/critic is retained. Allnewseeds use originaldev schedule.
 roots={0:R/'outputs/vision/value_kernel_pilot_20261104_v3'}
 for seed in [1,2]:
  data=O/f'seed{seed}_development';smoke=O/f'seed{seed}_collection_smoke'
  run(f'collect_smoke_seed{seed}',[str(R/'.venv-metaworld/bin/python'),'-u',str(D/'collect_visual_repl.py'),'--out',str(smoke),'--schedule',str(schedule),'--smoke'],seed)
  run(f'collect_seed{seed}',[str(R/'.venv-metaworld/bin/python'),'-u',str(D/'collect_visual_repl.py'),'--out',str(data),'--schedule',str(schedule)],seed)
  roots[seed]=O/f'seed{seed}_models'
  run(f'fit_seed{seed}',[str(R/'.venv-metaworld/bin/python'),'-u',str(R/'outputs/vision/fit_critic_curve_v2_20261104.py'),'--data',str(data),'--out',str(roots[seed]),'--train-count','14','--validation-count','6'],seed)
 # Freeze allactors/models and executable entries before the first test, includingsmoke.
 paths=[]
 for seed in [0,1,2]:
  paths.extend(p for p in roots[seed].iterdir() if p.suffix in ['.json','.npz'])
  for task in ['push-v3','pick-place-v3']:
   tag='push' if task=='push-v3' else 'pick';h=R/'outputs/vision'/f'head_training_{tag}_20261103_v1' if seed==0 else R/'outputs/jaamas/visual_full_training_20261105_v1'/f'{tag}_heads_seed{seed}';paths.extend(h/name/f for name in ['action_c16_h128','action_c32_h256','pose_c16_h128','pose_c32_h256'] for f in ['config.json','head.safetensors']);paths.extend(source_path(task,seed)/f for f in ['metadata.json','core.safetensors','source_config.json','normalization.json'])
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};save(O/'all_models_frozen_before_test.json',dict(utc=now(),files=hashes,model_roots={str(k):str(v) for k,v in roots.items()}))
 for seed in [0,1,2]:
  run(f'test_smoke_seed{seed}',[str(R/'.venv-metaworld/bin/python'),'-u',str(D/'test_visual_repl.py'),'--out',str(O/f'seed{seed}_test_smoke'),'--schedule',str(schedule),'--models',str(roots[seed]),'--smoke'],seed)
 # Test order fixed before outcomes, all actorseeds retained irrespective competence.
 for seed in [0,1,2]:
  run(f'test_seed{seed}',[str(R/'.venv-metaworld/bin/python'),'-u',str(D/'test_visual_repl.py'),'--out',str(O/f'seed{seed}_test'),'--schedule',str(schedule),'--models',str(roots[seed])],seed)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());assert fingerprint()==fp;save(O/'checks.json',dict(checks='passed',training_seeds=[0,1,2],independent_environment_seeds=100,logical_arrival_cases=300,full_source_competence_trajectories=300,all_models_unchanged=True));save(O/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,utc=now()));print('COMPLETE',flush=True)
except BaseException as ex:save(O/'run.json',dict(status='failed',exit_code=1,error=str(ex),utc=now()));raise
