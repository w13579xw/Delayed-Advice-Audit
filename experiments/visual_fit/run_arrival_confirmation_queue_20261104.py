import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from server_run import save,now
root=ROOT/'outputs/vision/arrival_confirmation_queue_20261104_v1'
if root.exists():raise RuntimeError('New queue directory required')
root.mkdir();save(root/'run.json',dict(status='waiting',created_utc=now()));smoke=ROOT/'outputs/vision/arrival_confirmation_smoke_20261104_v2';commands=[]
def run(cmd,name):
 commands.append(cmd);save(root/'commands.json',commands)
 with (root/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
 while True:
  state=json.loads((smoke/'run.json').read_text())
  if state['status']=='complete':assert json.loads((smoke/'checks.json').read_text())['checks']=='passed';break
  if state['status']=='failed':raise RuntimeError('Boundarysmokefailed')
  time.sleep(5)
 save(root/'smoke_timing.json',dict(seconds=state['seconds'],episodes=4,estimated_main_seconds=state['seconds']*125,budget_seconds=10800))
 if state['seconds']*125>10800:raise RuntimeError('Runtimeprofileexceeds3hourbudget;revisebudgetbeforetest')
 save(root/'run.json',dict(status='running',started_utc=now()));run([str(ROOT/'.venv-metaworld/bin/python'),'scripts/confirm_visual_arrival.py','--out','outputs/vision/arrival_confirmation_20261104_v1'],'confirmation');run([str(ROOT/'.venv-metaworld/bin/python'),'experiments/visual_fit/analyze_arrival_confirmation_20261104.py'],'analysis');save(root/'checks.json',dict(checks='passed',confirmation_and_analysis_complete=True));save(root/'run.json',dict(status='complete',completed_utc=now(),exit_code=0));print('COMPLETE',root,flush=True)
except BaseException as e:save(root/'run.json',dict(status='failed',ended_utc=now(),error=str(e)));raise
