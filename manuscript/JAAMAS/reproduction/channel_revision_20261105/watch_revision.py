"""Read-only live progress monitor; never reads outcomes or changes protocols."""
import json,time,datetime
from pathlib import Path
R=Path('/home/ez/DelayedAdviceAudit/code/outputs/jaamas');last=None
while True:
 def read(p):
  try:return json.loads(p.read_text())
  except (FileNotFoundError,json.JSONDecodeError):return {}
 p=R/'visual_pipeline_20261105_v1';state=read(p/'run.json');post=read(R/'visual_revision_postprocess_20261105_v1/run.json');counts={}
 for seed in [1,2]:
  rows=read(p/f'seed{seed}_development/rows.json');counts[f'development{seed}']=len(rows) if isinstance(rows,list) else 0
 for seed in [0,1,2]:
  rows=read(p/f'seed{seed}_test/episodes.json');counts[f'test{seed}']=len(rows) if isinstance(rows,list) else 0
 snapshot=dict(stage=state.get('stage',state.get('status')),postprocess=post.get('stage',post.get('status')),**counts)
 if snapshot!=last:print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(snapshot),flush=True);last=snapshot
 if state.get('status')=='failed' or post.get('status')=='failed':print('FAILURE inspect versioned run logs',flush=True);break
 if post.get('status')=='complete':print('COMPLETE postprocessing; manuscript final integration still required',flush=True);break
 time.sleep(10)
