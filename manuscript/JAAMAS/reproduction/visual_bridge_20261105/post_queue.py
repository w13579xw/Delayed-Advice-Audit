import time,json,subprocess,os
from pathlib import Path
R=Path('/home/ez/DACI/code');D=R/'outputs/jaamas/bridge_revision_20261105';B=R/'outputs/jaamas/visual_channel_bridge_20261105_v1'
while not (D/'queue_complete.json').exists():
 for mode in ['identity','main']:
  f=B/mode/'run.json'
  if f.exists():assert json.loads(f.read_text())['status']!='failed',f
 time.sleep(10)
for name in ['analyze_visual_channel','source_replay_audit']:
 with (D/(name+'.log')).open('x') as log:
  rc=subprocess.call([str(R/'.venv-metaworld/bin/python'),'-u',str(D/(name+'.py'))],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 assert rc==0,(name,rc)
 print(name,'complete',flush=True)
(D/'post_queue_complete.json').write_text(json.dumps(dict(status='complete')))
