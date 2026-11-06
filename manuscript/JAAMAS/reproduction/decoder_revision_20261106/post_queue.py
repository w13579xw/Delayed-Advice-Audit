import time,json,subprocess
from pathlib import Path
R=Path('/home/ez/DACI/code');D=R/'outputs/jaamas/decoder_revision_20261106';B=R/'outputs/jaamas/decoder_control_20261106_v1'
while not (D/'queue_complete.json').exists():
 for mode in ['identity','main']:
  f=B/mode/'run.json'
  if f.exists():assert json.loads(f.read_text())['status']!='failed',f
 time.sleep(10)
for stem in ['analyze_decoder','source_replay_audit']:
 with (D/(stem+'.log')).open('x') as log:
  code=subprocess.call([str(R/'.venv-metaworld/bin/python'),'-u',str(D/(stem+'.py'))],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 assert code==0,(stem,code)
 print(stem,'complete',flush=True)
(D/'post_queue_complete.json').write_text(json.dumps(dict(status='complete')))
