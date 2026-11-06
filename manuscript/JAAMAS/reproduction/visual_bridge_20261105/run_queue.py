import json,hashlib,subprocess,time
from pathlib import Path
R=Path('/home/ez/DACI/code');D=R/'outputs/jaamas/bridge_revision_20261105';B=R/'outputs/jaamas/visual_channel_bridge_20261105_v1'
p=json.loads((B/'smoke/protocol.json').read_text());assert json.loads((B/'smoke/run.json').read_text())['status']=='complete'
def check():
 assert hashlib.sha256((D/'visual_channel.py').read_bytes()).hexdigest()==p['entry_sha256']
 for f,h in p['model_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
check();(D/'queue_freeze.json').write_text(json.dumps(dict(entry=p['entry_sha256'],model_hashes=p['model_hashes'],modes=['identity','main'],smoke_seconds=json.loads((B/'smoke/run.json').read_text())['seconds']),indent=2))
for mode in ['identity','main']:
 check()
 with (D/f'{mode}.log').open('x') as log:
  rc=subprocess.call([str(R/'.venv-metaworld/bin/python'),'-u',str(D/'visual_channel.py'),'--out',str(B/mode),'--mode',mode],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 assert rc==0,(mode,rc)
 check();print(mode,'complete',flush=True)
(D/'queue_complete.json').write_text(json.dumps(dict(status='complete',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
