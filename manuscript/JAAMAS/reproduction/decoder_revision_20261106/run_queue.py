from pathlib import Path
import json,hashlib,subprocess
R=Path('/home/ez/DelayedAdviceAudit/code');D=R/'outputs/jaamas/decoder_revision_20261106';B=R/'outputs/jaamas/decoder_control_20261106_v1'
p=json.loads((B/'smoke/protocol.json').read_text());assert json.loads((B/'smoke/run.json').read_text())['status']=='complete'
def check():
 for f,h in {**p['model_hashes'],**p['entry_hashes']}.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
check();(D/'queue_freeze.json').write_text(json.dumps(dict(model_hashes=p['model_hashes'],entry_hashes=p['entry_hashes'],mode_order=['identity','main'],smoke_seconds=json.loads((B/'smoke/run.json').read_text())['seconds']),indent=2))
for mode in ['identity','main']:
 check()
 with (D/f'{mode}.log').open('x') as log:
  rc=subprocess.call([str(R/'.venv-metaworld/bin/python'),'-u',str(D/'decoder_native.py'),'--mode',mode,'--out',str(B/mode)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 assert rc==0,(mode,rc)
 check();print(mode,'complete',flush=True)
(D/'queue_complete.json').write_text(json.dumps(dict(status='complete')))
