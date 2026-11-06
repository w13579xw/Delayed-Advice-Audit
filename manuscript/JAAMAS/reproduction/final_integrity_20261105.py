import json,hashlib
from pathlib import Path
import sys
import numpy as np
r=Path(__file__).resolve().parents[2];sys.path[:0]=[str(r/'outputs/jaamas')];from state_core_20261105 import noise_stream
root=r/'outputs/jaamas/state_campaign_20261105_v1';entry=json.loads((root/'frozen_entries.json').read_text());assert all(hashlib.sha256((r/'outputs/jaamas'/p).read_bytes()).hexdigest()==h for p,h in entry['sha256'].items());models=json.loads((root/'models/analysis.json').read_text());errors=[]
for c in models:
 candidates=[(0,c['zero_mse'])]+[(v['step'],v['mse']) for v in c['history']];minimum=min(v for _,v in candidates);expected=min(s for s,v in candidates if v<=minimum+1e-12)
 assert expected==c['selected_step']
cases=[json.loads(p.read_text()) for p in root.glob('test/*/cell*/case.json')];seeds=set();nullchecks=0;noisechecks=0
for c in cases:
 assert c['seed'] not in seeds;seeds.add(c['seed']);p=root/'test'/c['task']/f"cell{c['cell']:02d}_ep{c['index']:02d}"
 for leaf in {v['leaf'] for v in c['results'].values()}:
  with np.load(p/f'leaf{leaf:03d}.npz') as z:np.testing.assert_array_equal(z['noise'],noise_stream(c['seed']+1000000,c['sigma']));noisechecks+=1
 for label,v in c['results'].items():
  if label=='reject' or label=='fixed_expert32':continue
  s,inp,act=label.split('/');conf=json.loads((root/'models'/c['task']/s/inp/act/'config.json').read_text())
  if conf['model']=='zero':assert all(x['choice']=='reject' for x in v['decisions']);assert v['leaf']==c['results']['reject']['leaf'];nullchecks+=1
normalizations=0
rows=json.loads((root/'development/rows.json').read_text())
for task in ['drawer-open-v3','door-open-v3','button-press-v3']:
 tr=[x for x in rows if x['task']==task and x['split']=='train'];assert len(tr)==30
 for inp,field in [('local','cheap'),('packet','packet_features')]:
  xx=np.stack([np.load(root/'development'/x['file'])[field] for x in tr]);mean=xx.mean(0);std=np.maximum(xx.std(0),.05)
  for seed in range(5):
   for act in ['cheap','full']:
    with np.load(root/'models'/task/f'seed{seed}'/inp/act/'model.npz') as z:np.testing.assert_array_equal(z['mean'],mean);np.testing.assert_array_equal(z['std'],std);normalizations+=1
out=root/'additional_integrity.json';out.write_text(json.dumps(dict(checks='passed',entry_hashes=len(entry['sha256']),training_only_normalizations_reconstructed=normalizations,global_tie_parity_models=60,noise_arrays_verified=noisechecks,null_policy_episode_identities=nullchecks,unique_environment_seeds=len(seeds),note='Extra forensic checks after planned analysis; no refit or additional outcome hypotheses'),indent=2));print(out.read_text())
