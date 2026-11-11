"""Additional read-only index audit, including gate-selected16-step source options."""
from pathlib import Path
import numpy as np,json
R=Path(str(Path(__file__).resolve().parents[2]));B=R/'outputs/jaamas/decoder_control_20261106_v1';counts=dict(cases=0,suffix_plans=0,raw_plans=0,blend_plans=0,source_steps_exact=0)
for mode in ['smoke','identity','main']:
 for row in json.loads((B/mode/'rows.json').read_text()):
  folder=Path(row['folder']);d=json.loads((folder/'choices_before_outcomes.json').read_text())
  with np.load(folder/'observed_inputs.npz') as z:packets={ch:z['packet_'+ch] for ch in ['S','F']}
  for key,p in d['plans'].items():
   name=p['name']
   if not name.startswith('act_'):continue
   age=row['delay'] if p['decoder']=='legacy' else row['arrival']-d['generation_steps'][p['channel']]
   if name=='act_blend48':counts['blend_plans']+=1;continue
   hold=16 if name.endswith('16') else 48;start=age if name.startswith('act_suffix') else 0
   with np.load(folder/row['branches'][key]['file']) as z:np.testing.assert_array_equal(z['action'][row['arrival']:row['arrival']+hold],packets[p['channel']][start:start+hold])
   counts['suffix_plans' if name.startswith('act_suffix') else 'raw_plans']+=1;counts['source_steps_exact']+=hold
  counts['cases']+=1
assert counts['cases']==68
counts.update(status='passed',scope='direct source suffix/raw commands; blend local component not independently regenerated')
(R/'outputs/jaamas/decoder_revision_20261106/all_source_index_audit.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts,indent=2))
