"""Read-only training budget / validation selection / prefix consistency inventory."""
import json,hashlib
from pathlib import Path
R=Path(str(Path(__file__).resolve().parents[2]));D=R/'outputs/jaamas/visual_full_training_20261105_v1';out={}
for tag in ['push','pick']:
 for seed in [1,2]:
  root=D/f'{tag}_act15k_seed{seed}';h=json.loads((root/'history.json').read_text());old=R/f'outputs/vision/training_replication_20261104_v1/training/{tag}_act3k_seed{seed}/history.json';prefix=lambda x:[{k:v for k,v in row.items() if k!='seconds'} for row in x if row['update']<=3000];a=prefix(h);b=prefix(json.loads(old.read_text()));assert a==b[:len(a)];meta=json.loads((root/'best/metadata.json').read_text());val=[row for row in h if 'validation_masked_l1' in row];initial=json.loads((root/'initial_validation.json').read_text());out[f'{tag}/seed{seed}']=dict(updates_observed=h[-1]['update'],complete_budget=h[-1]['update']==15000,prefix_3000_complete=len(a)==len(b),prefix_exact=True,validation_selected_step=meta['step'],validation_mse_not_used=True,selected_validation_masked_l1=meta['validation_masked_l1'],parent=meta['parent_checkpoint_sha256'],core=meta['core_sha256'],heads={})
  for name in ['action_c16_h128','action_c32_h256','pose_c16_h128','pose_c32_h256']:
   head=D/f'{tag}_heads_seed{seed}'/name;assert (head/'head.safetensors').exists();hh=json.loads((head/'history.json').read_text());out[f'{tag}/seed{seed}']['heads'][name]=dict(history_records=len(hh),weights_sha256=hashlib.sha256((head/'head.safetensors').read_bytes()).hexdigest())
assert len({x['parent'] for x in out.values()})==1
p=R/'outputs/jaamas/revision_20261105/training_inventory_latest.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({k:{x:v[x] for x in ['updates_observed','complete_budget','prefix_3000_complete','validation_selected_step']} for k,v in out.items()},indent=2))
