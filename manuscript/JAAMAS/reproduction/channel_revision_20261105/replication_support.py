"""Frozen actor selection for complete visual training replicas; no outcome use."""
import os,json,hashlib
from pathlib import Path
R=Path('/home/ez/DACI/code');TRAIN=R/'outputs/jaamas/visual_full_training_20261105_v1'
def actor_seed():return int(os.environ['DACI_REPLICATION_ACTOR_SEED'])
def source_path(task,seed=None):
 seed=actor_seed() if seed is None else seed;tag='push' if task=='push-v3' else 'pick'
 return R/'outputs/vision'/f'act_budget15k_{tag}_20261103_v1/best' if seed==0 else TRAIN/f'{tag}_act15k_seed{seed}'/'best'
def make_manifest(task,root,ROOT,source_override=None):
 seed=actor_seed();tag='push' if task=='push-v3' else 'pick';training=R/'outputs/vision'/f'head_training_{tag}_20261103_v1' if seed==0 else TRAIN/f'{tag}_heads_seed{seed}';heads=[training/name for name in ('action_c16_h128','action_c32_h256','pose_c16_h128','pose_c32_h256')];source=source_path(task,seed);meta=json.loads((source/'metadata.json').read_text());assert meta['task']==task and not meta.get('smoke_only',False);result=dict(task=task,actor_seed=seed,heads=[str(v) for v in heads],configs=[json.loads((v/'config.json').read_text()) for v in heads],selected=0 if task=='push-v3' else 1,source=str(source),source_freeze=dict(model=str(source),source_metadata=meta,reason='validationonly15k replicate'))
 path=root/(task+'_manifest.json');path.write_text(json.dumps(result,indent=2));return result,path
