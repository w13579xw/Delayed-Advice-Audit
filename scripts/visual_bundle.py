"""Simulator-side observed-input-only bundle client and cheap controls."""
import json,subprocess
from pathlib import Path
import numpy as np
from advice_audit.act_bridge import ACTClient,ACTCamera
from evaluate_visual_heads import pose_action
class BundleClient(ACTClient):
 def predict(self,robot,image,source=False,features=False):
  robot=np.asarray(robot);image=np.asarray(image)
  if robot.shape!=(4,) or image.shape!=(512,512,3):raise ValueError('Only RGB+robot4 allowed')
  src=self.scratch/'request.npz';out=self.scratch/'response.npz';np.savez(src,state=robot,base=image)
  self.process.stdin.write(json.dumps(dict(input=str(src),output=str(out),kind='source' if source else 'cheap',features=bool(features)))+'\n');self.process.stdin.flush();response=self._response()
  if response.get('status')!='ok':raise RuntimeError(response)
  with np.load(out,allow_pickle=False) as z:prediction={k:z[k][0].copy() for k in z.files}
  return prediction,response

def actions(task,robot,prediction,manifest):
 result=[]
 for i,c in enumerate(manifest['configs']):
  value=prediction[f'head{i}'];result.append(pose_action(task,robot,value) if c['target']=='pose' else np.clip(value,-1,1))
 return result  # Preserve action-head float32 vs native pose-policy float64 semantics.

def make_manifest(task,root,ROOT,source_override=None):
 training=ROOT/'experiments/visual_fit'/('head_training_push_20261103_v1' if task=='push-v3' else 'head_training_pick_20261103_v1')
 heads=[training/name for name in ('action_c16_h128','action_c32_h256','pose_c16_h128','pose_c32_h256')]
 freeze=json.loads((ROOT/'outputs/vision/development_summary_20261103_v1'/(task+'_freeze.json')).read_text());selected=[v.resolve() for v in heads].index(Path(freeze['selected']['model']).resolve())
 sourcefreeze=json.loads((ROOT/'outputs/vision/matched_source_development_20261103_v1'/(task+'_matched_freeze.json')).read_text())
 if source_override is not None:
  source=Path(source_override).resolve();meta=json.loads((source/'metadata.json').read_text());assert meta['task']==task and not meta.get('smoke_only',False);sourcefreeze=dict(model=str(source),source_metadata=meta,override_reason='explicit frozen source for new experiment; prior source metadata remains in prior read-only artifacts')
 result=dict(task=task,heads=[str(v) for v in heads],configs=[json.loads((v/'config.json').read_text()) for v in heads],selected=selected,source=sourcefreeze['model'],source_freeze=sourcefreeze)
 (root/(task+'_manifest.json')).write_text(json.dumps(result,indent=2));return result,root/(task+'_manifest.json')

class DeterministicCamera(ACTCamera):
 """Explicit new evaluator observation protocol: offscreen MSAA=0.

 The accepted training data and old competence scores used native MSAA=4.
 Those stay untouched. This camera is only for new visual-value runs after
 rendering repeatability and frozen-controller competence checks.
 """
 def __init__(self,env):
  env.env.model.vis.quality.offsamples=0
  super().__init__(env)
