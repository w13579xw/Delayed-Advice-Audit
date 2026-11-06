"""CPU ACT chunk worker; separate interpreter avoids simulator Gym conflict."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch,draccus
from safetensors.torch import load_file
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
root=Path(a.checkpoint);cfg=json.loads((root/'config.json').read_text());cfg.pop('type');cfg['device']='cpu';cfg['pretrained_backbone_weights']=None
config=draccus.decode(ACTConfig,cfg)
t0=time.monotonic();policy=ACTPolicy(config);result=policy.load_state_dict(load_file(str(root/'model.safetensors'),device='cpu'),strict=True);policy.eval()
print(json.dumps(dict(status='ready',load_seconds=time.monotonic()-t0,missing=list(result.missing_keys),unexpected=list(result.unexpected_keys))),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('op')=='close':break
  with np.load(req['input'],allow_pickle=False) as data:
   state=np.asarray(data['state'],dtype=np.float32);base=data['base'];wrist=data['wrist']
  if state.ndim==1:state=state[None];base=base[None];wrist=wrist[None]
  assert state.shape[1:]==(4,) and base.shape[1:]==wrist.shape[1:]==(512,512,3)
  batch={'observation.state':torch.from_numpy(state),
    'observation.images.base':torch.from_numpy(np.ascontiguousarray(base.transpose(0,3,1,2))).float()/255,
    'observation.images.wrist':torch.from_numpy(np.ascontiguousarray(wrist.transpose(0,3,1,2))).float()/255}
  t0=time.monotonic()
  with torch.inference_mode():out=policy.predict_action_chunk(batch).cpu().numpy()
  assert out.shape==(len(state),100,4) and np.isfinite(out).all();np.save(req['output'],out)
  print(json.dumps(dict(status='ok',seconds=time.monotonic()-t0,shape=list(out.shape))),flush=True)
 except Exception as e:
  print(json.dumps(dict(status='error',type=type(e).__name__,message=str(e))),flush=True)
