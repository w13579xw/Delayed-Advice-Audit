"""All cheap heads and optional ACT from one RGB+robot4 encoder call."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from safetensors.torch import load_file
from advice_audit.vision import VisualEncoder,VisualHead
from advice_audit.act_adaptation import adapted_adapter
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args();manifest=json.loads(Path(a.checkpoint).read_text())
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
configs=[];heads=[];hashes=[]
for path in manifest['heads']:
 root=Path(path);c=json.loads((root/'config.json').read_text());assert c['inputs']==['RGB512corner4verticalflip','robot4D'];assert c['task']==manifest['task'];assert not c.get('overfit_debug',False)
 head=VisualHead(c['channels'],c['hidden'],4 if c['target']=='action' else 6).cuda().eval();head.load_state_dict(load_file(str(root/'head.safetensors')),strict=True);heads.append(head);configs.append(c);hashes.append(hashlib.sha256((root/'head.safetensors').read_bytes()).hexdigest())
assert len(heads)==4
encoder=VisualEncoder(configs[0]['assets']).cuda().eval();source=adapted_adapter(manifest['source']).eval();metadata=json.loads((Path(manifest['source'])/'metadata.json').read_text());assert metadata['task']==manifest['task'] and not metadata['smoke_only'];assert hashlib.sha256((Path(manifest['source'])/'core.safetensors').read_bytes()).hexdigest()==metadata['core_sha256']
encoder_state=encoder.backbone.state_dict();source_state=source.core.backbone.state_dict();assert encoder_state.keys()==source_state.keys();assert all(torch.equal(encoder_state[k],source_state[k]) for k in encoder_state), 'Shared encoder must be bitwise identical to source frozen backbone'
stats=[{key:torch.tensor(c[key],device='cuda') for key in ('robot_mean','robot_std','target_mean','target_std')} for c in configs]
print(json.dumps(dict(status='ready',heads_sha256=hashes,source_metadata=metadata,inputs=['RGB512corner4verticalflip','robot4D'],stateless=True)),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('op')=='close':break
  with np.load(req['input'],allow_pickle=False) as d:images=d['base'];robot=d['state']
  if images.ndim==3:images=images[None];robot=robot[None]
  assert robot.shape==(len(images),4) and np.isfinite(robot).all();start=time.monotonic()
  with torch.inference_mode():
   f=encoder.encode(images).half().float();r=torch.as_tensor(robot,device='cuda',dtype=torch.float32);pred=[]
   for h,c,s in zip(heads,configs,stats):
    y=h(f,(r-s['robot_mean'])/s['robot_std'])*s['target_std']+s['target_mean'];pred.append(y.cpu().numpy())
   result={f'head{i}':v for i,v in enumerate(pred)}
   if req.get('features',False):result['features']=f.half().cpu().numpy()
   if req.get('kind')=='source':
    raw,_=source.cached_output(r,f);native=source.unnormalize(raw)*source.source_sign;result['source']=native.clamp(-1,1).cpu().numpy()
  assert all(np.isfinite(v).all() for v in result.values());np.savez(req['output'],**result);print(json.dumps(dict(status='ok',seconds=time.monotonic()-start)),flush=True)
 except Exception as e:print(json.dumps(dict(status='error',type=type(e).__name__,message=str(e))),flush=True)
