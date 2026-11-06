"""Stateless visual BC/pose inference; RGB and robot4D are the only inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists());sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from safetensors.torch import load_file
from advice_audit.vision import VisualEncoder,VisualHead
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
root=Path(a.checkpoint);config=json.loads((root/'config.json').read_text());assert config['inputs']==['RGB512corner4verticalflip','robot4D'];encoder=VisualEncoder(config['assets']).cuda().eval();head=VisualHead(config['channels'],config['hidden'],4 if config['target']=='action' else 6).cuda().eval();head.load_state_dict(load_file(str(root/'head.safetensors')),strict=True)
rmean=torch.tensor(config['robot_mean'],device='cuda');rstd=torch.tensor(config['robot_std'],device='cuda');tmean=torch.tensor(config['target_mean'],device='cuda');tstd=torch.tensor(config['target_std'],device='cuda');assert torch.all(rstd>0) and torch.all(tstd>0)
print(json.dumps(dict(status='ready',target=config['target'],head_sha256=hashlib.sha256((root/'head.safetensors').read_bytes()).hexdigest(),inputs=config['inputs'],device=torch.cuda.get_device_name())),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('op')=='close':break
  with np.load(req['input'],allow_pickle=False) as data:images=data['base'];robot=data['state']
  if images.ndim==3:images=images[None];robot=robot[None]
  assert robot.shape==(len(images),4) and np.isfinite(robot).all();start=time.monotonic()
  with torch.inference_mode():
   f=encoder.encode(images).half().float();r=(torch.from_numpy(robot.astype('float32')).cuda()-rmean)/rstd;prediction=head(f,r)*tstd+tmean
   if config['target']=='action':prediction=prediction.clamp(-1,1)
   result=prediction.cpu().numpy()
  assert np.isfinite(result).all();seconds=time.monotonic()-start;np.save(req['output'],result);print(json.dumps(dict(status='ok',seconds=seconds,shape=list(result.shape))),flush=True)
 except Exception as e:print(json.dumps(dict(status='error',type=type(e).__name__,message=str(e))),flush=True)
