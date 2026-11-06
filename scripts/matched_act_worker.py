"""GPU ACT model-core inference with the verified recorded feature path."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from advice_audit.act_adaptation import adapted_adapter
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();root=Path(a.checkpoint);meta=json.loads((root/'metadata.json').read_text());assert hashlib.sha256((root/'core.safetensors').read_bytes()).hexdigest()==meta['core_sha256']
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);model=adapted_adapter(root).eval()
print(json.dumps(dict(status='ready',device=torch.cuda.get_device_name(),model_sha256=meta['core_sha256'],inputs=['RGB','robot4D'],precision='FP32inference,FP16frozenfeatures')),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('op')=='close':break
  with np.load(req['input'],allow_pickle=False) as data:robot=data['state'];images=data['base']
  if images.ndim==3:robot=robot[None];images=images[None]
  start=time.monotonic()
  with torch.inference_mode():out=model.cached_image_output(robot,images).cpu().numpy()
  assert out.shape==(len(images),100,4) and np.isfinite(out).all();seconds=time.monotonic()-start;np.save(req['output'],out);print(json.dumps(dict(status='ok',seconds=seconds,shape=list(out.shape))),flush=True)
 except Exception as e:print(json.dumps(dict(status='error',type=type(e).__name__,message=str(e))),flush=True)
