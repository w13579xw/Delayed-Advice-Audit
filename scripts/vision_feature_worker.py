"""Frozen visual-feature worker in the isolated Torch/CUDA environment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from advice_audit.vision import VisualEncoder
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
assets=Path(a.checkpoint);manifest=json.loads((assets/'manifest.json').read_text());assert hashlib.sha256((assets/'encoder.safetensors').read_bytes()).hexdigest()==manifest['encoder_sha256'];encoder=VisualEncoder(assets).cuda().eval()
print(json.dumps(dict(status='ready',encoder_sha256=manifest['encoder_sha256'],torch=torch.__version__,device=torch.cuda.get_device_name(),output='B,512,16,16float16; features only')),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('op')=='close':break
  with np.load(req['input'],allow_pickle=False) as data:images=data['base']
  if images.ndim==3:images=images[None]
  start=time.monotonic()
  with torch.inference_mode():features=encoder.encode(images).half().cpu().numpy()
  elapsed=time.monotonic()-start;assert features.shape==(len(images),512,16,16) and np.isfinite(features).all();np.save(req['output'],features)
  print(json.dumps(dict(status='ok',seconds=elapsed,shape=list(features.shape))),flush=True)
 except Exception as e:print(json.dumps(dict(status='error',type=type(e).__name__,message=str(e))),flush=True)
