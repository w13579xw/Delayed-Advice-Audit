"""Train sensory-matched visual BC/pose heads on frozen encoder features."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from safetensors.torch import save_file
from advice_audit.vision import VisualHead
from server_run import fingerprint,save,now


def cache_task(parent,records,device):
 count=sum(r['rows'] for r in records);required=count*512*16*16*2;free,_=torch.cuda.mem_get_info()
 if free<required+4*1024**3:raise RuntimeError(f'GPU cache needs{required}bytes plus4GiBreserve;free={free}')
 features=torch.empty((count,512,16,16),dtype=torch.float16,device=device);robot=[];action=[];pose=[];steps=[];episode=[];offset=0
 for r in records:
  path=parent/r['features'];array=np.load(path,mmap_mode='r',allow_pickle=False);assert array.dtype==np.float16 and array.shape==(r['rows'],512,16,16)
  for start in range(0,len(array),64):
   block=torch.from_numpy(np.array(array[start:start+64],copy=True));features[offset+start:offset+start+len(block)].copy_(block.to(device))
  with np.load(parent/r['targets'],allow_pickle=False) as label:
   assert label['robot'].shape==(len(array),4) and label['target_pose'].shape==(len(array),6)
   robot.append(label['robot']);action.append(label['target_action']);pose.append(label['target_pose']);steps.append(label['step']);episode.extend([r['episode']]*len(array))
  offset+=len(array)
 torch.cuda.synchronize();return features,np.concatenate(robot),np.concatenate(action),np.concatenate(pose),np.concatenate(steps),np.asarray(episode)


def train_one(root,features,robot,targets,train_indices,val_indices,steps,args,target,channels,hidden,normalization):
 torch.manual_seed(args.seed);np.random.seed(args.seed);device=features.device;model=VisualHead(channels,hidden,targets.shape[1]).to(device)
 train_targets=targets[train_indices];mean=train_targets.mean(0) if target=='pose' else np.zeros(targets.shape[1],dtype='float32');std=np.maximum(train_targets.std(0),1e-3) if target=='pose' else np.ones(targets.shape[1],dtype='float32')
 normalized=(targets-mean)/std;robot_mean=np.array(normalization['robot_mean'],dtype='float32');robot_std=np.array(normalization['robot_std'],dtype='float32');assert (robot_std>0).all()
 robot_tensor=torch.from_numpy((robot-robot_mean)/robot_std).to(device);target_tensor=torch.from_numpy(normalized).to(device)
 path=root/f'{target}_c{channels}_h{hidden}';path.mkdir();config=dict(task=args.task,target=target,channels=channels,hidden=hidden,seed=args.seed,assets=str((ROOT/'outputs/vision/encoder_assets_20261103').resolve()),robot_mean=robot_mean.tolist(),robot_std=robot_std.tolist(),target_mean=mean.tolist(),target_std=std.tolist(),features='FP32encoder->FP16quantization->FP32head',native_action_convention='no x flip for these native teacher labels',inputs=['RGB512corner4verticalflip','robot4D'],offline_only_targets='pose6D is supervision, not model input',overfit_debug=args.overfit_only)
 save(path/'config.json',config);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(args.seed);weights=np.ones(len(train_indices))*.5/len(train_indices);early=steps[train_indices]<100;weights+=early*.5/early.sum();best=float('inf');history=[];started=time.monotonic()
 def validation():
  model.eval();total=0.;native_squared=np.zeros(targets.shape[1]);n=0
  with torch.inference_mode():
   for start in range(0,len(val_indices),128):
    ids=torch.as_tensor(val_indices[start:start+128],device=device);prediction=model(features[ids].float(),robot_tensor[ids]);error=prediction-target_tensor[ids];total+=float(error.square().sum().item());native_squared+=(error.square().sum(0).cpu().numpy()*std**2);n+=len(ids)
  return total/(n*targets.shape[1]),np.sqrt(native_squared/n).tolist()
 for epoch in range(args.epochs):
  model.train();order=train_indices[rng.choice(len(train_indices),size=len(train_indices),replace=True,p=weights)];total=0.;n=0
  for start in range(0,len(order),64):
   ids=torch.as_tensor(order[start:start+64],device=device);prediction=model(features[ids].float(),robot_tensor[ids]);loss=(prediction-target_tensor[ids]).square().mean();assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step();total+=float(loss.detach().item())*len(ids);n+=len(ids)
  val,rmse=validation() if len(val_indices) else (None,None);score=val if val is not None else total/n
  row=dict(epoch=epoch+1,train_weighted_sample_mse=total/n,validation_mse=val,validation_native_rmse=rmse,seconds=time.monotonic()-started);history.append(row);save(path/'history.json',history)
  if score<best:
   best=score;save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path/'head.safetensors'));save(path/'selected_checkpoint.json',dict(epoch=epoch+1,score=best,selection='offline validation MSE' if len(val_indices) else 'overfit-debug training MSE; no validation claim'))
  print('EPOCH',args.task,target,channels,hidden,row,flush=True)
 save(path/'metrics.json',dict(best_validation_mse=best if len(val_indices) else None,train_rows=len(train_indices),validation_rows=len(val_indices),seconds=time.monotonic()-started,head_sha256=hashlib.sha256((path/'head.safetensors').read_bytes()).hexdigest(),overfit_debug=args.overfit_only))
 del model,optimizer;torch.cuda.empty_cache()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--task',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=20);p.add_argument('--targets',nargs='+',choices=['action','pose'],default=['action','pose']);p.add_argument('--overfit-only',action='store_true');a=p.parse_args();parent=Path(a.data).resolve();root=Path(a.out).resolve()
 if root.exists():p.error('new run directory required')
 assert json.loads((parent/'run.json').read_text())['status']=='complete';assert json.loads((parent/'checks.json').read_text())['checks']=='passed';records=sorted([r for r in json.loads((parent/'episodes.json').read_text()) if r['task']==a.task],key=lambda x:x['episode'])
 if not a.overfit_only:assert len(records)==200
 else:records=records[:1]
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);root.mkdir(parents=True);code=fingerprint();t0=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
 train_ids=[r['episode'] for r in records[:180]];val_ids=[] if a.overfit_only else [r['episode'] for r in records[180:]]
 save(root/'protocol.json',dict(config=vars(a),frozen_utc=now(),train_episode_ids=train_ids,validation_episode_ids=val_ids,capacities=[[16,128]] if a.overfit_only else [[16,128],[32,256]],optimizer='AdamWlr3e-4,wd1e-4,batch64,clipnorm1,max20epochs',sampler='halfearly<100/halfuniform,withreplacement',cache='float16GPUresident,4GiBminimumreserve',no_privileged_online_features=True,parent_run=json.loads((parent/'run.json').read_text()),parent_episodes_sha256=hashlib.sha256((parent/'episodes.json').read_bytes()).hexdigest()))
 for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
  d=root/'source_snapshot'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
 try:
  features,robot,actions,poses,steps,episodes=cache_task(parent,records,torch.device('cuda'));train=np.flatnonzero(np.isin(episodes,train_ids));val=np.flatnonzero(np.isin(episodes,val_ids));assert set(train_ids).isdisjoint(val_ids);assert robot.shape[1]==4
  normalization=json.loads((ROOT/'outputs/vision/encoder_assets_20261103/normalization.json').read_text());print('CACHED',a.task,list(features.shape),'GiB',features.numel()*features.element_size()/1024**3,flush=True)
  for target in a.targets:
   for channels,hidden in ([[16,128]] if a.overfit_only else [[16,128],[32,256]]):train_one(root,features,robot,actions if target=='action' else poses,train,val,steps,a,target,channels,hidden,normalization)
  assert code==fingerprint();save(root/'checks.json',dict(checks='passed',episode_split_disjoint=True,robot_input_dimensions=4,overfit_debug=a.overfit_only,trained_models=len(a.targets)*(1 if a.overfit_only else 2)));state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-t0,exit_code=0)
 except BaseException as e:state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state);raise
 save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
