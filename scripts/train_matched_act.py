"""Adapt frozen-encoder ACT on the same complete demos as visual cheap heads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from safetensors.torch import save_file
from advice_audit.act_adaptation import original_adapter
from train_visual_heads import cache_task
from server_run import fingerprint,save,now


def future_targets(parent,records,task):
    protocol=json.loads((parent/'protocol.json').read_text());entry=next(r for r in protocol['sources'] if r['task']==task)
    data=np.load(ROOT/entry['path'],allow_pickle=False);actions=data['action'];episodes=data['episode'];future=[];padding=[]
    for row in records:
        indices=np.flatnonzero(episodes==row['episode']);assert len(indices)==500
        with np.load(parent/row['targets']) as labels:steps=labels['step']
        grid=steps[:,None]+np.arange(100)[None,:];pad=grid>=500;values=actions[indices[np.minimum(grid,499)]].astype('float32');values[pad]=0
        future.append(values);padding.append(pad)
    return np.concatenate(future),np.concatenate(padding)


def save_model(path,model,task,step,score,smoke=False):
    path.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.core.state_dict().items()},str(path/'core.safetensors'))
    save(path/'source_config.json',model.raw_config);save(path/'normalization.json',model.statistics)
    save(path/'metadata.json',dict(task=task,step=step,smoke_only=smoke,validation_masked_l1=score,inputs=['RGB512corner4verticalflip','robot4D'],frozen_backbone=True,features='FP32encoder->FP16quantization->FP32ACT',parent_checkpoint_sha256='7f2472be5ecf68cf99b031583c79bbd2bf8981145a4fbcba9ba3c4f69cb70d04',core_sha256=hashlib.sha256((path/'core.safetensors').read_bytes()).hexdigest()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--task',required=True);p.add_argument('--updates',type=int,default=3000);p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();parent=Path(a.data).resolve();root=Path(a.out).resolve()
    if root.exists():p.error('New run directory required')
    assert json.loads((parent/'run.json').read_text())['status']=='complete';assert json.loads((ROOT/'outputs/vision/act_parity_20261103_v2/checks.json').read_text())['checks']=='passed'
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    records=sorted([r for r in json.loads((parent/'episodes.json').read_text()) if r['task']==a.task],key=lambda r:r['episode']);assert len(records)==200
    root.mkdir(parents=True);code=fingerprint();started=time.monotonic();state=dict(status='running',created_utc=now(),source_fingerprint=code);save(root/'run.json',state)
    amp=torch.cuda.is_bf16_supported();save(root/'protocol.json',dict(config=vars(a),frozen_utc=now(),data=str(parent),train_episodes=[r['episode'] for r in records[:180]],validation_episodes=[r['episode'] for r in records[180:]],batch_size=16,updates=a.updates,lr=1e-5,weight_decay=1e-4,kl_weight=10,loss='original masked normalizedL1.mean+10KL',precision='BF16autocast/FP32weights' if amp else 'FP32',validation_precision='FP32',sampler='halfearly<100/halfuniform',grad_clip=1,backbone_frozen=True,pose_not_input=True,parent_manifest_sha256=hashlib.sha256((parent/'episodes.json').read_bytes()).hexdigest()))
    for f in [*ROOT.glob('src/**/*.py'),*ROOT.glob('scripts/*.py')]:
        d=root/'source_snapshot'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
    try:
        features,robot,_,_,steps,episodes=cache_task(parent,records,torch.device('cuda'));native,padding=future_targets(parent,records,a.task);model=original_adapter(ROOT/'outputs/learned_source_intake/act_checkpoint');model.eval();robot=torch.from_numpy(robot).cuda();targets=model.normalize_targets(torch.from_numpy(native).cuda());padding=torch.from_numpy(padding).cuda()
        with torch.no_grad():model.cached_output(robot[:1],features[:1].float())
        train=np.flatnonzero(np.isin(episodes,[r['episode'] for r in records[:180]]));val=np.flatnonzero(np.isin(episodes,[r['episode'] for r in records[180:]]));assert set(episodes[train]).isdisjoint(episodes[val]);weights=np.ones(len(train))*.5/len(train);early=steps[train]<100;weights+=early*.5/early.sum();rng=np.random.default_rng(a.seed)
        optimizer=torch.optim.AdamW([p for p in model.core.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4);history=[];best=float('inf');torch.cuda.reset_peak_memory_stats()
        def validate():
            model.eval();total=0.;count=0
            with torch.inference_mode():
                for start in range(0,len(val),16):
                    ids=torch.as_tensor(val[start:start+16],device='cuda');prediction,_=model.cached_output(robot[ids],features[ids].float());loss=((prediction-targets[ids]).abs()*(~padding[ids]).unsqueeze(-1)).mean();total+=float(loss.item())*len(ids);count+=len(ids)
            return total/count
        initial=validate();save(root/'initial_validation.json',dict(masked_l1=initial));print('INITIAL',a.task,initial,flush=True)
        for update in range(1,a.updates+1):
            model.train();ids=torch.as_tensor(train[rng.choice(len(train),size=16,replace=True,p=weights)],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                prediction,(mu,logvar)=model.cached_output(robot[ids],features[ids].float(),targets[ids],padding[ids]);l1=((prediction.float()-targets[ids]).abs()*(~padding[ids]).unsqueeze(-1)).mean();kl=(-.5*(1+logvar.float()-mu.float().square()-logvar.float().exp()).sum(1)).mean();loss=l1+10*kl
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.core.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);optimizer.step()
            if update==1 or update%50==0 or update==a.updates:
                record=dict(update=update,train_masked_l1=float(l1.item()),train_kl=float(kl.item()),train_loss=float(loss.item()),gradient_norm_before_clip=float(norm.item()),seconds=time.monotonic()-started);history.append(record);save(root/'history.json',history);print('UPDATE',a.task,record,flush=True)
            if update%250==0 or update==a.updates:
                score=validate();history[-1]['validation_masked_l1']=score;save(root/'history.json',history)
                if score<best:best=score;save_model(root/'best',model,a.task,update,score,smoke=a.smoke)
                print('VALIDATION',a.task,update,score,flush=True)
        assert code==fingerprint();save(root/'metrics.json',dict(initial_validation_masked_l1=initial,best_validation_masked_l1=best,updates=a.updates,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,smoke_only=a.smoke));save(root/'checks.json',dict(checks='passed',all_gradients_finite=True,backbone_frozen=not any(p.requires_grad for p in model.core.backbone.parameters()),episode_split_disjoint=True,robot_input_dim=4,smoke_only=a.smoke));state.update(status='complete',completed_utc=now(),exit_code=0,seconds=time.monotonic()-started)
    except BaseException as e:state.update(status='failed',error=str(e),ended_utc=now());save(root/'run.json',state);raise
    save(root/'run.json',state);print('COMPLETE',root,flush=True)
if __name__=='__main__':main()
