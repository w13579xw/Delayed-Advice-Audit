"""Fixed five-initialization MLP comparison, validation/null selection only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'outputs/jaamas')]
from server_run import save,now
from state_core_20261105 import TASKS,Critic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--parity-only',action='store_true');a=p.parse_args();data=Path(a.data).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rows=json.loads((data/'rows.json').read_text());results=[]
 for ti,task in enumerate(TASKS):
  tr=[x for x in rows if x['task']==task and x['split']=='train'];va=[x for x in rows if x['task']==task and x['split']=='validation'];assert len(tr)==30 and len(va)==12
  names=tr[0]['names'];ys={s:np.array([r['target'] for r in rr],dtype=np.float64) for s,rr in [('train',tr),('validation',va)]}
  for packet in (False,True):
   field='packet_features' if packet else 'cheap';xs={s:np.stack([np.load(data/r['file'])[field] for r in rr]) for s,rr in [('train',tr),('validation',va)]};mean=xs['train'].mean(0);std=np.maximum(xs['train'].std(0),.05);xx=torch.tensor((xs['train']-mean)/std,dtype=torch.float64);vx=torch.tensor((xs['validation']-mean)/std,dtype=torch.float64)
   for full in (False,True):
    allowed=list(range(len(names) if full else 6));y=torch.tensor(ys['train'][:,allowed],dtype=torch.float64);vy=torch.tensor(ys['validation'][:,allowed],dtype=torch.float64)
    for seed in range(5):
     torch.manual_seed(seed);rng=np.random.default_rng(seed);model=torch.nn.Sequential(torch.nn.Linear(len(mean),64),torch.nn.Tanh(),torch.nn.Linear(64,32),torch.nn.Tanh(),torch.nn.Linear(32,len(names))).double();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);zero=float(torch.mean(vy[:,1:]**2));bestmse=zero;chosen='zero';best={k:v.detach().clone() for k,v in model.state_dict().items()};history=[];beststep=0
     for step in range(1,1001):
      idx=rng.integers(0,len(tr),16);pred=model(xx[idx])[:,allowed];loss=torch.mean((pred[:,1:]-y[idx,1:])**2);opt.zero_grad();loss.backward();opt.step()
      if step%250==0:
       with torch.no_grad():mse=float(torch.mean((model(vx)[:,allowed][:,1:]-vy[:,1:])**2))
       history.append(dict(step=step,mse=mse))
       if mse<bestmse-1e-12:bestmse=mse;chosen='mlp';beststep=step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
     model.load_state_dict(best);folder=out/task/f'seed{seed}'/('packet' if packet else 'local')/('full' if full else 'cheap');folder.mkdir(parents=True);arrays=dict(mean=mean,std=std)
     for i,key in enumerate((0,2,4),1):arrays[f'w{i}']=best[f'{key}.weight'].numpy();arrays[f'b{i}']=best[f'{key}.bias'].numpy()
     np.savez_compressed(folder/'model.npz',**arrays);config=dict(task=task,seed=seed,input=field,actions='full' if full else 'cheap',model=chosen,names=names,allowed=allowed,zero_mse=zero,validation_mse=bestmse,selected_step=beststep,history=history,training_episodes=30,validation_episodes=12,frozen_utc=now());save(folder/'config.json',config)
     exported=Critic(folder);withx=np.vstack([xs['train'],xs['validation']]);maxerr=0
     with torch.no_grad():ref=model(torch.tensor((withx-mean)/std,dtype=torch.float64)).numpy();ref[:,0]=0
     if chosen=='zero':ref[:]=0
     for x,yy in zip(withx,ref):maxerr=max(maxerr,float(np.max(np.abs(exported.scores(x)-yy))))
     assert maxerr<1e-10;config['export_max_error']=maxerr;results.append(config);print('FIT',task,seed,field,config['actions'],chosen,beststep,flush=True)
 save(out/'analysis.json',results);save(out/'checks.json',dict(checks='passed',models=len(results),export_parity_max_error=max(x['export_max_error'] for x in results)));save(out/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,completed_utc=now()))
if __name__=='__main__':main()
