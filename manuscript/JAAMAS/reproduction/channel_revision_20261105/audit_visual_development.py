"""Read-only raw native-label and normalization audit for one complete replica."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from daci.value_kernel import FrozenKernelDecisions,select_global_minimum
from server_run import save,now
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--models',required=True);p.add_argument('--out',required=True);a=p.parse_args();D=Path(a.data);M=Path(a.models);O=Path(a.out);assert json.loads((D/'run.json').read_text())['status']=='complete';assert json.loads((M/'run.json').read_text())['status']=='complete';O.mkdir(exist_ok=False);rows=json.loads((D/'rows.json').read_text());assert len(rows)==40;count=0;groups_by_task={}
for task in ['push-v3','pick-place-v3']:
 rr=[r for r in rows if r['task']==task];train=sorted([r for r in rr if r['split']=='train'],key=lambda r:r.get('train_rank',r['seed']));val=[r for r in rr if r['split']=='validation'];assert len(train)==14 and len(val)==6;assert set(r['seed'] for r in train).isdisjoint(r['seed'] for r in val);pol=FrozenKernelDecisions(M,task);groups=[[] for _ in range(6)];trainnames=list(train[0]['outcomes'])
 for row in train+val:
  stem=f"{task}_{row['seed']}";arrival=row['query']+row['delay'];assert arrival==row['arrival'];obs={}
  for label in ['request','arrival']:
   with np.load(D/row[label+'_inputs'],allow_pickle=False) as z:obs[label]={k:z[k].copy() for k in z.files};assert set(z.files)=={'visual_features','robot','head0','head1','head2','head3'}
  with np.load(D/row['source_packet'],allow_pickle=False) as z:packet=z['source'].copy();assert hashlib.sha256(packet.tobytes()).hexdigest()==row['source_packet_sha256']
  q,ar=obs['request'],obs['arrival'];qr=q['robot'].astype('float64');av=ar['robot'].astype('float64');gg=[q['visual_features'].astype('float32').ravel(),ar['visual_features'].astype('float32').ravel(),np.r_[qr,av,av-qr],np.concatenate([x[k] for x in [q,ar] for k in ['head0','head1','head2','head3']]),np.array([row['query']/500,row['delay']/100,arrival/500]),packet.ravel()]
  for group,value in zip(groups,gg):group.append(value)
  with np.load(D/(stem+'_baseline.npz'),allow_pickle=False) as z:base=z['rewards'].copy();assert z['actions'].shape==(500,4) and base.shape==(500,)
  for name,v in row['outcomes'].items():
   if name=='reject':reward=base
   elif row['delay']==0 and name.startswith('act_suffix'):assert v==row['outcomes'][name.replace('suffix','raw')];continue
   else:
    with np.load(D/(stem+'_'+name+'.npz'),allow_pickle=False) as z:reward=z['rewards'].copy();assert z['actions'].shape==(500-arrival,4)
   assert reward.shape==(500,) and float(reward.mean())==v['native_return'];assert np.isfinite(reward).all() and reward.min()>=-1e-12 and reward.max()<=10+1e-12;np.testing.assert_array_equal(reward[:arrival],base[:arrival]);count+=1
 td=[];vd=[]
 for i,xx in enumerate(groups):
  rawall=np.array(xx,dtype='float32');raw=rawall[:14];mean=raw.mean(0);std=np.maximum(raw.std(0),.05);np.testing.assert_array_equal(mean,pol.geometry[f'mean_{i}']);np.testing.assert_array_equal(std,pol.geometry[f'std_{i}']);xt=((raw-mean)/std).astype('float64');xv=((rawall[14:]-mean)/std).astype('float64');np.testing.assert_array_equal(xt,pol.geometry[f'train_{i}'])
  dt=np.maximum((xt*xt).mean(1)[:,None]+(xt*xt).mean(1)[None,:]-2*(xt@xt.T)/xt.shape[1],0.);np.fill_diagonal(dt,0.);positive=dt[np.triu_indices(14,1)];positive=positive[positive>1e-12];scale=float(np.median(positive)) if len(positive) else 1.;assert scale==pol.norm[str(i)]['distance_scale'];td.append(dt/scale);vd.append(np.maximum((xv*xv).mean(1)[:,None]+(xt*xt).mean(1)[None,:]-2*(xv@xt.T)/xt.shape[1],0.)/scale)
 # Independently replay candidate-selection order from saved grid and raw targets.
 y=np.array([[(r['outcomes'][n]['native_return']-r['outcomes']['reject']['native_return'])/10 for n in trainnames] for r in val])
 yt=np.array([[(r['outcomes'][n]['native_return']-r['outcomes']['reject']['native_return'])/10 for n in trainnames] for r in train])
 for f in ['cheap_inputs','with_packet']:
  for ac in ['cheap_actions','all_actions']:
   cell=pol.analysis['models'][f]['decision_cells'][ac];ids=[trainnames.index(n) for n in cell['allowed_names']];stem=task+'_native_return_'+f+'_'+ac;grid=json.loads((M/(stem+'_grid.json')).read_text());block=5 if f=='cheap_inputs' else 6
   for g in grid:
    kt=np.exp(-sum(td[:block])/(2*g['bandwidth']**2));kv=np.exp(-sum(vd[:block])/(2*g['bandwidth']**2));dual=np.linalg.solve(kt+g['ridge']*np.eye(14),yt);pred=kv@dual;pred[:,0]=0.;np.testing.assert_allclose(float(np.mean((pred[:,ids]-y[:,ids])**2)),g['validation_mse'],atol=1e-14,rtol=1e-12)
   candidates=[dict(model='zero',bandwidth=None,ridge=None,validation_mse=float(np.mean(y[:,ids]**2)))]+[dict(model='RBFkernelridge',**g) for g in grid];chosen=select_global_minimum(candidates)
   for key in ['model','bandwidth','ridge','validation_mse','global_minimum_mse']:assert chosen[key]==cell['selected'][key]
  
 groups_by_task[task]=dict(training=14,validation=6,raw_input_normalization_exact=True,global_tie_selection_exact=True)
save(O/'checks.json',dict(checks='passed',logical_episodes=len(rows),audited_native_labels=count,raw_reward_error=0,train_only_normalization=True,paired_prefixes_equal=True,no_refit=True,tasks=groups_by_task));save(O/'run.json',dict(status='complete',exit_code=0,utc=now()));print('AUDITED',O,count)
