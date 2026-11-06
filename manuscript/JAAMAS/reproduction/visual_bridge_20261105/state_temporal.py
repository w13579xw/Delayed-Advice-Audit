"""Posthoc descriptive nonzero-delay and Door temporal diagnostics; no primary retesting."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');O=R/'outputs/jaamas/state_temporal_20261105_v1';O.mkdir(exist_ok=False)
root=R/'outputs/jaamas/channel_revision_20261105_v1/main';rows=json.loads((root/'rows.json').read_text())
proto=dict(status='frozen_before_aggregation',rng=2026110564,resamples=30000,confidence=.95,role='exploratory; original primary families unchanged',subsets={'nonzero':[2,3,4,5],'delay16':[2,3],'delay64':[4,5]},unit='environment within original delay/noise cells crossed with five critic seeds',door='all60 environments, all5packet/fullcritics, allS/F/Echannels; no outcome selection')
(O/'protocol.json').write_text(json.dumps(proto,indent=2))
vectors=np.load(R/'outputs/jaamas/channel_revision_analysis_20261105_v2/primary_vectors.npz')
results={};curves={};temporal={}
def boot(x,cells,seed):
 rng=np.random.default_rng(seed);vals=[]
 for _ in range(60):
  si=rng.integers(0,5,(500,5));ei=np.concatenate([rng.integers(c*10,(c+1)*10,(500,10)) for c in cells],axis=1)
  vals.extend(x[si[:,:,None],ei[:,None,:]].mean((1,2)))
 return np.quantile(vals,[.025,.975]).tolist()
for ti,task in enumerate(['drawer-open-v3','door-open-v3','button-press-v3']):
 rr=sorted([r for r in rows if r['task']==task],key=lambda r:(r['cell'],r['index']))
 assert len(rr)==60
 raw=[]
 for r in rr:
  c=json.loads((Path(r['folder'])/'case.json').read_text())
  raw.append([c['outcomes'][f'seed{s}/packet/full']['F']['native_return']-c['outcomes'][f'seed{s}/packet/full']['S']['native_return'] for s in range(5)])
 diff=np.array(raw).T;np.testing.assert_array_equal(diff,vectors[task+'__refresh_minus_stale_full'])
 results[task]={}
 for j,(name,cells) in enumerate(proto['subsets'].items()):
  ids=np.concatenate([np.arange(c*10,(c+1)*10) for c in cells])
  results[task][name]=dict(mean=float(diff[:,ids].mean()),ci=boot(diff,cells,2026110564+ti*10+j),n_env=len(ids),per_critic_seed=diff[:,ids].mean(1).tolist(),role='posthoc exploratory, not a new confirmatory test')
 if task!='door-open-v3':continue
 for ch in ['S','F','E']:
  ss=[];rewards=[];first=[];last=[];longest=[]
  for r in rr:
   folder=Path(r['folder']);c=json.loads((folder/'case.json').read_text())
   fenv=[];renv=[]
   for s in range(5):
    with np.load(folder/c['outcomes'][f'seed{s}/packet/full'][ch]['file']) as z:
     flags=z['last_success'].astype(bool);reward=z['reward'];assert flags.shape==reward.shape==(500,)
     assert int(flags.any())==c['outcomes'][f'seed{s}/packet/full'][ch]['success']
     pos=np.flatnonzero(flags)+1
     first.append(float(pos[0]) if len(pos) else np.nan);last.append(float(pos[-1]) if len(pos) else np.nan)
     runs=np.diff(np.flatnonzero(np.diff(np.r_[False,flags,False]))).reshape(-1,2) if False else None
     edges=np.flatnonzero(np.diff(np.r_[0,flags.astype(int),0]));lengths=edges[1::2]-edges[::2];longest.append(int(max(lengths,default=0)))
     fenv.append(flags);renv.append(reward)
   ss.append(fenv);rewards.append(renv)
  ss=np.array(ss);rewards=np.array(rewards);curves[ch+'_success']=ss.mean((0,1));curves[ch+'_reward']=rewards.mean((0,1));curves[ch+'_environment_occupancy']=ss.mean((1,2))
  temporal[ch]=dict(n_env=60,n_critic_seeds=5,any_success=float(ss.any(2).mean()),terminal_success=float(ss[:,:,-1].mean()),mean_success_steps=float(ss.sum(2).mean()),occupancy=float(ss.mean()),conditional_first_mean=float(np.nanmean(first)),conditional_last_mean=float(np.nanmean(last)),mean_longest_run=float(np.mean(longest)),first_range=[float(np.nanmin(first)),float(np.nanmax(first))],last_range=[float(np.nanmin(last)),float(np.nanmax(last))],native_mean_reward=float(rewards.mean()),successful_records=int(ss.any(2).sum()),last_success_step_with_any_record=int(np.flatnonzero(ss.any((0,1)))[-1])+1)
source=R/'.venv-metaworld/lib/python3.12/site-packages/metaworld/envs/sawyer_door_v3.py'
out=dict(nonzero_delay=results,door=temporal,door_native_criterion='abs(obs[4]-target_pos[0])<=0.08; success sets reward10; otherwise shaped grasp/readiness/opening terms; flag loss alone does not identify reclosure',door_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checks='raw contrasts equal independently audited vectors; all cases retained; environment-equal and critic-equal weighting; no independence claim for300records')
(O/'analysis.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'curves.npz',**curves);print(json.dumps(out,indent=2))
