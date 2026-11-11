"""Independent trace/prediction audit and predeclared stratified crossed bootstrap."""
import argparse,hashlib,json,sys,time,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'experiments/state')]
from server_run import save,now
from state_core_20261105 import TASKS,Critic,feature_groups

def main():
 p=argparse.ArgumentParser();p.add_argument('--test',required=True);p.add_argument('--models',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args();test=Path(a.test).resolve();models=Path(a.models).resolve();data=Path(a.data).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=json.loads((test/'frozen_before_test.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items());cases=[json.loads(p.read_text()) for p in sorted(test.glob('*/cell*/case.json'))];schedule=json.loads((test/'schedule.json').read_text());assert len(cases)==len(schedule)==360;assert len({x['seed'] for x in cases})==360
 rows=json.loads((data/'rows.json').read_text());assert len(rows)==126
 for r in rows:
  z=np.load(data/r['file']);np.testing.assert_array_equal(z['target'],np.mean(z['reward'],axis=1)-np.mean(z['reward'][0]));assert z['reward'].shape==(10,500);np.testing.assert_array_equal(z['reward'][:,:r['q']+r['delay']],np.broadcast_to(z['prefix'],(10,len(z['prefix']))));c,pf=feature_groups(r['request'],r['arrival'],r['q'],r['delay'],r['sigma'],r['packet']);np.testing.assert_array_equal(c,z['cheap']);np.testing.assert_array_equal(pf,z['packet_features'])
 rng=np.random.default_rng(2026110531);results={};decisions=0;leafcount=0;maxerr=0;vectors={}
 for task in TASKS:
  rr=sorted([r for r in cases if r['task']==task],key=lambda x:(x['cell'],x['index']));assert len(rr)==120 and collections.Counter(x['cell'] for x in rr)=={i:10 for i in range(12)};critics={f'seed{s}/{i}/{ac}':Critic(models/task/f'seed{s}'/i/ac) for s in range(5) for i in ['local','packet'] for ac in ['cheap','full']};means={};success={};choicecounts=collections.defaultdict(collections.Counter);sensitivity={k:dict(total=0,zero=0,negated=0,reversed=0) for k in critics if '/packet/' in k}
  for r in rr:
   folder=test/task/f"cell{r['cell']:02d}_ep{r['index']:02d}";loaded={}
   for leaf in sorted({x['leaf'] for x in r['results'].values()}):
    z=np.load(folder/f'leaf{leaf:03d}.npz');assert z['reward'].shape==(500,) and z['action'].shape==(500,4) and np.isfinite(z['reward']).all() and np.isfinite(z['action']).all() and np.max(np.abs(z['action']))<=1+1e-12;loaded[leaf]=z['reward'].mean();leafcount+=1
   for label,res in r['results'].items():
    maxerr=max(maxerr,abs(res['native_return']-loaded[res['leaf']]));means.setdefault(label,[]).append(loaded[res['leaf']]);success.setdefault(label,[]).append(res['success']);choicecounts[label].update(x['choice'] for x in res['decisions'])
    for dec in res['decisions']:
     event=r['events'][dec['event']];c,pf=feature_groups(event['request'],event['observation'],event['q'],r['delay'],r['sigma'],event['packet']);np.testing.assert_array_equal(c,event['cheap']);np.testing.assert_array_equal(pf,event['packet_features']);assert event['arrival']==event['q']+r['delay'];assert event['arrival']+32<=500
     if label in critics:
      cc=critics[label];assert cc.decide(pf if '/packet/' in label else c)==dec['choice'];decisions+=1
      if label in sensitivity:
       v=sensitivity[label];v['total']+=1
       for name,pp in [('zero',np.zeros(4)),('negated',-np.array(event['packet'])),('reversed',np.array(event['packet'])[::-1])]:v[name]+=int(cc.decide(np.r_[c,pp])!=dec['choice'])
   # All reference paths have the same exogenous noise and actual horizon.
  assert maxerr==0;arrays={k:np.asarray(v) for k,v in means.items()};contrasts={}
  def mat(inp,ac):return np.stack([arrays[f'seed{s}/{inp}/{ac}'] for s in range(5)])
  for name,diff in [('packet_cheap_minus_local_cheap',mat('packet','cheap')-mat('local','cheap')),('packet_full_minus_packet_cheap',mat('packet','full')-mat('packet','cheap')),('packet_full_minus_reject',mat('packet','full')-arrays['reject'][None,:])]:
   boot=[]
   for b in range(40):
    ns=rng.integers(0,5,(500,5));inds=np.concatenate([rng.integers(c*10,(c+1)*10,(500,10)) for c in range(12)],axis=1);vals=diff[ns[:,:,None],inds[:,None,:]].mean((1,2));boot.extend(vals)
   ci=np.quantile(boot,[.05/9/2,1-.05/9/2]);contrasts[name]=dict(mean=float(diff.mean()),ci=ci.tolist(),confidence=1-.05/9,per_training_seed=diff.mean(1).tolist(),positive_lower=bool(ci[0]>0));vectors[task+'__'+name]=diff
  breakdown=[]
  for cell in range(12):
   idx=np.arange(cell*10,(cell+1)*10);case=rr[idx[0]];breakdown.append(dict(cell=cell,delay=case['delay'],frequency=case['frequency'],sigma=case['sigma'],packet_full_minus_reject=float((mat('packet','full')[:,idx]-arrays['reject'][idx]).mean()),packet_cheap_minus_local_cheap=float((mat('packet','cheap')[:,idx]-mat('local','cheap')[:,idx]).mean()),packet_full_minus_packet_cheap=float((mat('packet','full')[:,idx]-mat('packet','cheap')[:,idx]).mean())))
  results[task]=dict(primary_contrasts=contrasts,condition_means={k:dict(reward=float(v.mean()),any_success=float(np.mean(success[k]))) for k,v in arrays.items()},choice_counts={k:dict(v) for k,v in choicecounts.items()},finite_packet_sensitivity=sensitivity,global_always_reject_certificates=[k for k,c in critics.items() if c.config['model']=='zero'],descriptive_condition_breakdown=breakdown)
 save(out/'analysis.json',results);np.savez_compressed(out/'primary_vectors.npz',**vectors);save(out/'checks.json',dict(checks='passed',cases=len(cases),development_groups=len(rows),independent_development_target_feature_checks=True,choices_replayed=decisions,unique_leaves_checked=leafcount,reward_reconstruction_max_error=maxerr,model_hashes=True));save(out/'run.json',dict(status='complete',exit_code=0,completed_utc=now(),seconds=time.monotonic()-start))
if __name__=='__main__':main()
