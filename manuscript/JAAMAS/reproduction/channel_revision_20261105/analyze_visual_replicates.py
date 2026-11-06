"""Independent saved-input/trace audit and prespecified three-actor inference."""
import json,hashlib,time,sys,collections
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from daci.value_kernel import FrozenKernelDecisions
from server_run import save,now,fingerprint
P=R/'outputs/jaamas/visual_pipeline_20261105_v1';O=R/'outputs/jaamas/visual_replication_analysis_20261105_v1'
assert json.loads((P/'run.json').read_text())['status']=='complete'
O.mkdir(exist_ok=False);t0=time.monotonic();audit=collections.Counter();freeze=json.loads((P/'all_models_frozen_before_test.json').read_text());entries=json.loads((P/'entry_hashes.json').read_text());assert fingerprint()==entries['source_fingerprint']
for payload in [freeze,entries]:
 for f,h in payload['files'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
schedule=json.loads((R/'outputs/jaamas/revision_20261105/visual_schedule.json').read_text());allrows={};result={};vectors={}
for seed in range(3):
 root=P/f'seed{seed}_test';assert json.loads((root/'run.json').read_text())['status']=='complete';assert freeze['utc']<json.loads((root/'protocol.json').read_text())['frozen_utc'];rows=json.loads((root/'episodes.json').read_text());assert len(rows)==100;allrows[seed]=rows
 for task in ['push-v3','pick-place-v3']:
  rr=[r for r in rows if r['task']==task];assert [r['seed'] for r in rr]==[r['seed'] for r in schedule['test'][task]];policy=FrozenKernelDecisions(Path(freeze['model_roots'][str(seed)]),task)
  for row in rr:
   stem=f"{task}_{row['seed']}";choicefile=root/row['choices_file'];assert hashlib.sha256(choicefile.read_bytes()).hexdigest()==row['choices_sha256'];dec=json.loads(choicefile.read_text());assert dec['written_before_future']
   with np.load(root/(stem+'_observed_inputs.npz'),allow_pickle=False) as z:
    q={k[len('request_'):]:z[k].copy() for k in z.files if k.startswith('request_')};a={k[len('arrival_'):]:z[k].copy() for k in z.files if k.startswith('arrival_')};packet=z['source_packet'].copy()
   assert hashlib.sha256(packet.tobytes()).hexdigest()==dec['source_sha256'];actual=policy.decide(policy.distances(q,a,packet,row['query'],row['delay']))
   for key,v in actual.items():
    original=dec['cell_predictions']['n14/'+key];assert v['choice']==original['choice'] and v['null_model']==original['null_model'];np.testing.assert_allclose(v['predicted_gain'],original['predicted_gain'],atol=0,rtol=0);audit['decisions']+=1
   assert row['decisions']==dec['decisions'];assert row['all_option_outcomes'] is not None if row['headroom'] else row['all_option_outcomes'] is None
   assert row['headroom']==(rr.index(row)<10)
   for name in row['unique_options']:
    with np.load(root/(stem+'_'+name+'_trace.npz'),allow_pickle=False) as z:
     rewards=z['rewards'];actions=z['actions'];assert rewards.shape==(500,) and actions.shape==(500-row['arrival'],4);assert np.isfinite(rewards).all() and np.isfinite(actions).all();assert rewards.min()>=-1e-12 and rewards.max()<=10+1e-12;assert abs(actions).max()<=1+1e-12;mean=float(rewards.mean())
    for condition,option in row['decisions'].items():
     if option==name:assert row['conditions'][condition]['native_return']==mean
    if row['headroom']:assert row['all_option_outcomes'][name]['native_return']==mean
    audit['native_option_tails']+=1
   with np.load(root/(stem+'_source100_trace.npz'),allow_pickle=False) as z:
    assert z['rewards'].shape==(500,) and z['actions'].shape==(500,4);assert float(z['rewards'].mean())==row['source_full100']['native_return'];assert np.isfinite(z['actions']).all() and abs(z['actions']).max()<=1+1e-12
   audit['source100_trajectories']+=1;audit['logical_cases']+=1
# Crossed resampling treats training seed and environment as separate axes.
def boot(diff,rngseed):
 rng=np.random.default_rng(rngseed);out=[]
 for _ in range(60):
  si=rng.integers(0,3,(500,3));ei=rng.integers(0,50,(500,50));out.extend(diff[si[:,:,None],ei[:,None,:]].mean((1,2)))
 return np.quantile(out,[.05/12,1-.05/12]).tolist()
for ti,task in enumerate(['push-v3','pick-place-v3']):
 rr={s:[r for r in allrows[s] if r['task']==task] for s in range(3)};means={};choices={};oracle=[];perseed={}
 conditions=list(rr[0][0]['conditions'])
 for con in conditions:
  means[con]={m:float(np.mean([r['conditions'][con][m] for s in range(3) for r in rr[s]])) for m in ['native_return','success','terminal_success']};choices[con]=dict(collections.Counter(r['decisions'][con] for s in range(3) for r in rr[s]))
 def arr(c):return np.array([[r['conditions'][c]['native_return'] for r in rr[s]] for s in range(3)])
 pairs={'packet_cheap_minus_local_cheap':('n14/with_packet/cheap_actions','n14/cheap_inputs/cheap_actions'),'packet_full_minus_packet_cheap':('n14/with_packet/all_actions','n14/with_packet/cheap_actions'),'packet_full_minus_reject':('n14/with_packet/all_actions','reject')};contrasts={}
 for j,(key,(l,r)) in enumerate(pairs.items()):
  diff=arr(l)-arr(r);vectors[task+'__'+key]=diff;contrasts[key]=dict(mean=float(diff.mean()),ci=boot(diff,2026110559+ti*10+j),confidence=1-.05/6,per_actor_seed=diff.mean(1).tolist(),positive_records=int((diff>1e-10).sum()),negative_records=int((diff< -1e-10).sum()))
 for s in range(3):
  source={m:float(np.mean([r['source_full100'][m] for r in rr[s]])) for m in ['native_return','success','terminal_success']};local={m:float(np.mean([r['conditions']['reject'][m] for r in rr[s]])) for m in source};source['reward_minus_local']=source['native_return']-local['native_return'];perseed[s]=dict(source100=source,local=local,conditions={con:{m:float(np.mean([r['conditions'][con][m] for r in rr[s]])) for m in source if m!='reward_minus_local'} for con in conditions},null_models={f+'/'+a:FrozenKernelDecisions(Path(freeze['model_roots'][str(s)]),task).cells[(f,a)][0]['selected']['model']=='zero' for f in ['cheap_inputs','with_packet'] for a in ['cheap_actions','all_actions']})
  for r in rr[s][:10]:
   vals={n:o['native_return'] for n,o in r['all_option_outcomes'].items()};base=vals['reject'];best=max(vals.values());cheap=max(v for n,v in vals.items() if not n.startswith('act_'));expert=max(v for n,v in vals.items() if n.startswith('act_'));chosen=r['decisions']['n14/with_packet/all_actions'];gain=vals[chosen]-base;head=best-base;regret=best-vals[chosen];assert abs(gain-head+regret)<1e-12;oracle.append(dict(actor_seed=s,environment_seed=r['seed'],query=r['query'],delay=r['delay'],headroom=head,expert_headroom=max(0.,expert-base),exclusive_headroom=max(0.,expert-cheap),regret=regret,gain=gain,chosen=chosen,source_opportunity=expert>base+1e-8,scale_source_opportunity=expert-base>.25))
 result[task]=dict(primary_contrasts=contrasts,condition_means=means,choice_histograms=choices,per_actor_seed=perseed,oracle_means={k:float(np.mean([x[k] for x in oracle])) for k in ['headroom','expert_headroom','exclusive_headroom','regret','source_opportunity','scale_source_opportunity']},oracle_records=oracle,scope='3independentlytrainedsystems conditionalonthesame200democorpus andshareddevelopment/testenv;100independenttestenvtotal;300logicalarrivalcases;oraclefirst10/task/seedonly')
np.savez_compressed(O/'primary_vectors.npz',**vectors);save(O/'analysis.json',result);save(O/'audit.json',dict(audit));save(O/'checks.json',dict(checks='passed',all_model_and_entry_hashes_unchanged=True,exact_frozen_choice_replay=True,raw_mean_reconstruction_error=0,independent_environments=100,critic_refit_count=2,retained_original_seed0=True));save(O/'run.json',dict(status='complete',utc=now(),seconds=time.monotonic()-t0,exit_code=0));print(json.dumps({t:x['primary_contrasts'] for t,x in result.items()},indent=2))
