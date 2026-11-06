"""Independent saved-input/decision/raw-trajectory audit and prespecified paired inference."""
import json,sys,hashlib,collections
from pathlib import Path
import numpy as np
R=Path('/home/ez/DelayedAdviceAudit/code');sys.path.insert(0,str(R/'src'));from advice_audit.value_kernel import FrozenKernelDecisions
O=R/'outputs/jaamas/visual_channel_bridge_analysis_20261105_v1';O.mkdir(exist_ok=False)
B=R/'outputs/jaamas/visual_channel_bridge_20261105_v1';policy=FrozenKernelDecisions(R/'outputs/vision/value_kernel_pilot_20261104_v3','pick-place-v3')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
all_results={};audit=dict(branches=0,decisions=0,source_packets=0,primary_envs=60,identity_envs=6,smoke_envs=2,maximum_reward_error=0.)
for mode,n in [('smoke',2),('identity',6),('main',60)]:
 root=B/mode;p=json.loads((root/'protocol.json').read_text());assert json.loads((root/'run.json').read_text())['status']=='complete'
 assert sha(root/'entry_snapshot.py')==p['entry_sha256']
 for f,h in p['model_hashes'].items():assert sha(Path(f))==h
 rows=json.loads((root/'rows.json').read_text());assert len(rows)==n and [{k:r[k] for k in ['seed','q','delay','cell','index']} for r in rows]==p['schedule']
 vals=collections.defaultdict(list);choices=collections.defaultdict(list);acts=collections.defaultdict(list)
 for r in rows:
  folder=Path(r['folder']);f=folder/'choices_before_outcomes.json';assert sha(f)==r['choice_sha256'];d=json.loads(f.read_text());assert d['written_before_future']
  with np.load(folder/'observed_inputs.npz') as z:
   request={k[8:]:z[k].copy() for k in z.files if k.startswith('request_')};arrival={k[8:]:z[k].copy() for k in z.files if k.startswith('arrival_')}
   for ch in ['S','F']:
    packet=z['packet_'+ch];assert hashlib.sha256(packet.tobytes()).hexdigest()==d['source_packet_hashes'][ch];audit['source_packets']+=1
    dec=policy.decide(policy.distances(request,arrival,packet,r['q'],r['delay']));assert dec==d['decisions'][ch]
   if r['delay']==0:np.testing.assert_array_equal(z['packet_S'],z['packet_F']);assert d['decisions']['S']==d['decisions']['F'] and d['plan_keys']['S']==d['plan_keys']['F']
  traces={}
  for key,b in r['branches'].items():
   with np.load(folder/b['file']) as z:
    rr=z['reward'];aa=z['action'];ss=z['last_success'];assert rr.shape==(500,) and aa.shape==(500,4) and z['robot4'].shape==(500,4);assert np.isfinite(rr).all() and np.max(np.abs(aa))<=1+1e-12
    assert float(rr.mean())==b['native_return'] and int(ss.any())==b['success'] and int(ss[-1])==b['terminal_success'];traces[key]=aa.copy()
   audit['branches']+=1
  for ch in ['S','F','E']:
   for label,dec in d['decisions'][ch].items():
    if ch=='E':assert dec['choice']=='reject'
    b=r['branches'][d['plan_keys'][ch][label]];assert b==r['outcomes'][ch][label];vals[ch+'/'+label].append([b['native_return'],b['success'],b['terminal_success']]);audit['decisions']+=1
  for label in d['decisions']['S']:
   choices[label].append(d['decisions']['S'][label]['choice']!=d['decisions']['F'][label]['choice']);acts[label].append(bool(np.any(np.abs(traces[d['plan_keys']['S'][label]]-traces[d['plan_keys']['F'][label]])>1e-8)))
  assert d['decisions']['S']['with_packet/cheap_actions']['null_model'] and d['decisions']['S']['with_packet/cheap_actions']['choice']=='reject'
 arrays={k:np.array(v) for k,v in vals.items()};res=dict(n_env=n,condition_means={k:dict(reward=float(v[:,0].mean()),any_success=float(v[:,1].mean()),terminal_success=float(v[:,2].mean())) for k,v in arrays.items()},dependence={k:dict(choices=int(sum(v)),actions=int(sum(acts[k])),denominator=n) for k,v in choices.items()},native_checks=json.loads((root/'checks.json').read_text()))
 if mode=='main':
  rngseed=2026110563;contrasts={}
  for j,(name,c1,c0) in enumerate([('refresh_minus_stale','F','S'),('stale_minus_absence','S','E'),('refresh_minus_absence','F','E')]):
   diff=arrays[c1+'/with_packet/all_actions']-arrays[c0+'/with_packet/all_actions'];rng=np.random.default_rng(rngseed+j);boot=[]
   for _ in range(60):
    ids=np.concatenate([rng.integers(c*10,(c+1)*10,(500,10)) for c in range(6)],axis=1);boot.extend(diff[ids].mean(1))
   quant=np.quantile(boot,[(.05/3)/2,1-(.05/3)/2],axis=0)
   contrasts[name]=dict(reward=float(diff[:,0].mean()),reward_ci=quant[:,0].tolist(),confidence=1-.05/3,any_success_diff=float(diff[:,1].mean()),terminal_success_diff=float(diff[:,2].mean()),role='primaryreward;successdescriptive',per_cell=[float(diff[c*10:(c+1)*10,0].mean()) for c in range(6)])
  res['primary_contrasts']=contrasts
  np.savez_compressed(O/'condition_vectors.npz',**arrays)
 all_results[mode]=res
(O/'analysis.json').write_text(json.dumps(all_results,indent=2));(O/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(all_results['main'],indent=2))
