"""Independent input/choice/index/trajectory audit and frozen four-contrast inference."""
from pathlib import Path
import numpy as np,json,hashlib,sys,collections
R=Path('/home/ez/DelayedAdviceAudit/code');sys.path.insert(0,str(R/'src'));from advice_audit.value_kernel import FrozenKernelDecisions
B=R/'outputs/jaamas/decoder_control_20261106_v1';smoke_only='--smoke-only' in sys.argv;O=R/('outputs/jaamas/decoder_analysis_smoke_20261106_v1' if smoke_only else 'outputs/jaamas/decoder_analysis_20261106_v1');O.mkdir(exist_ok=False);PF='with_packet/all_actions'
policy=FrozenKernelDecisions(R/'outputs/vision/value_kernel_pilot_20261104_v3','pick-place-v3');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
audit=dict(cases=0,choices=0,unique_paths=0,fixed48_index_checks=0,maximum_reward_mean_error=0.,source_packet_hash_checks=0);results={}
conditions=['R','fixed_S_legacy','gate_S_legacy','fixed_F_legacy','gate_F_legacy','fixed_F_time','gate_F_time']
for mode,n in ([('smoke',2),('identity',6)] if smoke_only else [('smoke',2),('identity',6),('main',60)]):
 root=B/mode;p=json.loads((root/'protocol.json').read_text());assert json.loads((root/'run.json').read_text())['status']=='complete'
 for f,h in {**p['model_hashes'],**p['entry_hashes']}.items():assert sha(f)==h
 assert sha(root/'entry_snapshot.py')==p['entry_hashes'][str(R/'outputs/jaamas/decoder_revision_20261106/decoder_native.py')]
 rows=json.loads((root/'rows.json').read_text());assert len(rows)==n
 assert [{k:r[k] for k in ['seed','q','delay','cell','index']} for r in rows]==p['schedule']
 vectors=collections.defaultdict(list);change=[];exposure=[];choiceSF=[]
 for row in rows:
  folder=Path(row['folder']);f=folder/'choices_before_outcomes.json';assert sha(f)==row['choice_sha256'];d=json.loads(f.read_text());assert d['written_before_future'];assert d['generation_steps']=={'S':row['q'],'F':row['arrival']}
  with np.load(folder/'observed_inputs.npz') as z:
   request={k[8:]:z[k].copy() for k in z.files if k.startswith('request_')};arrival={k[8:]:z[k].copy() for k in z.files if k.startswith('arrival_')};packets={ch:z['packet_'+ch].copy() for ch in ['S','F']}
   for ch,packet in packets.items():
    assert hashlib.sha256(packet.tobytes()).hexdigest()==d['source_packet_hashes'][ch];audit['source_packet_hash_checks']+=1
    assert policy.decide(policy.distances(request,arrival,packet,row['q'],row['delay']))==d['decisions'][ch];audit['choices']+=4
  traces={}
  for key,b in row['branches'].items():
   with np.load(folder/b['file']) as z:
    tr={k:z[k].copy() for k in z.files};assert tr['reward'].shape==(500,) and tr['action'].shape==(500,4) and tr['robot4'].shape==(500,4) and tr['last_success'].shape==(500,)
    assert np.isfinite(tr['reward']).all() and min(tr['reward'])>=0 and max(tr['reward'])<=10+1e-12
    assert float(tr['reward'].mean())==b['native_return'] and int(tr['last_success'].any())==b['success'] and int(tr['last_success'][-1])==b['terminal_success']
    assert np.max(np.abs(tr['action']))<=1+1e-12;traces[key]=tr
   audit['unique_paths']+=1
  base=traces[d['plan_mapping']['R']]
  for key,tr in traces.items():
   for item in ['reward','action','last_success','robot4']:np.testing.assert_array_equal(tr[item][:row['arrival']],base[item][:row['arrival']])
  for condition in conditions:
   key=d['plan_mapping'][condition];b=row['branches'][key];assert row['outcomes'][condition]==b
   vectors[condition].append([b['native_return'],b['success'],b['terminal_success']])
   if condition=='R':assert b['name']=='reject';continue
   kind,ch,decoder=condition.split('_');age=row['delay'] if decoder=='legacy' else row['arrival']-d['generation_steps'][ch]
   if b['name'].startswith('act_suffix') or b['name']=='act_blend48':assert d['plans'][key]['effective_age']==age
   if kind=='fixed':
    assert b['name']=='act_suffix48';np.testing.assert_array_equal(traces[key]['action'][row['arrival']:row['arrival']+48],packets[ch][age:age+48]);audit['fixed48_index_checks']+=1
   else:assert b['name']==d['decisions'][ch][PF]['choice']
  kk=d['plan_mapping'];choice=d['decisions']['F'][PF]['choice'];consumes_age=choice.startswith('act_suffix') or choice=='act_blend48';exposure.append(consumes_age)
  actionchange=bool(np.any(np.abs(traces[kk['gate_F_time']]['action']-traces[kk['gate_F_legacy']]['action'])>1e-8));change.append(actionchange)
  if not consumes_age:assert not actionchange and kk['gate_F_time']==kk['gate_F_legacy']
  choiceSF.append(d['decisions']['S'][PF]['choice']!=choice)
  if row['delay']==0:
   np.testing.assert_array_equal(packets['S'],packets['F'])
   for kind in ['fixed','gate']:assert kk[kind+'_S_legacy']==kk[kind+'_F_legacy']==kk[kind+'_F_time']
  audit['cases']+=1
 arr={c:np.array(vectors[c]) for c in conditions};r=dict(n_env=n,condition_means={c:dict(reward=float(x[:,0].mean()),any_success=float(x[:,1].mean()),terminal_success=float(x[:,2].mean())) for c,x in arr.items()},gate_fresh_choice_counts=dict(collections.Counter(json.loads((Path(row['folder'])/'choices_before_outcomes.json').read_text())['decisions']['F'][PF]['choice'] for row in rows)),time_sensitive_source_choices=int(sum(exposure)),decoder_action_changes=int(sum(change)),stale_fresh_gate_choice_changes=int(sum(choiceSF)),native_checks=json.loads((root/'checks.json').read_text()))
 if mode=='main':
  specs=[('fixed_time_minus_fixed_fresh_legacy','fixed_F_time','fixed_F_legacy'),('fixed_time_minus_fixed_stale_legacy','fixed_F_time','fixed_S_legacy'),('gate_time_minus_gate_fresh_legacy','gate_F_time','gate_F_legacy'),('gate_time_minus_gate_stale_legacy','gate_F_time','gate_S_legacy')]
  diffs=np.stack([arr[a][:,0]-arr[b][:,0] for name,a,b in specs],axis=1);rng=np.random.default_rng(2026110601);boot=[]
  secondary_specs=[(c,arr[c]-arr['R']) for c in conditions if c!='R'];secondary=np.stack([v for _,v in secondary_specs],axis=1);secondary_boot=[]
  for _ in range(60):
   ix=np.concatenate([rng.integers(c*10,(c+1)*10,(500,10)) for c in range(6)],axis=1);boot.append(diffs[ix].mean(1));secondary_boot.append(secondary[ix].mean(1))
  ci=np.quantile(np.concatenate(boot),[.00625,.99375],axis=0);sb=np.concatenate(secondary_boot);sci=np.quantile(sb,[.025,.975],axis=0)
  r['primary_contrasts']={name:dict(mean=float(diffs[:,j].mean()),ci=ci[:,j].tolist(),confidence=.9875,n_env=60,per_cell=diffs[:,j].reshape(6,10).mean(1).tolist()) for j,(name,a,b) in enumerate(specs)}
  r['secondary_vs_reject']={c:dict(mean=float(v[:,0].mean()),reward_ci95=sci[:,j,0].tolist(),any_success_difference=float(v[:,1].mean()),terminal_success_difference=float(v[:,2].mean()),role='descriptive;notnewprimaryinference') for j,(c,v) in enumerate(secondary_specs)}
  np.savez_compressed(O/'condition_vectors.npz',**arr)
 results[mode]=r
assert audit['cases']==(8 if smoke_only else 68) and audit['source_packet_hash_checks']==(16 if smoke_only else 136)
(O/'analysis.json').write_text(json.dumps(results,indent=2));(O/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit if smoke_only else results['main'],indent=2))
