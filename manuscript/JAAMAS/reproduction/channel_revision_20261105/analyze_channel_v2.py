"""Independent channel trace audit and predeclared crossed inference/headroom."""
import json,hashlib,sys,time,collections
from pathlib import Path
import numpy as np
R=Path('/home/ez/DelayedAdviceAudit/code');sys.path[:0]=[str(R/'scripts'),str(R/'src'),str(R/'outputs/jaamas')]
from server_run import save,now
from state_core_20261105 import TASKS,OBS_IDS,Locals,Teacher,Critic,feature_groups,noise_stream
ROOT=R/'outputs/jaamas/channel_revision_20261105_v1/main';OUT=R/'outputs/jaamas/channel_revision_analysis_20261105_v2';OUT.mkdir(exist_ok=False);start=time.monotonic();proto=json.loads((ROOT/'protocol.json').read_text());rows=json.loads((ROOT/'rows.json').read_text());assert len(rows)==180
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in proto['model_hashes'].items());rng=np.random.default_rng(2026110553);results={};audit=dict(branches=0,choices=0,delay0_case_checks=0,source_provenance_checks=0,maximum_reward_error=0.,native_reward_min=100.,native_reward_max=-100.);vectors={}
def expanded(x):
 a=np.zeros(39);a[OBS_IDS]=x;return a

def boot(diff,confidence,taskseed):
 brng=np.random.default_rng(taskseed);vals=[];n=diff.shape[1];assert n==60
 for b in range(60):
  si=brng.integers(0,5,(500,5));ei=np.concatenate([brng.integers(c*10,(c+1)*10,(500,10)) for c in range(6)],axis=1);vals.extend(diff[si[:,:,None],ei[:,None,:]].mean((1,2)))
 return np.quantile(vals,[(1-confidence)/2,1-(1-confidence)/2]).tolist()
for ti,task in enumerate(TASKS):
 teacher=Teacher(task);local=Locals(R/'outputs/jaamas/state_qualification_20261105_v1'/task);models=R/'outputs/jaamas/state_campaign_20261105_v1/models'/task;critics={f'seed{s}/{inp}/{act}':Critic(models/f'seed{s}'/inp/act) for s in range(5) for inp in ['local','packet'] for act in ['cheap','full']};cases=[];values={};success={};changes={};head=[];classification=collections.defaultdict(list)
 for row in sorted([r for r in rows if r['task']==task],key=lambda x:(x['cell'],x['index'])):
  folder=Path(row['folder']);c=json.loads((folder/'case.json').read_text());f=folder/'choices_before_outcomes.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==c['choices_sha256'];d=json.loads(f.read_text());assert d['written_before_future'];noise=noise_stream(c['seed']+1000000,c['sigma']);np.testing.assert_array_equal(teacher.action(expanded(d['request']['state10'])),d['packets']['S']);np.testing.assert_array_equal(teacher.action(expanded(d['arrival_observation']['state10'])),d['packets']['F']);audit['source_provenance_checks']+=2
  for x in [d['request'],d['arrival_observation']]:np.testing.assert_array_equal(local.all(expanded(x['state10'])),x['bc'])
  features={}
  for ch in ['S','F']:features[ch]=feature_groups(d['request'],d['arrival_observation'],c['q'],c['delay'],c['sigma'],d['packets'][ch])
  traces={};means={}
  for key,b in c['branches'].items():
   with np.load(folder/b['file']) as z:
    rr=z['reward'];aa=z['action'];ss=z['last_success'];states=z['state10'];assert rr.shape==(500,) and aa.shape==(500,4) and states.shape==(500,10);assert np.max(np.abs(aa))<=1+1e-12 and np.isfinite(rr).all();assert bool(ss.any())==bool(b['success']) and int(ss[-1])==b['terminal_success'];error=abs(float(rr.mean())-b['native_return']);assert error==0;traces[key]=aa.copy();means[key]=float(rr.mean());audit['native_reward_min']=min(audit['native_reward_min'],float(rr.min()));audit['native_reward_max']=max(audit['native_reward_max'],float(rr.max()));audit['branches']+=1
  basekey=next(k for k,b in c['branches'].items() if b['name']=='reject');base=means[basekey];names=d['names'];bych={ch:{name:next(k for k,b in c['branches'].items() if b['name']==name and ch in b['channels']) for name in names} for ch in ['S','F']}
  for label,cc in critics.items():
   for ch in ['S','F','E']:
    dec=d['decisions'][label][ch];key=dec['key'];assert key in c['branches'];value=c['outcomes'][label][ch];assert value['native_return']==means[key];values.setdefault((label,ch),[]).append(means[key]);success.setdefault((label,ch),[]).append(value['success'])
    if ch!='E':
     xx=features[ch][1 if cc.config['input']=='packet_features' else 0];assert cc.decide(xx)==dec['choice'];np.testing.assert_allclose(cc.scores(xx),dec['scores'],atol=0,rtol=0)
    else:assert dec['choice']=='reject'
    audit['choices']+=1
   ds=d['decisions'][label];choicechange=ds['S']['choice']!=ds['F']['choice'];actionchange=bool(np.any(np.abs(traces[ds['S']['key']]-traces[ds['F']['key']])>1e-8));changes.setdefault(label,[]).append(dict(choice_changed=choicechange,action_changed=actionchange,delay=c['delay']))
   if c['delay']==0:
    assert ds['S']['key']==ds['F']['key'] and not actionchange and not choicechange
  if c['delay']==0:np.testing.assert_array_equal(d['packets']['S'],d['packets']['F']);audit['delay0_case_checks']+=1
  for ch in ['S','F']:
   vals=np.array([means[bych[ch][n]] for n in names]);gain=vals-base;cheap=vals[:6].max();expert=vals[6:].max();best=vals.max();h=dict(channel=ch,delay=c['delay'],sigma=c['sigma'],q=c['q'],cheap_headroom=float(cheap-base),expert_headroom=float(max(0.,expert-base)),exclusive_headroom=float(max(0.,expert-cheap)),full_headroom=float(best-base),expert_positive=bool(expert-base>1e-8),expert_scale_opportunity=bool(expert-base>.25),full_positive=bool(best-base>1e-8))
   for seed in range(5):
    lab=f'seed{seed}/packet/full';dec=d['decisions'][lab][ch];chosen=names.index(dec['choice']);selected=vals[chosen];regret=best-selected;assert regret>=-1e-10 and abs((selected-base)-((best-base)-regret))<1e-12;source=chosen>=6;hit=bool(source and selected-base>1e-8);optimal=bool(abs(best-selected)<=1e-8);pred=np.array(dec['scores']);sourcescore=float(pred[6:].max());reason='no_expert_opportunity' if not h['expert_positive'] else ('beneficial_source_selected' if hit else ('rejected_nonpositive_source_prediction' if chosen==0 and sourcescore<=1e-8 else ('cheap_action_selected' if chosen<6 else 'nonbeneficial_source_selected')));classification[f'{ch}/seed{seed}'].append(dict(regret=float(regret),gain=float(selected-base),expert_opportunity=h['expert_positive'],source_selected=source,beneficial_source_selected=hit,optimal=optimal,reason=reason,source_prediction_mse=float(np.mean((pred[6:]-gain[6:])**2)),choice_class='source' if source else ('reject' if chosen==0 else 'local')))
   head.append(h)
  cases.append(c)
 arr={k:np.array(v) for k,v in values.items()}
 def mat(inp,ac,ch,metric=arr):return np.stack([metric[(f'seed{s}/{inp}/{ac}',ch)] for s in range(5)])
 specs=[('refresh_minus_stale_cheap',mat('packet','cheap','F')-mat('packet','cheap','S')),('refresh_minus_stale_full',mat('packet','full','F')-mat('packet','full','S')),('stale_minus_erasure_full',mat('packet','full','S')-mat('packet','full','E')),('refresh_minus_erasure_full',mat('packet','full','F')-mat('packet','full','E'))];contrasts={}
 succ={k:np.array(v) for k,v in success.items()}
 for j,(name,diff) in enumerate(specs):
  ci=boot(diff,1-.05/12,2026110553+ti*10+j);vectors[task+'__'+name]=diff
  dd=mat('packet','cheap','F',succ)-mat('packet','cheap','S',succ) if j==0 else (mat('packet','full','F',succ)-mat('packet','full','S',succ) if j==1 else mat('packet','full','S' if j==2 else 'F',succ)-mat('packet','full','E',succ))
  contrasts[name]=dict(mean=float(diff.mean()),ci=ci,confidence=1-.05/12,per_critic_seed=diff.mean(1).tolist(),native_success_difference=float(dd.mean()),success_ci95_descriptive=boot(dd,.95,2026110653+ti*10+j))
 breakdown=[]
 for cell in range(6):
  idx=np.arange(cell*10,(cell+1)*10);breakdown.append(dict(cell=cell,delay=cases[idx[0]]['delay'],sigma=cases[idx[0]]['sigma'],**{k:float(v[:,idx].mean()) for k,v in specs}))
 oracle={}
 for ch in ['S','F']:
  hh=[h for h in head if h['channel']==ch];sourcepoint={k:float(np.mean([h[k] for h in hh])) for k in ['cheap_headroom','expert_headroom','exclusive_headroom','full_headroom','expert_positive','expert_scale_opportunity','full_positive']};ss=[v for seed in range(5) for v in classification[f'{ch}/seed{seed}']];nsource=sum(v['source_selected'] for v in ss);nopp=sum(v['expert_opportunity'] for v in ss);nhit=sum(v['beneficial_source_selected'] for v in ss);sourcepoint.update(mean_selector_regret=float(np.mean([v['regret'] for v in ss])),beneficial_source_precision=nhit/nsource if nsource else None,beneficial_source_recall=nhit/nopp if nopp else None,optimal_choice_fraction=float(np.mean([v['optimal'] for v in ss])),source_prediction_mse=float(np.mean([v['source_prediction_mse'] for v in ss])),reasons=dict(collections.Counter(v['reason'] for v in ss)),gain_contributions={cl:float(np.mean([v['gain'] if v['choice_class']==cl else 0 for v in ss])) for cl in ['reject','local','source']});oracle[ch]=sourcepoint
 change_summary={lab:dict(choices=sum(x['choice_changed'] for x in xs),actions=sum(x['action_changed'] for x in xs),denominator=len(xs),by_delay={str(d):dict(choices=sum(x['choice_changed'] for x in xs if x['delay']==d),actions=sum(x['action_changed'] for x in xs if x['delay']==d),n=sum(x['delay']==d for x in xs)) for d in [0,16,64]}) for lab,xs in changes.items()}
 results[task]=dict(primary_contrasts=contrasts,condition_means={f'{lab}/{ch}':dict(reward=float(v.mean()),success=float(np.mean(success[(lab,ch)]))) for (lab,ch),v in arr.items()},dependence=change_summary,oracle=oracle,descriptive_breakdown=breakdown,channel_boundary='F uses real source atarrival butjointinput differs from S-onlytraining;valueofthisdefinedrefreshchannel,notintrinsicprivateinfo');save(OUT/'analysis.json',results);save(OUT/(task+'_headroom_cases.json'),head);save(OUT/(task+'_selector_diagnostics.json'),classification)
assert len({x['seed'] for x in rows})==180;assert audit['native_reward_min']>=0 and audit['native_reward_max']<=10;assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in proto['model_hashes'].items());save(OUT/'audit.json',audit);np.savez_compressed(OUT/'primary_vectors.npz',**vectors);save(OUT/'checks.json',dict(checks='passed',source_provenance_checked=True,all_models_unchanged=True,choices_recomputed=audit['choices'],oracle_regret_identity=True,raw_reward_reconstruction_error=0));save(OUT/'run.json',dict(status='complete',exit_code=0,seconds=time.monotonic()-start,utc=now()));print(json.dumps({t:x['primary_contrasts'] for t,x in results.items()},indent=2))
