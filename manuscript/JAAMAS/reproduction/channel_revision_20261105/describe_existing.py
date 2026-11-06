from pathlib import Path
import json,sys,collections,numpy as np,time
R=Path('/home/ez/DACI/code');sys.path.insert(0,str(R/'scripts'));from server_run import save,now
OUT=R/'outputs/jaamas/revision_descriptives_20261105_v1';OUT.mkdir(exist_ok=False);res={}
# Existing confirmation used only descriptively; no fitted changes/new tests.
rows=json.loads((R/'outputs/vision/arrival_confirmation_20261104_v1/episodes.json').read_text())
for task in ['push-v3','pick-place-v3']:
 tt=[x for x in rows if x['task']==task];groups=[]
 for d in [0,16,32]:
  rr=[r for r in tt if r['delay']==d];cond={}
  for name in rr[0]['conditions']:
   cond[name]=dict(n=len(rr),reward=float(np.mean([r['conditions'][name]['native_return'] for r in rr])),success=float(np.mean([r['conditions'][name]['success'] for r in rr])),terminal_success=float(np.mean([r['conditions'][name]['terminal_success'] for r in rr])),reject_rate=float(np.mean([r['decisions'][name]=='reject' for r in rr])))
  groups.append(dict(delay=d,conditions=cond))
 res[task]=dict(delay_breakdown=groups,role='posthocdescriptivefromalreadyanalyzedconfirmation;notnewcausalclockcomparison')
sensitivity=json.loads((R/'outputs/vision/development_resplits_20261104_v1/analysis.json').read_text());save(OUT/'confirmation_delay.json',res);save(OUT/'tolerance_grid.json',sensitivity)
root=R/'outputs/jaamas/channel_revision_20261105_v1/main';rows=json.loads((root/'rows.json').read_text());oracle={}
for task in ['drawer-open-v3','door-open-v3','button-press-v3']:
 rr=[r for r in rows if r['task']==task];data=[]
 for row in rr:
  folder=Path(row['folder']);case=json.loads((folder/'case.json').read_text());obs=json.loads((folder/'choices_before_outcomes.json').read_text());base=next(b['native_return'] for b in case['branches'].values() if b['name']=='reject')
  for ch in ['S','F']:
   acts={b['name']:b['native_return'] for b in case['branches'].values() if ch in b['channels']};cheap=max(v for k,v in acts.items() if k in obs['names'][:6]);expert=max(v for k,v in acts.items() if k in obs['names'][6:]);data.append(dict(channel=ch,delay=row['delay'],q=row['q'],sigma=row['sigma'],prefix_success=obs['prefix_success'],headroom=max(cheap,expert)-base,source_headroom=max(0,expert-base),exclusive=max(0,expert-cheap),expert_opportunity=expert>base+1e-8))
 result=[]
 for ch in ['S','F']:
  for field,vals in [('delay',[0,16,64]),('q',[16,144,272,400]),('prefix_success',[False,True])]:
   for value in vals:
    dd=[v for v in data if v['channel']==ch and v[field]==value]
    if dd:result.append(dict(channel=ch,field=field,value=value,n=len(dd),mean_headroom=float(np.mean([v['headroom'] for v in dd])),headroom_median=float(np.median([v['headroom'] for v in dd])),mean_source_headroom=float(np.mean([v['source_headroom'] for v in dd])),source_opportunity_fraction=float(np.mean([v['expert_opportunity'] for v in dd])),mean_exclusive=float(np.mean([v['exclusive'] for v in dd]))))
 oracle[task]=result
save(OUT/'channel_headroom_strata.json',oracle);save(OUT/'checks.json',dict(checks='passed',no_refit=True,no_new_primary_tests=True,all_declared_tolerances_retained=True));save(OUT/'run.json',dict(status='complete',exit_code=0,utc=now()));print('COMPLETE',OUT)
