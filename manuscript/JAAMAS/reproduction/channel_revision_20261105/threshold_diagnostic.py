import json,collections
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');OUT=R/'outputs/jaamas/revision_descriptives_20261105_v1';rows=json.loads((R/'outputs/jaamas/channel_revision_20261105_v1/main/rows.json').read_text());result={}
for task in ['drawer-open-v3','door-open-v3','button-press-v3']:
 for ch in ['S','F']:
  cts=collections.Counter();values=[]
  for row in [r for r in rows if r['task']==task]:
   f=Path(row['folder']);case=json.loads((f/'case.json').read_text());d=json.loads((f/'choices_before_outcomes.json').read_text());names=d['names'];uu={b['name']:b['native_return'] for b in case['branches'].values() if ch in b['channels']};expert=max(uu[n] for n in names[6:]);cheap=max(uu[n] for n in names[:6]);base=uu['reject']
   for seed in range(5):
    dec=d['decisions'][f'seed{seed}/packet/full'][ch];scores=np.array(dec['scores']);best=max(scores[1:]);source_score=max(scores[6:]);chosen=dec['choice'];cts['decisions']+=1
    if chosen=='reject' and 0<best<=1e-8:cts['reject_changed_by_acceptance_threshold_0_vs_1e-8']+=1
    if expert>base+1e-8:
     cts['positive_source_opportunities_repeated_over_seeds']+=1
     if chosen=='reject':cts['reject_with_nonpositive_source_prediction' if source_score<=0 else 'reject_with_small_positive_source_prediction']+=1
    if expert>cheap+1e-8:
     cts['source_exclusive_opportunities_repeated_over_seeds']+=1
     if chosen in names[:6]:cts['local_or_reject_despite_better_source']+=1
    if chosen in names[6:] and uu[chosen]<base-1e-8:cts['harmful_source_selection']+=1
    values.append(float(best))
  result[task+'/'+ch]=dict(counts=dict(cts),max_predicted_gain_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist(),role='posthocfixedscorediagnostic;no thresholdretuning,nooptimalnewpolicyclaim')
(OUT/'threshold_diagnostic.json').write_text(json.dumps(result,indent=2));print(result)
