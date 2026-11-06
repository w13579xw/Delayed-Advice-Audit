import json,collections
from pathlib import Path
import numpy as np
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');A=R/'outputs/jaamas/channel_revision_analysis_20261105_v2';d=json.loads((A/'analysis.json').read_text());labels={'drawer-open-v3':'Drawer','door-open-v3':'Door','button-press-v3':'Button'}
s=[r'\begin{table}[tb]\centering\small',r'\caption{Fixed-policy channel contrasts on 60 fresh environments/task and five previously frozen critic seeds. $c_s$: stale public-source message; $c_f$: refreshed message; $c_\emptyset$: missing message, handled by a frozen reject wrapper. Confidence is approximately 99.5833\% per contrast (12-contrast family). Reward is the mean over the full 500-step trajectory.}\label{tab:channel}',r'\begin{tabular}{llrr}\toprule Task & Fixed-policy contrast & Mean & Interval\\\midrule']
for t,x in d.items():
 for k,lab in [('refresh_minus_stale_cheap',r'$c_f-c_s$, $P,C$'),('refresh_minus_stale_full',r'$c_f-c_s$, $P,F$'),('stale_minus_erasure_full',r'$c_s-c_\emptyset$, $P,F$'),('refresh_minus_erasure_full',r'$c_f-c_\emptyset$, $P,F$')]:
  v=x['primary_contrasts'][k];s.append(f"{labels[t]} & {lab} & {v['mean']:+.6f} & [{v['ci'][0]:+.6f}, {v['ci'][1]:+.6f}]"+r'\\')
s.extend([r'\bottomrule\end{tabular}\end{table}']);(P/'source_data/channel_table.tex').write_text('\n'.join(s)+'\n')
s=[r'\begin{table}[tb]\centering\small',r'\caption{Offline finite-pool diagnosis. $H$: best of all ten options minus reject; $H_E$: positive part of best source option minus reject; $G$: full packet selector regret. Opp.: fraction with $H>10^{-8}$. Hit: source selections beating reject by $10^{-8}$, undefined if none. Opt.: selector matches the pool maximum within $10^{-8}$. Rates are percentages. Headroom and opportunity span 60 environments; regret, hit and optimal-choice rates also span five critic seeds. This is an offline diagnostic, not deployable oracle performance.}\label{tab:headroom}',r'\setlength{\tabcolsep}{3pt}\begin{tabular}{llrrrrrr}\toprule Task & Ch. & $H$ & $H_E$ & $G$ & Opp. & Hit & Opt.\\\midrule']
for t,x in d.items():
 for ch in ['S','F']:
  v=x['oracle'][ch];hit='--' if v['beneficial_source_precision'] is None else f"{100*v['beneficial_source_precision']:.1f}";s.append(f"{labels[t]} & $c_{ch.lower()}$ & {v['full_headroom']:.6f} & {v['expert_headroom']:.6f} & {v['mean_selector_regret']:.6f} & {100*v['full_positive']:.1f} & {hit} & {100*v['optimal_choice_fraction']:.1f}"+r'\\')
s.extend([r'\bottomrule\end{tabular}\end{table}']);(P/'source_data/headroom_table.tex').write_text('\n'.join(s)+'\n')
rows=json.loads((R/'outputs/jaamas/channel_revision_20261105_v1/main/rows.json').read_text());extra={}
for t in labels:
 cases=[json.loads((Path(r['folder'])/'case.json').read_text()) for r in rows if r['task']==t];metrics={}
 for inp,ac in [('local','full'),('packet','cheap'),('packet','full')]:
  for ch in ['S','F','E']:
   vals=[c['outcomes'][f'seed{s}/{inp}/{ac}'][ch] for s in range(5) for c in cases];metrics[f'{inp}/{ac}/{ch}']={m:float(np.mean([v[m] for v in vals])) for m in ['native_return','success','terminal_success']}
 extra[t]=metrics
(R/'outputs/jaamas/revision_20261105/derived_channel_secondary_success.json').write_text(json.dumps(extra,indent=2));(P/'source_data/channel_analysis.json').write_text(json.dumps(d,indent=2));(P/'source_data/channel_secondary_success.json').write_text(json.dumps(extra,indent=2));print(json.dumps(extra,indent=2))
