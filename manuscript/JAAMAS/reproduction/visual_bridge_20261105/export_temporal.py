from pathlib import Path
import json,shutil
P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS');R=Path('/home/ez/DelayedAdviceAudit/code');A=R/'outputs/jaamas/state_temporal_20261105_v1/analysis.json';a=json.loads(A.read_text());shutil.copy2(A,P/'source_data/state_temporal_analysis.json')
lines=[r'\begin{table}[tb]\centering\small',r'\caption{Post-hoc state $P,F$ refresh-minus-stale reward differences with exploratory 95\% crossed-bootstrap intervals. Pooled nonzero delay uses 40 environments/task; individual delays use 20. All original noise cells and five critics are retained. These are dependent reanalyses, not new cohorts or confirmatory claims.}\label{tab:nonzero}',r'\begin{tabular}{llrr}\toprule Task & Delay & Mean & Interval\\\midrule']
for task,title in [('drawer-open-v3','Drawer'),('door-open-v3','Door'),('button-press-v3','Button')]:
 for key,label in [('nonzero','16 and 64'),('delay16','16'),('delay64','64')]:
  x=a['nonzero_delay'][task][key];lo,hi=x['ci'];lines.append(f"{title} & {label} & {x['mean']:+.4f} & [{lo:+.4f}, {hi:+.4f}]"+r'\\')
lines.extend([r'\bottomrule\end{tabular}\end{table}']);(P/'source_data/nonzero_delay_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[tb]\centering\small',r'\caption{Door success duration over all 500 native steps. Success-step and longest-run means include every trajectory, including failures; first/last means condition on ever-success. Entries average the 60 environments and five critics.}\label{tab:door-duration}',r'\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrrr}\toprule Channel & Success steps & Occupancy (\%) & First & Last & Longest run\\\midrule']
for ch,title in [('S',r'$c_s$'),('F',r'$c_f$'),('E',r'$c_\emptyset$')]:
 x=a['door'][ch];lines.append(f"{title} & {x['mean_success_steps']:.2f} & {100*x['occupancy']:.2f} & {x['conditional_first_mean']:.2f} & {x['conditional_last_mean']:.2f} & {x['mean_longest_run']:.2f}"+r'\\')
lines.extend([r'\bottomrule\end{tabular}\end{table}']);(P/'source_data/door_duration_table.tex').write_text('\n'.join(lines)+'\n')
# Retain complete former main result tables in SI before compact replacements.
sup=P/'supplement/tables'
for stem in ['channel_table','headroom_table','visual_oracle_table','state_table']:
 txt=(P/'source_data'/f'{stem}.tex').read_text()
 import re
 txt=re.sub(r'\\label\{([^}]+)\}',lambda m:r'\label{s-'+m[1]+'}',txt)
 txt=re.sub(r'\\ref\{([^}]+)\}',lambda m:r'\ref{main-'+m[1]+'}',txt)
 (sup/f'{stem}.tex').write_text(txt)
with (P/'supplement/bridge_details.tex').open('a') as f:
 f.write('\n'+r'\subsection{Complete state-channel and headroom contrasts}'+'\n')
 for stem in ['channel_table','headroom_table','visual_oracle_table','state_table']:f.write(r'\input{tables/'+stem+'}\n')
 f.write('All original state-channel primary contrasts, including the local-action refresh diagnostic, and both channel headroom tables are preserved here. The compact main article retains the full packet contrasts and the cross-regime opportunity comparison.\n')
