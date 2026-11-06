import json
from pathlib import Path
R=Path('/home/ez/DelayedAdviceAudit/code');P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS/source_data');D=R/'outputs/jaamas/revision_descriptives_20261105_v1';x=json.loads((D/'tolerance_grid.json').read_text());s=[r'\begin{table}[tb]\centering\small',r'\caption{Zero-predictor selections among 100 preassigned 14/6 resplits of the original 20 development environments per task. Columns vary model-selection MSE tie tolerance, not the deployment acceptance threshold. These reuse development outcomes and are not independent test-performance estimates.}\label{tab:tolerance}',r'\begin{tabular}{llrrrrr}\toprule Task & Cell & $0$ & $10^{-14}$ & $10^{-12}$ & $10^{-10}$ & $10^{-8}$\\\midrule']
for task,label in [('push-v3','Push'),('pick-place-v3','PickPlace')]:
 for key,lab in [('cheap_inputs/cheap_actions','$L,C$'),('cheap_inputs/all_actions','$L,F$'),('with_packet/cheap_actions','$P,C$'),('with_packet/all_actions','$P,F$')]:
  a=x[task][key];assert [v['tolerance'] for v in a]==[0,1e-14,1e-12,1e-10,1e-8];s.append(label+' & '+lab+' & '+' & '.join(str(v['null_count']) for v in a)+r'\\')
s.append(r'\bottomrule\end{tabular}\end{table}');(P/'tolerance_table.tex').write_text('\n'.join(s)+'\n')
x=json.loads((D/'confirmation_delay.json').read_text());s=[r'\begin{table}[tb]\centering\small',r'\caption{Descriptive original visual confirmation strata for the full packet policy $P,F$. Mean native reward spans all 500 steps; success endpoints and reject are percentages. Independent environment counts sum to 250 per task. Delay strata use different environment seeds, so differences between rows are not a paired causal effect of delay.}\label{tab:visualdelay}',r'\begin{tabular}{lrrrrrr}\toprule Task & Delay & $n$ & Reward & Any success & Terminal & Reject\\\midrule']
for task,label in [('push-v3','Push'),('pick-place-v3','PickPlace')]:
 for row in x[task]['delay_breakdown']:
  v=row['conditions']['with_packet/all_actions'];s.append(f"{label} & {row['delay']} & {v['n']} & {v['reward']:.3f} & {100*v['success']:.1f} & {100*v['terminal_success']:.1f} & {100*v['reject_rate']:.1f}"+r'\\')
s.append(r'\bottomrule\end{tabular}\end{table}');(P/'visual_delay_table.tex').write_text('\n'.join(s)+'\n');(P/'tolerance_grid.json').write_text((D/'tolerance_grid.json').read_text());(P/'confirmation_delay.json').write_text((D/'confirmation_delay.json').read_text());print('tables exported')
p=P.parent/'sections/appendix.tex';s=p.read_text();a=s.index('\\section{Reproduction and provenance}');s=s[:a]+r'''\subsection{Development tolerance and descriptive delay strata}
\input{source_data/tolerance_table}
\input{source_data/visual_delay_table}
These analyses do not refit on test environments. The full JSON retains every visual policy's reward, both success endpoints and rejection rate at each delay, along with every declared tolerance and resplit cell. The model-selection sensitivity question is distinct from changing the deployment threshold after seeing outcomes.

'''+s[a:];p.write_text(s)
