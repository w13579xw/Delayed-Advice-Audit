from pathlib import Path
import json,re,shutil,hashlib
R=Path('/home/ez/DelayedAdviceAudit/code');P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS');A=R/'outputs/jaamas/visual_channel_bridge_analysis_20261105_v1';b=json.loads((A/'analysis.json').read_text());v=b['main'];s=json.loads((P/'source_data/channel_analysis.json').read_text())
manifest=json.loads((P/'source_data/manifest.json').read_text())
copies={'visual_channel_analysis.json':A/'analysis.json','visual_channel_audit.json':A/'audit.json','visual_channel_source_audit.json':R/'outputs/jaamas/visual_channel_source_audit_20261105_v1/checks.json','state_temporal_analysis.json':R/'outputs/jaamas/state_temporal_20261105_v1/analysis.json','door_curves.npz':R/'outputs/jaamas/state_temporal_20261105_v1/curves.npz','visual_channel_vectors.npz':A/'condition_vectors.npz'}
for name,f in copies.items():
 shutil.copy2(f,P/'source_data'/name);manifest=[x for x in manifest if x['file']!=name];manifest.append(dict(file=name,source=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(P/'source_data/manifest.json').write_text(json.dumps(manifest,indent=2))
lines=[r'\begin{table}[tb]\centering\small',r'\caption{Fixed full packet receiver: primary native reward channel effects. PickPlace uses one frozen visual system and 60 nonzero-delay environments (98.3333\% intervals, family of three). State rows use 60 environments/task including zero delay and five frozen critics (99.5833\%, original family of 12). $S/F/E$ denote stale/fresh/absent; absence invokes the fixed reject wrapper. These distinct regimes are not pooled. Button zeros are structural.}\label{tab:channel}',r'\setlength{\tabcolsep}{4pt}\begin{tabular}{llrr}\toprule Task / observation & Contrast & Mean & Interval\\\midrule']
for key,c in v['primary_contrasts'].items():
 label={'refresh_minus_stale':'F-S','stale_minus_absence':'S-E','refresh_minus_absence':'F-E'}[key];lo,hi=c['reward_ci'];lines.append("PickPlace / RGB & $"+label+f"$ & {c['reward']:+.4f} & [{lo:+.4f}, {hi:+.4f}]"+r'\\')
for task,title in [('drawer-open-v3','Drawer'),('door-open-v3','Door'),('button-press-v3','Button')]:
 for key,label in [('refresh_minus_stale_full','F-S'),('stale_minus_erasure_full','S-E'),('refresh_minus_erasure_full','F-E')]:
  c=s[task]['primary_contrasts'][key];lo,hi=c['ci'];lines.append(title+" / state & $"+label+f"$ & {c['mean']:+.4f} & [{lo:+.4f}, {hi:+.4f}]"+r'\\')
lines.append(r'\bottomrule\end{tabular}\end{table}');(P/'source_data/channel_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[tb]\centering\small',r'\caption{Frozen original PickPlace visual system on 60 nonzero-delay environments. Native success endpoints are descriptive. All four pre-existing receivers and every channel are retained; no model is refitted.}\label{s-tab:visualchannelsall}',r'\begin{tabular}{llrrr}\toprule Receiver & Channel & Mean reward & Any (\%) & Terminal (\%)\\\midrule']
names={'cheap_inputs/cheap_actions':r'$L,C$','cheap_inputs/all_actions':r'$L,F$','with_packet/cheap_actions':r'$P,C$','with_packet/all_actions':r'$P,F$'}
for label,title in names.items():
 for ch in ['S','F','E']:
  c=v['condition_means'][ch+'/'+label];lines.append(f"{title} & {ch} & {c['reward']:.6f} & {100*c['any_success']:.1f} & {100*c['terminal_success']:.1f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}\end{table}');(P/'supplement/tables/visual_channels_all.tex').write_text('\n'.join(lines)+'\n')
with (P/'supplement/bridge_details.tex').open('a') as h:
 h.write('\n'+r'\subsection{Visual channel bridge: complete outcomes}'+'\n'+r'\input{tables/visual_channels_all}'+'\n')
 for label,c in v['dependence'].items():h.write(names[label]+f": fresh/stale changes {c['choices']}/60 option names and {c['actions']}/60 full action sequences. "+'\n')
 h.write('The visual identity cohort has six excluded zero-delay cases, plus two excluded engineering cases. The primary 60 environments cross two nonzero delays and three request times. Per-cell vectors, model hashes and independent source-regeneration checks are in the accompanying source data. The literal refreshed packet retains the original delay-indexed decoder; its contrast is not optimal refresh value.\n')
print(json.dumps(v,indent=2))
