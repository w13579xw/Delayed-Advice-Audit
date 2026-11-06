from pathlib import Path
import json,hashlib,shutil
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');A=R/'outputs/jaamas/decoder_analysis_20261106_v1';a=json.loads((A/'analysis.json').read_text())['main']
assert json.loads((R/'outputs/jaamas/decoder_revision_20261106/post_queue_complete.json').read_text())['status']=='complete'
manifest=json.loads((P/'source_data/manifest.json').read_text())
copies={'decoder_analysis.json':A/'analysis.json','decoder_audit.json':A/'audit.json','decoder_source_audit.json':R/'outputs/jaamas/decoder_source_audit_20261106_v1/checks.json','decoder_vectors.npz':A/'condition_vectors.npz'}
for name,f in copies.items():
 shutil.copy2(f,P/'source_data'/name);manifest=[x for x in manifest if x['file']!=name];manifest.append(dict(file=name,source=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(P/'source_data/manifest.json').write_text(json.dumps(manifest,indent=2))
names=[('fixed_time_minus_fixed_fresh_legacy','Fixed48','Fresh: time $-$ legacy'),('fixed_time_minus_fixed_stale_legacy','Fixed48','Time-fresh $-$ stale'),('gate_time_minus_gate_fresh_legacy',r'Frozen $P,F$','Fresh: time $-$ legacy'),('gate_time_minus_gate_stale_legacy',r'Frozen $P,F$','Time-fresh $-$ stale')]
lines=[r'\begin{table}[tb]\centering\small',r'\caption{Independent generation-time decoding control: four primary reward contrasts on 60 new environments. Time uses generation age; legacy uses request delay. Stale decoding is identical for both. Approximate 98.75\% stratified paired-bootstrap intervals retain the four-contrast family. Fixed48 applies the predefined source suffix on every environment; the frozen gate uses unchanged choices across fresh decoders. This study does not alter the original channel estimates.}\label{tab:decoder}',r'\setlength{\tabcolsep}{3pt}\begin{tabular}{llrr}\toprule Receiver & Contrast & Mean & Interval\\\midrule']
for key,title,label in names:
 x=a['primary_contrasts'][key];lo,hi=x['ci'];lines.append(f"{title} & {label} & {x['mean']:+.3f} & [{lo:+.3f}, {hi:+.3f}]"+r'\\')
lines.append(r'\bottomrule\end{tabular}\end{table}');(P/'source_data/decoder_table.tex').write_text('\n'.join(lines)+'\n')
text=r'''Matching generation time improves the predefined fresh 48-step chunk relative to applying its old delay offset: the paired mean gain is 0.579 (Table~\ref{tab:decoder}). This establishes an execution-index effect for that fixed option in the new cohort. Its time-matched fresh versus stale difference remains unresolved, as does its descriptive advantage over continued local control ($+0.120$, 95\% interval $[-0.187,0.433]$). A better decoder comparison is not evidence of useful refreshing or an overall advice advantage.
\input{source_data/decoder_table}

The frozen $P,F$ gate responds differently. It chooses a suffix in 24/60 environments (21 for48 steps and three for16), so time matching changes 24 action sequences while changing no fresh-channel choice. Its mean decoder effect is $-0.030$, with a primary interval crossing zero. Local-option selections are unaffected. The fixed option's overall gain therefore does not transfer automatically to the gate's selected states and intervention lengths. The gate still predicts legacy-execution gains; this study does not test learning or recalibrating it for the new rule. Complete channel/decoder outcomes, success endpoints and index checks are in Online Resource1.

These new results qualify a practical reading of the earlier channel test: literal replacement in the legacy receiver and generation-matched execution answer different questions. We retain both, with separate environments, receivers and inference families. The positive fixed-chunk decoder result is not used to relabel the original refresh effect or the packet-independent $+0.735$ counterexample.
'''
(P/'source_data/decoder_results.tex').write_text(text)
conditions=[('R','Reject'),('fixed_S_legacy','Fixed48 / stale'),('fixed_F_legacy','Fixed48 / fresh legacy'),('fixed_F_time','Fixed48 / fresh time'),('gate_S_legacy',r'$P,F$ / stale'),('gate_F_legacy',r'$P,F$ / fresh legacy'),('gate_F_time',r'$P,F$ / fresh time')]
lines=[r'\begin{table}[H]\centering\small',r'\caption{All independent decoder-study conditions on the same 60 new environments. Native success endpoints are descriptive, not extra primary tests. Main Table~\ref{main-tab:decoder} reports the four primary reward contrasts once.}\label{s-tab:decoder-outcomes}',r'\begin{tabular}{lrrr}\toprule Condition & Mean reward & Any (\%) & Terminal (\%)\\\midrule']
for c,label in conditions:
 x=a['condition_means'][c];lines.append(f"{label} & {x['reward']:.6f} & {100*x['any_success']:.1f} & {100*x['terminal_success']:.1f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}\end{table}')
lines.extend([r'\begin{table}[H]\centering\small',r'\caption{Secondary decoder-study differences from reject. Reward intervals are nominal95\% descriptive intervals, not the four primary tests or a new confirmatory family. Success differences are percentage points. All conditions are retained.}\label{s-tab:decoder-reject}',r'\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}\toprule Condition & Mean & Reward interval & Any & Terminal\\\midrule'])
for c,label in conditions[1:]:
 x=a['secondary_vs_reject'][c];lo,hi=x['reward_ci95'];lines.append(f"{label} & {x['mean']:+.4f} & [{lo:+.4f},{hi:+.4f}] & {100*x['any_success_difference']:+.1f} & {100*x['terminal_success_difference']:+.1f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}\end{table}')
(P/'supplement/tables/decoder_outcomes.tex').write_text('\n'.join(lines)+'\n')
with (P/'supplement/decoder_details.tex').open('a') as f:f.write('\n'+r'\subsection{Complete observed outcomes}'+'\n'+r'\input{tables/decoder_outcomes}'+'\nThe fresh gate selects local head3 in15 environments, request-action hold in14 and local ensemble in7, plus source suffix48 in21 and suffix16 in3. All24 age-sensitive selections change actions. These conditional selections are not randomized subgroups. The fixed48 analysis applies to all60 environments; comparing it with selected gate cases is not an identified causal decomposition of router failure.\n')
print(json.dumps(a['primary_contrasts'],indent=2))
