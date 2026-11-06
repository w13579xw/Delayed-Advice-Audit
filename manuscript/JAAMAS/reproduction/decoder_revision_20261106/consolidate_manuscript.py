from pathlib import Path
import re,json,shutil
P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');R=Path('/home/ez/DACI/code');O=R/'outputs/jaamas/decoder_revision_20261106';A=(P/'sections/appendix.tex').read_text()
baseline={str(p.relative_to(P)):p.read_text() for p in [P/'main.tex',*list((P/'sections').glob('*.tex'))]}
(O/'prose_before.json').write_text(json.dumps(baseline,indent=2))
shutil.copy2(P/'FINAL_CHECKS.json',O/'previous_final_checks.json')
x=json.loads((P/'FINAL_CHECKS.json').read_text());x.update(status='revision_in_progress',current_revision_status='decoder_control_and_consolidation_in_progress',previous_complete_draft='DACI_JAAMAS_before_decoder_20261106.zip');(P/'FINAL_CHECKS.json').write_text(json.dumps(x,indent=2))
# Move only unique details; original optimization and per-seed tables already exist in SI.
candidates=A.split(r'\subsection{Candidate functions and fixed continuation}')[1].split('Local pose heads')[0]
nonzero=A.split(r'\subsection{Exploratory state results at nonzero delay}')[1].split(r'\subsection{Why Door')[0]
temporal=A.split(r'\subsection{Why Door ever-success and terminal-success disagree}')[1].split(r'\clearpage')[0]
temporal=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',temporal,flags=re.S)
def siconvert(t):
 t=t.replace(r'\label{app:nonzero}',r'\label{s-sec:nonzero}').replace(r'\label{app:temporal}',r'\label{s-sec:temporal}')
 t=t.replace(r'\ref{tab:nonzero}',r'\ref{s-tab:nonzero}').replace(r'\ref{tab:door-duration}',r'\ref{s-tab:door-duration}').replace(r'\ref{fig:door-time}',r'\ref{main-fig:door-time}')
 t=t.replace(r'\input{source_data/nonzero_delay_table}',r'\input{tables/nonzero_delay_table}').replace(r'\input{source_data/door_duration_table}',r'\input{tables/door_duration_table}')
 return t
for stem in ['inference_table','nonzero_delay_table','door_duration_table']:
 txt=(P/'source_data'/f'{stem}.tex').read_text().replace(r'\begin{table}[tb]',r'\begin{table}[H]')
 txt=re.sub(r'\\label\{([^}]+)\}',lambda m:r'\label{s-'+m[1]+'}',txt)
 if stem=='inference_table':txt=txt.replace(r'\\\bottomrule',r'\\'+'\n'+r'Decoder control & 60 & 30,000 & 98.7500 & Stratified env.; 4\\\bottomrule')
 (P/'supplement/tables'/f'{stem}.tex').write_text(txt)
t=r'''\section{Consolidated execution and statistical contract}
\subsection{Candidate execution and fixed continuation}
'''+candidates+r'''
\subsection{Primary inference families}\label{s-sec:inferencematrix}
\input{tables/inference_table}
The original and decoder-control studies have separate predeclared inference families. The decoder changes execution semantics and therefore cannot revise the earlier fixed-policy channel estimate. Source and entry hashes, complete replay data and source-message regeneration are provenance in the accompanying files, rather than additional independent environments.
\section{State delay and temporal diagnostics}
\subsection{Exploratory nonzero delays}
'''+siconvert(nonzero)+r'''
\subsection{Door success duration}
'''+siconvert(temporal)+r'''
\section{Independent generation-time decoder control}\label{s-sec:decoder}
\input{decoder_details}
'''
(P/'supplement/consolidated_details.tex').write_text(t)
(P/'supplement/decoder_details.tex').write_text('The independently frozen decoder-control study is in progress. No result is inferred from prior channel outcomes.\n')
p=P/'supplement/supplement.tex';s=p.read_text().replace(r'\input{bridge_details}',r'\input{bridge_details}'+'\n'+r'\input{consolidated_details}');p.write_text(s)
p=P/'main.tex';s=p.read_text();s=s.replace(r'\appendix'+'\n'+r'\normalsize'+'\n'+r'\input{sections/appendix}'+'\n','');p.write_text(s)
(P/'sections/appendix.tex').write_text('% Former main Appendices A-C are consolidated in Online Resource 1; not included by main.tex.\n')
p=P/'sections/experiments.tex';s=p.read_text()
s=s.replace(r'Appendix~\ref{app:implementation} specifies their execution rules.','Online Resource 1 specifies their execution rules.')
s=s.replace(r'Table~\ref{tab:inference} gives every primary family, sample size and confidence level.','The inference contract in Online Resource 1 gives every primary family, sample size and confidence level.')
anchor=r'\subsection{Training variation and supporting studies}'
method=r'''\subsection{Independent generation-time decoder control}\label{sec:decodersetup}
A separate experiment tests execution semantics on 60 new PickPlace environments, using the original seed-0 actors and critics without retraining. Delays 16/32 cross requests 16/32/64, with ten environments per cell; six zero-delay identities and two engineering cases are excluded. For a message generated at $g$, the time-matched suffix/blend decoder uses $(a-g)+\text{elapsed}$. Stale messages have $g=q$, reproducing the original index; refreshed messages have $g=a$, starting at zero. Raw-prefix and local options are unchanged.

We evaluate both a predeclared fixed 48-step source suffix and the frozen $P,F$ gate. The gate receives unchanged packet/nonmessage features and selects the same option for both fresh decoders; only the downstream execution index changes. The four primary paired reward contrasts compare time-matched fresh execution against legacy fresh and legacy stale execution, separately for the fixed option and gate. Thirty thousand whole-environment bootstrap resamples within the six cells give approximate 98.75\% intervals for this independent four-contrast family. Secondary success and reject comparisons are descriptive.

This changes the executable receiver and therefore is an independent decoder-policy comparison, not a reinterpretation of the original fixed-policy channel effect. The critic remains fitted to legacy outcomes; neither age-aware retraining nor optimal source timing is tested. No extra refresh latency or service cost is charged. Online Resource 1 retains the complete protocol, outcomes and index checks.

'''
s=s.replace(anchor,method+anchor);p.write_text(s)
p=P/'sections/results.tex';s=p.read_text()
s=s.replace(r"Appendix~\ref{app:temporal} shows its brief success occupancy and full time courses; mean shaped reward captures trajectory differences hidden by these binary endpoints. This does not make reward a universally superior task objective. Post-hoc nonzero-delay state intervals likewise retain uncertainty (Appendix~\ref{app:nonzero}); they do not replace the original primary analysis.",r"Only about 30 of 500 steps satisfy its native coordinate-band criterion. Figure~\ref{fig:door-time} shows this transient success alongside sustained shaped reward; flag loss alone does not identify door reclosure. Full duration data and post-hoc nonzero-delay uncertainty are in Online Resource 1. These endpoints answer different questions; reward is not a universally superior objective.")
figure=r'''
\begin{figure}[tb]\centering
\includegraphics[width=\textwidth]{figures/door_temporal.pdf}
\caption{Door success and native reward under stale, fresh and absent-message channels. Means retain all 60 environments and five frozen $P,F$ critics with environment-equal weighting. Brief success-band occupancy explains the 95\% ever-success/0\% terminal-success discrepancy, while shaped reward remains high. No trajectory selection or timewise inference is used. Full descriptive data are in Online Resource 1.}\label{fig:door-time}
\end{figure}
'''
s=s.replace(r'\subsection{Headroom separates opportunity',figure+'\n'+r'\subsection{Headroom separates opportunity')
s=s.replace(r'Per-seed results and weak source competence remain included in Appendix~\ref{app:seeds}.','Per-seed contrasts and complete-source competence, including all weak outcomes, are retained only in Online Resource 1, Tables S8--S9.')
s=s.replace(r'\subsection{Supporting analyses bound rather than erase the counterexample}',r'\subsection{Message-generation time is a separate execution intervention}\label{sec:decoderresults}'+'\n'+r'\input{source_data/decoder_results}'+'\n'+r'\subsection{Supporting analyses bound rather than erase the counterexample}')
p.write_text(s)
(P/'source_data/decoder_results.tex').write_text('The new decoder study is being completed under its separate predeclared protocol.\n')
