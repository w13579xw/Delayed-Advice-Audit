from pathlib import Path
import re,json
P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS');S=P/'sections';T=P/'supplement';T.mkdir(exist_ok=False)
methods=(S/'experiments.tex').read_text().split(r'\subsection{Research execution and AI assistance}')[0]
methods=methods.replace(r'\section{Experimental design}\label{sec:design}',r'\section{Extended experimental methods}')
methods=re.sub(r'\\input\{source_data/study_inventory\}\n','',methods)
methods=re.sub(r'\\begin\{figure\}\[tb\].*?\\end\{figure\}\n','',methods,flags=re.S)
combined=methods+'\n'+(S/'appendix.tex').read_text()
inputs=re.findall(r'\\input\{([^}]+)\}',combined)
labels=set(re.findall(r'\\label\{([^}]+)\}',combined))
for f in inputs:labels.update(re.findall(r'\\label\{([^}]+)\}',(P/f).with_suffix('.tex').read_text()))
def convert(txt):
 txt=re.sub(r'\\label\{([^}]+)\}',lambda m:r'\label{s-'+m[1]+'}',txt)
 return re.sub(r'\\ref\{([^}]+)\}',lambda m:r'\ref{'+('s-' if m[1] in labels else 'main-')+m[1]+'}',txt)
combined=convert(combined).replace('figures/','../figures/')
for f in inputs:
 path=(P/f).with_suffix('.tex');dest=T/'tables'/path.name;dest.parent.mkdir(exist_ok=True);dest.write_text(convert(path.read_text()));combined=combined.replace(r'\input{'+f+'}',r'\input{tables/'+path.stem+'}')
(T/'extended_evidence.tex').write_text(combined)
preamble=(P/'main.tex').read_text().split(r'\begin{document}')[0]
preamble=preamble.replace(r'\journalname{Autonomous Agents and Multi-Agent Systems}',r'\journalname{Autonomous Agents and Multi-Agent Systems}'+'\n'+r'\usepackage{xr}'+'\n'+r'\externaldocument[main-]{../main}')
preamble=preamble.replace(r'\hypersetup{pdftitle={Delayed Advice Audit: Auditing the Value of Delayed Advice in Autonomous Agents}}',r'\hypersetup{pdftitle={Delayed Advice Audit Online Resource 1: Extended Evidence}}')
(T/'supplement.tex').write_text(preamble+r'''
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\title{Online Resource 1: Extended Evidence for Delayed Advice Audit}
\titlerunning{Delayed Advice Audit: Extended Evidence}
\author{Author information pending}
\institute{Companion to the Delayed Advice Audit main article.}
\date{}\maketitle
This resource preserves detailed experimental methods, every reported condition, numerical selection grids, historical source-budget studies and engineering provenance. It supports the compact main article without supplying additional independent cohorts. Section and table references without an S prefix refer to the main article. Current visual-channel and temporal extensions are documented after the retained evidence; historical protocols remain distinguishable.
\input{extended_evidence}
\input{bridge_details}
\begin{thebibliography}{3}
\bibitem{metaworld} Yu T et al (2020) Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. CoRL, PMLR 100:1094--1100.
\bibitem{act} Zhao T Z et al (2023) Learning fine-grained bimanual manipulation with low-cost hardware. RSS. \url{https://doi.org/10.15607/RSS.2023.XIX.016}
\bibitem{agarwal} Agarwal R et al (2021) Deep reinforcement learning at the edge of the statistical precipice. NeurIPS 34:29304--29320.
\end{thebibliography}
\end{document}
''')
(T/'bridge_details.tex').write_text(r'\section{Additional visual-channel and temporal audit records}'+'\nDetailed frozen schedules, model and entry hashes, source regeneration, complete raw trajectories and analysis vectors accompany the source-data manifest. They are engineering provenance rather than additional independent performance samples.\n')
before={f.name:len(re.findall(r'\b[\w-]+\b',f.read_text())) for f in S.glob('*.tex')}
(P/'reproduction/compression_baseline.json').write_text(json.dumps(dict(section_token_proxy=before,prior_main_pdf_pages=40,role='token proxy includes TeX, not journal word count'),indent=2))
