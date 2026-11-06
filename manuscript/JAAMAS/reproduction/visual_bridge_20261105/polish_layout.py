from pathlib import Path
import re
P=Path('/home/ez/DACI/Springer/DACI_JAAMAS')
p=P/'submission/information_sheet.tex';p.write_text(p.read_text().replace(r'\newpage',''))
# Keep the final appendix coherent: its text and both six-row tables share one page.
p=P/'sections/appendix.tex';s=p.read_text();start=s.index(r'\section{Visual replication by training seed}')
tail=s[start:];tail=tail.replace(r'\input{source_data/visual_seed_contrasts}'+'\n','').replace(r'\input{source_data/visual_fullseed_competence}'+'\n','')
s=s[:start]+r'\clearpage'+'\n'+tail+'\n'+r'\input{source_data/visual_seed_contrasts}'+'\n'+r'\input{source_data/visual_fullseed_competence}'+'\n';p.write_text(s)
for stem in ['visual_seed_contrasts','visual_fullseed_competence']:
 p=P/'source_data'/f'{stem}.tex';p.write_text(p.read_text().replace(r'\begin{table}[tb]',r'\begin{table}[H]'))
# These short tables fit intact; prevent isolated continuation rows and deferred figure-only breaks.
for p in (P/'supplement/tables').glob('*.tex'):
 txt=p.read_text()
 def convert(m):
  col=m[1];lines=m[2].strip().splitlines();caption=lines[0]
  assert caption.endswith(r'\\');caption=caption[:-2]
  header=lines[1].replace(r'\endfirsthead','');assert lines[2].endswith(r'\endhead')
  rest='\n'.join(lines[3:]);return r'\begin{table}[H]\centering\small'+'\n'+caption+'\n'+r'\begin{tabular}{'+col+'}\n'+header+'\n'+rest+r'\end{tabular}\end{table}'
 txt=re.sub(r'\\begin\{longtable\}\{([^}]+)\}\s*\n(.*?)\\end\{longtable\}',convert,txt,flags=re.S)
 txt=txt.replace(r'\begin{table}[tb]',r'\begin{table}[H]');p.write_text(txt)
p=P/'supplement/extended_evidence.tex';s=p.read_text().replace(r'\begin{figure}[tb]',r'\begin{figure}[H]');p.write_text(s)
