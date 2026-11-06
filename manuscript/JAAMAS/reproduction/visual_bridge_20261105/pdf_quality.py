"""Geometry/font/text/log check and all-page renders of completed deliverables."""
from pathlib import Path
import json,subprocess,re,hashlib
import pdfplumber
from PIL import Image,ImageOps,ImageDraw
P=Path('/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS');O=P/'preview_bridge';O.mkdir(exist_ok=True)
docs={'main':P/'Delayed_Advice_Audit_manuscript.pdf','supplement':P/'supplement/Delayed_Advice_Audit_Online_Resource_1.pdf','information':P/'submission/information_sheet.pdf','letter':P/'submission/cover_letter.pdf'}
report={}
for name,file in docs.items():
 d=O/name;d.mkdir(exist_ok=True);viol=[];texts=[]
 for old in d.glob('page-*.png'):old.unlink()
 with pdfplumber.open(file) as pdf:
  for i,p in enumerate(pdf.pages):
   texts.append(p.extract_text() or '')
   for c in p.chars:
    if c['x0']<-.5 or c['x1']>p.width+.5 or c['top']<-.5 or c['bottom']>p.height+.5:viol.append([i+1,c['text'],c['x0'],c['top']])
  pages=len(pdf.pages)
 assert not viol,(name,viol[:5])
 text='\n\f\n'.join(texts);(O/(name+'_text.txt')).write_text(text)
 assert '??' not in text,(name,'unresolved reference')
 fonts=subprocess.check_output(['pdffonts',str(file)],text=True)
 assert all(line.split()[-5]=='yes' for line in fonts.splitlines()[2:] if line.strip()),fonts
 subprocess.run(['pdftoppm','-png','-r','85',str(file),str(d/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 images=sorted(d.glob('page-*.png'));assert len(images)==pages
 for start in range(0,pages,6):
  subset=images[start:start+6];w,h=430,620;canvas=Image.new('RGB',(w*3,h*2),'#e8e8e8');dr=ImageDraw.Draw(canvas)
  for k,f in enumerate(subset):
   im=Image.open(f).convert('RGB');im.thumbnail((w-16,h-30));x=(k%3)*w+(w-im.width)//2;y=(k//3)*h+24;canvas.paste(im,(x,y));dr.text(((k%3)*w+12,(k//3)*h+5),f'{name} p{start+k+1}',fill='black')
  canvas.save(O/f'{name}_montage_{start+1:02d}.png')
 report[name]=dict(path=str(file),pages=pages,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),characters_outside_page=len(viol),fonts_embedded=True,text_words=len(text.split()))
logs=[P/'main.log',P/'supplement/supplement.log',P/'submission/information_sheet.log',P/'submission/cover_letter.log']
for log in logs:
 txt=log.read_text();assert 'Overfull' not in txt and 'undefined' not in txt and 'multiply defined' not in txt,log
report['logs_clean']=True;(O/'quality_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
