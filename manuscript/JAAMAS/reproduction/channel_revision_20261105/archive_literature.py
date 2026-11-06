"""Public primary-source and bibliographic audit; no manuscript uploading."""
import json,urllib.request,hashlib,datetime,concurrent.futures,subprocess
from pathlib import Path
from html.parser import HTMLParser
O=Path('/home/ez/DACI/code/outputs/jaamas/literature_revision_20261105_v1');O.mkdir(exist_ok=False)
class P(HTMLParser):
 def __init__(self):super().__init__();self.s=[];self.skip=0
 def handle_starttag(self,t,a):
  if t in ['script','style']:self.skip+=1
 def handle_endtag(self,t):
  if t in ['script','style']:self.skip=max(0,self.skip-1)
 def handle_data(self,d):
  if not self.skip and d.strip():self.s.append(d.strip())
urls={
 'torrey':'https://www.ifaamas.org/Proceedings/aamas2013/docs/p1053.pdf',
 'taylor':'https://jmlr.org/papers/v10/taylor09a.html',
 'griffith':'https://proceedings.neurips.cc/paper_files/paper/2013/hash/e034fb6b66aacc1d48f445ddfb08da98-Abstract.html',
 'macglashan':'https://proceedings.mlr.press/v70/macglashan17a.html',
 'celemin':'https://link.springer.com/article/10.1007/s10846-018-0839-z',
 'dulac':'https://link.springer.com/article/10.1007/s10994-021-05961-4',
 'ramstedt':'https://proceedings.neurips.cc/paper/2019/hash/54e36c5ff5f6a1802925ca009f3ebb68-Abstract.html',
 'bouteiller':'https://arxiv.org/pdf/2010.02966',
 'diffusion':'https://www.roboticsproceedings.org/rss19/p026.html',
 'ross':'https://proceedings.mlr.press/v15/ross11a.html',
 'jiang':'https://proceedings.mlr.press/v48/jiang16.html',
 'thomas':'https://proceedings.mlr.press/v48/thomasa16.html',
 'swaminathan':'https://proceedings.mlr.press/v37/swaminathan15.html',
 'dudik':'https://www.microsoft.com/en-us/research/publication/doubly-robust-policy-evaluation-and-optimization/',
 'cawley':'https://jmlr.org/papers/v11/cawley10a.html',
 'varma':'https://link.springer.com/article/10.1186/1471-2105-7-91',
 'henderson':'https://ojs.aaai.org/index.php/AAAI/article/view/11694',
 'fisher':'https://jmlr.org/papers/v20/18-760.html',
 'hooker':'https://link.springer.com/article/10.1007/s11222-021-10057-z',
 'strobl':'https://link.springer.com/article/10.1186/1471-2105-9-307',
 'fachantidis_volume':'https://www.mdpi.com/2504-4990/1/1/2',
}
dois={'celemin':'10.1007/s10846-018-0839-z','dulac':'10.1007/s10994-021-05961-4','dudik':'10.1214/14-STS500','varma':'10.1186/1471-2105-7-91','henderson':'10.1609/aaai.v32i1.11694','hooker':'10.1007/s11222-021-10057-z','strobl':'10.1186/1471-2105-9-307','fachantidis':'10.3390/make1010002','act':'10.15607/RSS.2023.XIX.016','diffusion':'10.15607/RSS.2023.XIX.026'}
urls.update({k+'_crossref':'https://api.crossref.org/works/'+v for k,v in dois.items()})
def one(k,u):
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'DACI-research-source-audit/1.0'}),timeout=45) as response:b=response.read();code=response.status
  ext='.pdf' if b.startswith(b'%PDF') else ('.json' if '/works/' in u else '.html');file=O/(k+ext);file.write_bytes(b)
  if ext=='.pdf':subprocess.run(['pdftotext','-layout',str(file),str(O/(k+'.txt'))],check=True,stdout=subprocess.DEVNULL)
  elif ext=='.html':p=P();p.feed(b.decode('utf8',errors='replace'));(O/(k+'.txt')).write_text('\n'.join(p.s))
  return dict(key=k,url=u,http=code,file=str(file),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except Exception as e:return dict(key=k,url=u,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(lambda kv:one(*kv),urls.items()))
(O/'verification.json').write_text(json.dumps(records,indent=2));print([(v['key'],v.get('http',v.get('error'))) for v in records])
for key in dois:
 p=O/(key+'_crossref.json')
 if p.exists():
  x=json.loads(p.read_text())['message'];print(key,x.get('title'),x.get('container-title'),x.get('volume'),x.get('issue'),x.get('page'),x.get('article-number'),x.get('published'))
