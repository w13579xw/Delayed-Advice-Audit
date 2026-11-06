"""Versioned editable manuscript package after completed analyses and final QA."""
from pathlib import Path
import hashlib,json,re,zipfile,datetime
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');O=P.parent/'DACI_JAAMAS_revision_20261105_sources.zip';assert not O.exists()
assert json.loads((R/'outputs/jaamas/visual_revision_postprocess_20261105_v1/run.json').read_text())['status']=='complete'
assert json.loads((P/'FINAL_CHECKS.json').read_text())['current_revision_status']=='complete_with_author_checks'
files=[]
for name in ['main.tex','bibliography.tex','build.sh','svjour3.cls','svglov3.clo','spmpsci.bst','README.md','AUTHOR_CHECKLIST.md','WRITING_AUDIT.md','FINAL_CHECKS.json','DACI_JAAMAS_manuscript.pdf']:
 files.append(P/name)
for directory in ['sections','source_data','reproduction']:
 files.extend(p for p in (P/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.tex','.json','.md','.sh','.csv','.mplstyle','.txt'])
for stem in ['daci_workflow','null_counterexample','simulator_scenes','critic_learning_curve','state_contrasts']:
 files.extend(P/'figures'/(stem+'.'+ext) for ext in ['pdf','svg','png'])
files.extend((P/'figures').glob('*record*.json'))
files.extend(P/'submission'/(stem+'.'+ext) for stem in ['information_sheet','cover_letter'] for ext in ['tex','pdf'])
files=sorted(set(files));assert all(p.is_file() for p in files);hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),type='editablemanuscript/sourcefigures/auditentries;notstandaloneexperimentweightsrelease',file_sha256=hashes,full_research_workspace=str(R),author_verification_pending=True)
with zipfile.ZipFile(O,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,Path('DACI_JAAMAS')/p.relative_to(P))
 z.writestr('DACI_JAAMAS/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(O) as z:assert z.testzip() is None;assert len(z.namelist())==len(files)+1
record=dict(zip=str(O),bytes=O.stat().st_size,files=len(files)+1,sha256=hashlib.sha256(O.read_bytes()).hexdigest(),crc='passed',utc=manifest['utc']);(R/'outputs/jaamas/revision_20261105/source_package_check.json').write_text(json.dumps(record,indent=2));print(record)
