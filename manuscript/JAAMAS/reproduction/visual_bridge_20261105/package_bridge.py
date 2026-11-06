from pathlib import Path
import hashlib,json,zipfile,datetime
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');O=P.parent/'DACI_JAAMAS_bridge_revision_20261105_sources.zip'
assert not O.exists()
assert json.loads((R/'outputs/jaamas/bridge_revision_20261105/post_queue_complete.json').read_text())['status']=='complete'
assert json.loads((P/'FINAL_CHECKS.json').read_text())['current_revision_status']=='complete_with_author_checks'
paths=[P/x for x in ['main.tex','bibliography.tex','build.sh','svjour3.cls','svglov3.clo','spmpsci.bst','README.md','AUTHOR_CHECKLIST.md','WRITING_AUDIT.md','FINAL_CHECKS.json','DACI_JAAMAS_manuscript.pdf']]
for directory in ['sections','source_data','reproduction','supplement']:
 paths.extend(p for p in (P/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.tex','.json','.md','.sh','.csv','.mplstyle','.txt','.npz'])
paths.append(P/'supplement/DACI_Online_Resource_1.pdf')
paths.extend(P/'submission'/(stem+'.'+ext) for stem in ['information_sheet','cover_letter'] for ext in ['tex','pdf'])
paths.extend(p for p in (P/'figures').iterdir() if p.is_file() and p.suffix in ['.pdf','.svg','.png','.json'])
paths=sorted(set(paths));assert all(f.is_file() for f in paths)
manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='editablearticle+OnlineResource1+submissiondrafts+compactsourceData+auditentries;requiresfullworkspacefornativeexperimentregeneration',file_sha256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},full_workspace=str(R),author_checks_pending=True)
with zipfile.ZipFile(O,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in paths:z.write(f,Path('DACI_JAAMAS')/f.relative_to(P))
 z.writestr('DACI_JAAMAS/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(O) as z:
 assert z.testzip() is None
 for f,h in manifest['file_sha256'].items():assert hashlib.sha256(z.read('DACI_JAAMAS/'+f)).hexdigest()==h
result=dict(path=str(O),sha256=hashlib.sha256(O.read_bytes()).hexdigest(),bytes=O.stat().st_size,entries=len(paths)+1,crc='passed',every_member_sha256='passed',utc=manifest['utc'])
(R/'outputs/jaamas/bridge_revision_20261105/source_package_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
