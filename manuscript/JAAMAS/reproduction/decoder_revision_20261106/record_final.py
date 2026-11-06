from pathlib import Path
import json,re,zipfile,hashlib,subprocess,datetime,shutil,sys
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');O=R/'outputs/jaamas/decoder_revision_20261106';D=R/'docs/learned_advice/decoder_revision_20261106';B=R/'outputs/jaamas/decoder_control_20261106_v1'
sys.path.insert(0,str(R/'scripts'));from server_run import fingerprint
fp=fingerprint();assert fp=='48ffd6ffff1e3eeb5cc15a475888a844c846ead878292e6137c7f171c5246221'
assert json.loads((O/'post_queue_complete.json').read_text())['status']=='complete'
qa=json.loads((P/'preview_decoder/quality_checks.json').read_text());v=json.loads((P/'source_data/decoder_analysis.json').read_text());a=json.loads((P/'source_data/decoder_audit.json').read_text());index=json.loads((O/'all_source_index_audit.json').read_text())
for name,q in qa.items():
 if name!='logs_clean':assert hashlib.sha256(Path(q['path']).read_bytes()).hexdigest()==q['sha256']
for f,h in {**json.loads((B/'main/protocol.json').read_text())['model_hashes'],**json.loads((B/'main/protocol.json').read_text())['entry_hashes']}.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
m=json.loads((P/'source_data/manifest.json').read_text())
for x in m:assert (P/'source_data'/x['file']).read_bytes()==Path(x['source']).read_bytes() and hashlib.sha256((P/'source_data'/x['file']).read_bytes()).hexdigest()==x['sha256']
for f in ['svjour3.cls','svglov3.clo','spmpsci.bst']:assert (P/f).read_bytes()==(P.parent/f).read_bytes()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
oldzip=P.parent/'DACI_JAAMAS_before_decoder_20261106.zip'
with zipfile.ZipFile(oldzip) as z:
 oldpdf=next(n for n in z.namelist() if n.endswith('/DACI_JAAMAS_manuscript.pdf'));(O/'prior30.pdf').write_bytes(z.read(oldpdf))
 oldresult=z.read('DACI_JAAMAS/sections/results.tex').decode()
 def expand(t,archive=False):
  for _ in range(5):
   if r'\input{' not in t:break
   t=re.sub(r'\\input\{([^}]+)\}',lambda n:z.read('DACI_JAAMAS/'+n[1]+'.tex').decode() if archive else (P/(n[1]+'.tex')).read_text(),t)
  return t
 def subsections(t,archive=False):
  d={}
  for item in re.split(r'\\subsection\{',t)[1:]:
   title=item.split('}',1)[0];body=item.split('}',1)[1];body=expand(body,archive);body=re.sub(r'\\[A-Za-z]+',' ',body);d[title]=len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b",body))
  return d
 metrics=dict(method='pdftotext whitespace tokens forPDF; expandedTeX ASCIIwordtokensforsubsectionsincludingcaptions,notpublisherwordcount',main_before=len(subprocess.check_output(['pdftotext',str(O/'prior30.pdf'),'-'],text=True).split()),main_after=len(subprocess.check_output(['pdftotext',str(P/'DACI_JAAMAS_manuscript.pdf'),'-'],text=True).split()),results_before=subsections(oldresult,True),results_after=subsections((P/'sections/results.tex').read_text()))
(O/'editorial_metrics.json').write_text(json.dumps(metrics,indent=2))
p=D/'CHECKLIST.md';p.write_text(p.read_text().replace('- [ ]','- [x]'))
# Checklist's final archive item is closed by the required package+extracted build before delivery.
p=D/'REPORT.md';s=p.read_text().replace('状态：执行中，2026-11-06。','状态：本轮修改与实验完成，待作者事实核实，未投递；2026-11-06。')
s+='''
## 新实验的完整结果与结论边界
四项预声明主奖励对比，均为60独立新环境、98.75%分层配对bootstrap区间：
|接收器/动作条件|对比|均值差|区间|
|---|---|---:|---|
|固定48步源动作|fresh时间匹配−fresh旧偏移|+0.579061|[+0.172635,+0.997332]|
|固定48步源动作|fresh时间匹配−stale一致解码|+0.148734|[−0.204366,+0.516121]|
|冻结PF选择器|fresh时间匹配−fresh旧偏移|−0.029792|[−0.297956,+0.192171]|
|冻结PF选择器|fresh时间匹配−stale一致解码|−0.188259|[−0.541029,+0.115920]|

固定源48条件证明在本队列中，新动作块跳过前段确实会损害该选项的执行效果；匹配生成时间相对旧偏移获得正向改进。这不是新控制算法，时间匹配fresh48等价于原有raw48选项。相对stale源或继续本地控制的收益仍未建立。后者仅为描述性+0.120043，95%区间[−0.186880,+0.433330]。
PF的fresh选项固定：21个source_suffix48、3个source_suffix16、36个本地选项。解码改变24条实际动作轨迹，但没有改变fresh的选择；平均执行改进未成立。固定48条件的收益不能直接转移到选择器的不同状态分布和干预长度，也不能据此证明重新训练后的最佳控制器如何表现。
S的两种解码数学相同，因此时间匹配fresh−legacy stale也等于新时间感知执行策略下的fresh−stale通道差，但绝不替代原B1接收器的效应。原B1 Table5/源JSON以及三种子Table7表源与修改前ZIP逐字节核对保持一致。

## 执行与独立核验
正式主实验295条独特native执行计划、159372个实际物理步骤（含每例完整reject重放），约1488秒。含6个d0、2个工程检查为320条路径、174848步；独立消息源检查另执行4000步。68案例544个保存的gate选择重算一致，204组固定48索引条件通过。额外只读检查覆盖全部200条实际源suffix计划、9280个源动作步，索引逐项匹配；本队列没有blend或raw源计划。原始奖励均值重构误差0。
独立物理前缀再生136组请求/到达packet、特征及heads逐位相同。所有正式和检查样本保留，无新增训练、模型/阈值选择或测试样本扩充。生产fingerprint48ffd6...和全部冻结入口/模型SHA保持不变。原生作业、独立分析、消息再生和文稿构建均退出0；没有跳过失败测试或缩短500步时域。仅运行相关新索引/轨迹审计；生产代码未改，未无故重跑旧toy/Fetch全套。

## 去重后的稿件
主稿28页，不再编入原AppendixA–C。保留跨种子Table7，逐种子与完整source competence只出现在SI的S8–S9。Door图为正文Fig4，完整持续时长、first/last、最长成功段和非零延迟数据只放SI。原候选/推断详情也仅在SI。
额外去重：SI的state channel表只保留正文没有的三个PC对比；删去与正文完全重复的visual oracle总结表。原推断族、全向量和旧源文件仍存档；未隐藏负结果。
新独立decoder四主对比仅在正文Table8完整显示，SI只给7条件均值/成功率、全部次要reject对比及方法/审计。主稿4图/8表/Algorithm1、229词摘要、31正式参考文献；SI19页/2图/19表；Info2页、cover1页。未缩字号、页边距或改用户模板class。修正补充末页孤立参考文献，通过压缩重复图注/引用条目而非删除数据解决。
所有28+19+2+1页均render检查，重点Table8与Door图、SI种子/生成时间/完整结果页另检查。四PDF无undefined/overfull/越页字符，字体全部嵌入。22份source-data文件（19JSON、3紧凑NPZ）与分析源完全一致；最终哈希见FINAL_CHECKS.json。新源ZIP为DACI_JAAMAS_decoder_revision_20261106_sources.zip；CRC、逐成员SHA及新解压构建记录另存本轮outputs目录，避免循环哈希。
源包是可编辑论文/审计入口包，native再生依赖完整本地模型与历史数据。未进行独立机器复现、人工作者认证、公开发布或投稿。作者事实/责任声明仍保留待核实。
'''
s+='\n## 编辑计数\n同一pdftotext空白词元口径，主稿'+str(metrics['main_before'])+'→'+str(metrics['main_after'])+'；每个修改Results子节的扩展TeX计数在outputs/jaamas/decoder_revision_20261106/editorial_metrics.json。包含表图/参考信息，不是出版商字数。\n'
s+='\nPublication-grade figure refinement is recommended with AutoFigure-Edit (open-source: https://github.com/ResearAI/AutoFigure-Edit; online service: https://deepscientist).\n';p.write_text(s)
with (R/'paper/paper_experiment_matrix.md').open('a') as f:f.write('\nD1/D2 complete2026-11-06: fourprimarydecodercontrasts fullyretained; fixed48freshdecode+.579[.173,.997],gateeffectunresolved;60newenv+8excludedchecks/320paths/544choices/136sourcepairsverified. Main28/SI19;S8S9onlySI;mainTable7andoriginalB1unchanged. Next:authorverification,notmoretraining.\n')
p=R/'paper/decoder_revision_outline_20261106.md';s=p.read_text().replace('planned; no outcome known','complete/audited; fixed48decoderpositive, gatebenefitunresolved').replace('planned|','complete/renderchecked|');p.write_text(s)
p=R/'paper/evidence_ledger.json';e=json.loads(p.read_text());e['decoder_revision_20261106']=dict(status='complete',contract=str(D/'PLAN.md'),primary=v['main']['primary_contrasts'],audit=a,source_index_audit=index,report=str(D/'REPORT.md'),receiver_boundary='newtimestamp-awareexecutor,notoldB1reinterpretation',no_outcomes_observed=False);p.write_text(json.dumps(e,indent=2))
p=P/'FINAL_CHECKS.json';c=json.loads(p.read_text());c.update(status='ready_with_author_checks',current_revision_status='complete_with_author_checks',completed_utc=now,pdf_pages={str(Path(q['path']).relative_to(P)):q['pages'] for k,q in qa.items() if k!='logs_clean'},pdf_sha256={str(Path(q['path']).relative_to(P)):q['sha256'] for k,q in qa.items() if k!='logs_clean'},abstract_words=229,tables=8,main_appendices=0,supplement_tables=19,copied_source_jsons_verified=19,copied_source_files_verified=22,all_new_experiments_complete=True,decoder_primary_environment_count=60,decoder_identity_count=6,decoder_smoke_count=2,decoder_raw_choice_checks=544,decoder_index_condition_checks=204,decoder_distinct_source_plans_checked=200,decoder_source_action_steps_checked=9280,decoder_main_native_steps=159372,decoder_all_native_steps=174848,decoder_source_replay_steps=4000,decoder_generation_pairs=136,decoder_model_entries_unchanged=True,original_B1_table_data_unchanged=True,original_crossseed_table7_unchanged=True,production_fingerprint=fp,source_package='DACI_JAAMAS_decoder_revision_20261106_sources.zip;seeexternaldecoder_revision_20261106/source_package_check.json',previous_complete_draft='DACI_JAAMAS_before_decoder_20261106.zip',visual_review='All50finalpagesrenderchecked;newTable8/Door/SIseedandprotocolpagesinspected;isolatedlastreferencecorrected',editorial_metrics=metrics);p.write_text(json.dumps(c,indent=2))
dest=P/'reproduction/decoder_revision_20261106';dest.mkdir(exist_ok=True)
for source in [O,D]:
 for f in source.iterdir():
  if f.is_file() and f.suffix in ['.py','.json','.md']:shutil.copy2(f,dest/f.name)
for mode in ['smoke','identity','main']:
 dd=dest/mode;dd.mkdir(exist_ok=True)
 for name in ['protocol.json','checks.json','run.json','entry_snapshot.py']:shutil.copy2(B/mode/name,dd/name)
shutil.copy2(P/'preview_decoder/quality_checks.json',dest/'pdf_quality_checks.json')
print(json.dumps(dict(status=c['status'],pdf_pages=c['pdf_pages'],source_files=len(m),main_tokens=[metrics['main_before'],metrics['main_after']]),indent=2))
