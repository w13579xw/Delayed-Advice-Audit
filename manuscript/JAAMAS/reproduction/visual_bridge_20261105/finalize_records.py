from pathlib import Path
import json,re,hashlib,shutil,datetime,sys
R=Path('/home/ez/DACI/code');P=Path('/home/ez/DACI/Springer/DACI_JAAMAS');D=R/'docs/learned_advice/visual_bridge_revision_20261105';O=R/'outputs/jaamas/bridge_revision_20261105';B=R/'outputs/jaamas/visual_channel_bridge_20261105_v1'
sys.path.insert(0,str(R/'scripts'));from server_run import fingerprint
fp=fingerprint();assert fp=='48ffd6ffff1e3eeb5cc15a475888a844c846ead878292e6137c7f171c5246221'
assert json.loads((O/'post_queue_complete.json').read_text())['status']=='complete'
qa=json.loads((P/'preview_bridge/quality_checks.json').read_text());metrics=json.loads((O/'compression_metrics.json').read_text());v=json.loads((P/'source_data/visual_channel_analysis.json').read_text());audit=json.loads((P/'source_data/visual_channel_audit.json').read_text())
manifest=json.loads((P/'source_data/manifest.json').read_text())
for x in manifest:assert (P/'source_data'/x['file']).read_bytes()==Path(x['source']).read_bytes() and hashlib.sha256((P/'source_data'/x['file']).read_bytes()).hexdigest()==x['sha256']
for x in ['svjour3.cls','svglov3.clo','spmpsci.bst']:assert (P/x).read_bytes()==(P.parent/x).read_bytes()
# Reconfirm all frozen model hashes after independent native regeneration.
proto=json.loads((B/'main/protocol.json').read_text())
for f,h in proto['model_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=D/'REPORT.md';s=p.read_text();s=s.replace('状态：执行中；','状态：本轮B1–B4完成，待作者事实核实，未投递；')
s+='''
## B1正式完成结果
原始PickPlace seed0视觉ACT系统冻结，不新增训练或择优。主实验60个全非零延迟环境，另外6个零延迟一致性检查和2个工程样本不计入推断。原始PC零模型三通道均拒绝；非零PF刷新改变10/60个选项和28/60条动作序列。LF不改变选项但19/60条动作改变。
- F−S = −0.142777，98.3333%区间[−0.493178,+0.228476]：正向刷新收益未建立，也未证明等效。
- S−E = −0.517973，[−1.155691,+0.052102]：此可用性对比仍不确定。
- F−E = −0.660750，[−1.248585,−0.083647]：在这套接收器/固定解码器/回退规则下，fresh低于reject回退。
- S/F/E any成功率83.3%/80.0%/88.3%，terminal60.0%/61.7%/58.3%，仅描述，不能把不同端点合并为全面优劣。

主实验177条独特执行计划、107844个实际native步骤（包括每例完整reject重放）；含检查总计199条路径、121868步。独立消息来源审计额外执行4000个前缀物理步骤，逐位重生成136组请求/到达packet、特征和head输出。总816个保存决策与全部原生奖励均核验；模型、执行入口及生产指纹不变。全部60正式样本保留，未进行结果导向调参。统计为预声明3个奖励对比、30000次六格等权整环境配对bootstrap；无actor种子重采样，只有一个冻结系统。

这项桥接把原始视觉设置与真实通道对比联系起来，但PC和PF仍是不同接收器，不能把+.735、PC结构拒绝与PF通道效应称为同一策略同一量。三类估计不可互换是论文定位，不是延迟专家建议提高机器人性能的证据。

## 最终编辑与交付检查
主稿30页（原40页），正文/结论截至22页，声明延至23页，参考文献23–25页，必要附录26–30页；独立Online Resource1为15页。Information Sheet2页、cover letter1页。未缩小模板字号/页边距，三个原始class/style文件逐字节保持不变。摘要211词、5关键词、31条正式文献、Algorithm1；主稿3幅正文图+1幅附录图，补充材料2幅图。
同一pdftotext空白词元计数从14600降到10869，下降25.55%；包括表格/引文/附录，不是出版商字数。详细结果是迁移到补充材料而非删除。AI方法披露一次，文末交叉引用；原始推断族和补充子组身份保留。

四份PDF均编译成功；全部30+15+2+1页已通过拼图检查，主要通道表、非零延迟表、Door图及种子表另作放大检查。修正末页孤立表、短表跨页尾行及Information Sheet强制空页。无undefined引用、overfull或越页字符，字体全部嵌入。18份来源文件（16JSON、2紧凑NPZ）与原始分析/曲线文件精确一致。sourcezip逐成员SHA及CRC记录另存outputs/jaamas/bridge_revision_20261105/source_package_check.json。
英文可编辑源文件和四PDF均在 /home/ez/DACI/Springer/DACI_JAAMAS。新归档名DACI_JAAMAS_bridge_revision_20261105_sources.zip；旧40页版本仍保留。源包不是包含完整预训练权重/全部历史trace的独立再生发布；完整工作区仍是native重现的依赖。作者姓名、贡献、经费、利益冲突、既往发表/并投、伦理适用和正式公开归档仍需人类确认。未进行外部投稿、付费服务或发布，也不保证录用。
'''
p.write_text(s)
# Mark all campaign work completed.
p=D/'CHECKLIST.md';p.write_text(p.read_text().replace('- [ ]','- [x]'))
# Research ledger and experiment matrix retain earlier rounds.
p=R/'paper/paper_experiment_matrix.md'
with p.open('a') as h:h.write('\n## Visual bridge and compression, completed 2026-11-05\n|B1|Original frozen PickPlace S/F/E channels|Complete/native+independent audit|60nonzeroenv,6identity+2smokeexcluded;3primary98.3333%CI;10choices/28actionschanged;F-Sunresolved|\n|B2|Nonzero state-delay uncertainty|Complete/exploratory|40env/task;all16/64subsetsretained;no posthocconfirmatoryclaim|\n|B3|Door temporal success and reward|Complete/descriptive|All60env/5critics;~30successsteps/500;flaglossnotidentifiedreclosure|\n|B4|Compactmain+OnlineResource1|Complete/authorchecks|Main40→30pages;SI15;Info2;letter1;all48pagesrenderchecked|\n')
p=R/'paper/evidence_ledger.json';e=json.loads(p.read_text());e['visual_bridge_revision_20261105']=dict(status='complete',scope='onefrozenPickPlacesystem;notprivatemessagevalue/generalization/safety',main_envs=60,identity_envs=6,smoke_envs=2,primary=v['main']['primary_contrasts'],audit=audit,report=str(D/'REPORT.md'),manuscript=str(P/'DACI_JAAMAS_manuscript.pdf'));p.write_text(json.dumps(e,indent=2))
p=R/'paper/figures/figure_catalog.json';c=json.loads(p.read_text());c.append(dict(id='jaamas_visual_bridge_20261105_door_temporal',surface_class='appendix',display_number=4,claim='Native Door success is transient despite95%ever-success and0%terminal;different native reward endpoint',sources=[str(R/'outputs/jaamas/state_temporal_20261105_v1/curves.npz')],generator=str(O/'plot_temporal.py'),exports=[str(P/'figures'/('door_temporal.'+ext)) for ext in ['pdf','svg','png']],qa='FinalPNG andactualPDFp29inspected;quotedstylehexcompatibilityfix;allchannels/60env/5critics;no cherry-pick/timewiseinference',main_curve_state_figures_moved_to='OnlineResource1'));p.write_text(json.dumps(c,indent=2))
# Compact readiness and source/QA evidence.
checks=json.loads((P/'FINAL_CHECKS.json').read_text());checks.update(status='ready_with_author_checks',current_revision_status='complete_with_author_checks',completed_utc=now,pdf_pages={str(Path(q['path']).relative_to(P)):q['pages'] for k,q in qa.items() if k!='logs_clean'},pdf_sha256={str(Path(q['path']).relative_to(P)):q['sha256'] for k,q in qa.items() if k!='logs_clean'},abstract_words=211,figures=4,supplement_figures=2,tables=12,supplement_tables=15,copied_source_jsons_verified=16,copied_source_files_verified=18,all_new_experiments_complete=True,visual_defined_channel_value_tested=True,visual_channel_primary_environments=60,visual_channel_identity_environments=6,visual_channel_smoke_environments=2,visual_channel_main_paths=177,visual_channel_main_native_steps=107844,visual_channel_all_saved_decisions=816,visual_channel_regenerated_source_pairs=136,visual_channel_independent_prefix_native_steps=4000,production_fingerprint=fp,source_package='DACI_JAAMAS_bridge_revision_20261105_sources.zip;externalCRC/SHArecordinbridge_revision_20261105/source_package_check.json',visual_review='All30main/15SI/2Information/1letterpagesrender-inspected;criticalchannel/temporal/seedpagesenlarged;layoutrefined',compression_metrics=metrics,upload_ready=False,author_approval_obtained=False,public_release_verified=False);(P/'FINAL_CHECKS.json').write_text(json.dumps(checks,indent=2))
p=P/'AUTHOR_CHECKLIST.md';s=p.read_text().replace('Native channel values were tested in the state/public-source regime; visual experiments report pipeline contrasts and finite-pool diagnostics.','Native channel values were tested in three state/public-source tasks and one frozen original PickPlace visual system; the latter uses 60 nonzero-delay environments and a fixed delay-indexed decoder. Its PC null and PF nonzero receiver are distinct.');p.write_text(s)
# Copy the exact new campaign entries/protocols and report; native generation still needs the full workspace.
dest=P/'reproduction/visual_bridge_20261105';dest.mkdir(exist_ok=True)
for f in O.iterdir():
 if f.is_file() and f.suffix in ['.py','.json','.mplstyle']:shutil.copy2(f,dest/f.name)
for f in D.iterdir():
 if f.is_file() and f.suffix=='.md':shutil.copy2(f,dest/f.name)
for mode in ['main','identity','smoke']:
 dd=dest/mode;dd.mkdir(exist_ok=True)
 for name in ['protocol.json','checks.json','run.json','entry_snapshot.py']:shutil.copy2(B/mode/name,dd/name)
shutil.copy2(P/'preview_bridge/quality_checks.json',dest/'pdf_quality_checks.json')
print(json.dumps(dict(status='complete_with_author_checks',fp=fp,pdf_pages=checks['pdf_pages'],source_files=len(manifest)),indent=2))
