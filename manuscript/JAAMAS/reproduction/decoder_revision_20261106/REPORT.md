# 独立生成时间解码对照与论文去重
状态：本轮修改与实验完成，待作者事实核实，未投递；2026-11-06。原30页完整源/PDF保存在 /home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS_before_decoder_20261106.zip；原B1研究、数据、效应与推断族保持不变。

## 已完成的编辑结构
原主文AppendixA–C不再编入主稿。逐种子原Table11–12仅在OnlineResource1的S8–S9保留，正文原Table7跨种子总结保留。Door关键图从附录移入主文Results，只保留95%any/0%terminal、~30/500成功步及坐标带/奖励解释；完整时长、first/last、最长成功段以及非零延迟探索表迁至SI。原候选执行与统计推断矩阵的唯一完整位置也设在SI，主文保留核心执行/推断与边界说明。旧结果/负结果未删除。

## 独立实验的冻结契约
仍为原始PickPlace actor0、全部冻结head/source/critic，不训练、不择优、不改生产代码。
新60环境（890010000区间）全部d16/32，与q16/32/64相交各10例；另6个d0和2个工程案例排除正式推断。四个事先定义的主对比、30000分层整环境bootstrap、98.75%区间；只有一个冻结训练系统，不能推广为跨训练种子结论。
Flegacy对新包仍使用d+elapsed；Ftime使用(a-g)+elapsed=elapsed；S的g=q，故两种解码相同。raw-prefix及本地动作不变。固定48步suffix条件是事先选好的已有动作选项，Ftime等价于已有raw48前缀，不是新控制算法。
PF的packet/nonmessage输入及权重固定，同一F选择不因解码改变；只更换执行语义。因此该实验比较的是改变后的接收策略，不是重新估计原B1的固定策略通道收益。critic依然拟合旧解码标签，未做时间匹配重新校准。已知且正确的消息生成时间、无新增刷新时延/费用均是假设。
强制所有环境执行固定source48而非仅选择器选source时执行，以免将未暴露于decoder变化的样本误作充分控制证据。

## 工程预检
两例native500步smoke通过，耗时40.8秒；6例d0一致性约110.8秒。源开关不改变本地特征/头预测、每例reject完整重放、直接source索引断言通过。独立只读审计8个排除案例，64个选择和24个固定48步索引计划均一致，奖励重构误差0。正式预算按约25–40分钟估计，一次只运行一个仿真作业。
bash_exec/artifact/memory工具在本会话不可用，使用本地PLAN/CHECKLIST/冻结JSON/exec日志记录，未新增子代理或付费服务。

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
所有28+19+2+1页均render检查，重点Table8与Door图、SI种子/生成时间/完整结果页另检查。四PDF无undefined/overfull/越页字符，字体全部嵌入。22份source-data文件（19JSON、3紧凑NPZ）与分析源完全一致；最终哈希见FINAL_CHECKS.json。新源ZIP为Advice_Audit_JAAMAS_decoder_revision_20261106_sources.zip；CRC、逐成员SHA及新解压构建记录另存本轮outputs目录，避免循环哈希。
源包是可编辑论文/审计入口包，native再生依赖完整本地模型与历史数据。未进行独立机器复现、人工作者认证、公开发布或投稿。作者事实/责任声明仍保留待核实。

## 编辑计数
同一pdftotext空白词元口径，主稿10869→10309；每个修改Results子节的扩展TeX计数在outputs/jaamas/decoder_revision_20261106/editorial_metrics.json。包含表图/参考信息，不是出版商字数。

Publication-grade figure refinement is recommended with AutoFigure-Edit (open-source: https://github.com/ResearAI/AutoFigure-Edit; online service: https://deepscientist).
