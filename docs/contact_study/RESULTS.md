# 外部接触任务扩展：阶段结果

能力验证complete/passed。不是视觉模型或官方benchmark成绩。正式延迟/闭环研究运行中，不能先声称建议有效。

|任务|公开专家成功|行为克隆local成功|训练规模/邻居数|
|---|---:|---:|---|
|push-v3|1.0|0.95|40/1|
|drawer-open-v3|1.0|1.0|40/1|

每个成功率分母为20个独立测试episode。push local为19/20，其他三格20/20。不是总体成功率确定为95%或100%。公共策略和BC共享示范来源；这里的独立指相对原reach手写控制家族，不是独立知识来源。

- 能力run：outputs/contact/capability_20261102T154658Z，UTC 2026-11-02T15:46:58.719830+00:00 至 2026-11-02T15:49:50.020246+00:00。

- 训练400个完整示范episode（每task200，全部保留）；两任务都在事前梯度预算的首档40达到validation门槛，k=1。不能说已经比较了所有200-demo模型或穷尽了本地控制能力。

- 独立环境Meta-World3.1.1 / MuJoCo3.3.0 / gymnasium1.2.3 / numpy2.5.4 / scipy1.18.1。原Fetch环境继续MuJoCo3.3.7，未降级。两个环境pip check均通过。

- 实际任务reward_function_version为v2，原生截止500、原成功信号、不人为缩短预算。延迟源为请求时单次动作保持1/3/12步，不称VLA多步动作块预测。

- 原套件59 passed，新增独立可选套件7 passed，均0失败/0跳过。两suite在不同venv运行，不能写成单一环境66项的输出；可选测试见tests_metaworld。

- 核心测试在接触后完整回滚MuJoCo积分状态、Python奖励/历史、RNG与model数组，逐步重放观测/reward一致；公共policy边界复制输入。

- 协议：CAPABILITY_PROTOCOL.md、DELAY_PROTOCOL.md。新delay-test种子与能力test互斥；新test不用于重训或改选择。

- 写作进入判断：paper/NATURE_SKILL_READINESS.md。nature-writing/shared/reviewer已按用户链接固定版本安装并读取；尚未把创新性未证实的材料标记为正式论文完成。

## 延迟/来源与闭环正式结果（已完成）

run：outputs/contact/delay_20261102T155844Z；complete/passed。每task validation8/test20独立episode，72/180请求-延迟行，3来源×4hold含reject；两任务504配对行、6048数组单元。单元不是独立样本。闭环每task180条记录，来自相同20个test初始种子，不能叫180个独立环境初始状态。

|任务|全来源dense oracle|廉价组合dense oracle|加入专家的portfolio增量及95%CI|
|---|---:|---:|---|
|push-v3|0.443583188|0.135355165|0.308228023 [0.1115100079635265, 0.5577336831495463]|
|drawer-open-v3|0.0165288902|0.0161163855|0.000412504673 [8.076715516149067e-05, 0.0009041517489584427]|

推物仍有相对声明廉价组合的oracle机会，所以不能把旧Fetch近零增量外推到所有任务。冻结规则的单次平均success gain仍未证实正；真正重复干预闭环同样未建立稳定优势。

|推物delay|BC/reject any-success|full固定规则|cheap固定规则|full相对BC95%配对CI|
|---|---:|---:|---:|---|
|0|85%|95%|80%|[-0.1, 0.3]|
|4|85%|90%|80%|[-0.1, 0.2]|
|12|85%|75%|85%|[-0.3, 0.1]|

开抽屉三个闭环方法均20/20曾成功且截止成功。validation将full/cheap都选为bc_arrival hold1，功能上与原无状态BC控制相同；其2.47e-17 validation回报微小差为浮点精度，保留原选择，不当成真实收益。零bootstrap区间并不凭样本证明专家无益；此处还存在策略函数相同的独立说明。

后处理明确标为探索性：推物5/60、开抽屉8/60请求组的raw12 dense gain跨delay翻号，分别嵌套在20个独立episode内。开抽屉的原样public损失随delay0→12减小，但public duration oracle减cheap-union oracle的均值由正转负。这个union含两个本地来源的事后最优选择，不能说某一个廉价策略单独胜过专家。完整三来源值和配对区间见outputs/contact/posthoc_20261103_v3。

## 新40回合的事前冻结复核（已完成）

协议CONFIRMATION_PROTOCOL.md；run outputs/contact/confirmation_20261102T162603Z，complete/passed。新seed97000–97039，共360配对行，保留全部；复用原BC、原任务、原专家、原执行族，不重训、不重选阈值、无按显著性追加样本。

定义D_d=public执行时长oracle减两个cheap来源联合oracle，先每episode对3请求平均。三个primary使用98.3333%完整episode bootstrap区间，family3 Bonferroni名义总alpha=.05。

|量|均值|区间|
|---|---:|---|
|source_gap_d0|0.0006402660682|[0.0001865612418339309, 0.0012156673623162733] (98.3333%)|
|source_gap_d12|-0.004264767251|[-0.007161363734375069, -0.0015980215928117335] (98.3333%)|
|source_gap_d12_minus_d0|-0.004905033319|[-0.007741407531858763, -0.0022873080727902097] (98.3333%)|
|raw12_d12_minus_d0|0.007642964374|[0.002535409642389361, 0.01259998688748908] (95.0000%)|

interaction_supported=True；strict_crossover_supported=True。次要raw对比单独95%，不冒充primary family保证。复核source gap从+0.0006403变为-0.0042648；原样public仍平均有害，但损失减少约0.007643。

## 实际支持的贡献

- 固定返回内容，真正推进等待中的本地运动，再在同原截止下比较来源/执行族，这是本工作的操作性测量对象。不是首次反事实、首次延迟修复或新理论。

- 在此接触任务中，原样损失减小与相对廉价组合优势下降可以同时发生，独立新回合支持预先指定的反转；不能只凭“修复更好”判断专家来源更必要。

- cheap union的hindsight选择能力是解释的一部分。复核delay12的public均值0.023101仍略高于两个单独cheap均值0.022913/0.014374，但cheap union为0.027366。这不是单个廉价控制器优越或专家知识无信息的证明。

- oracle机会、验证集固定规则和真实闭环收益分开；目前没有可靠的新控制算法效果。任务成功天花板、数据共享、synthetic delay和单动作hold的边界写入正文。

## 写作与检查

按用户指定nature-writing开始并完成英文内部初稿paper/manuscript.md，含4张已渲染检查的矢量/PNG图、完整方法/指标定义、引用与待补作者信息。主张足以组织为受限测量研究，但不是投稿就绪或新颖性已获同行认可。三个独立预写作评估在新接触结果前冻结，不能称审稿人已验证本轮新发现。

后处理首次因numpy.int64不能JSON编码退出1，原空目录/失败摘要保留；转Python int后v2/v3新目录运行退出0，新增序列化及统计界回归验证。最终可选suite9 passed/0failed/0skipped，原suite59 passed/0failed/0skipped分别在各自venv执行。native脚本截断动作提示为上游警告，不是数值不稳定或测试失败。
