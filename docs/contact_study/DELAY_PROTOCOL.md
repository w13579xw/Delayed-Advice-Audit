# 接触任务：来源替换与执行时长协议

2026-11-02，能力验证进行中、延迟study test尚未读取。能力资格未通过的任务保留并标明不进入，不能把资格选择当普遍任务结论。任务/本地模型随能力结果冻结，不重新调参。

## 与旧reach研究的差别

外部公开专家来自Meta-World，local为专家示范拟合的kNN BC。两者共享示范知识，非两种独立知识源；专家本身也很便宜，延迟为合成，不称实际昂贵VLA服务。建议是请求时一次固定4D动作，按zero-order hold执行1/3/12步，再回到原BC。不是12步VLA预测序列或新修复算法。

三来源：public_requested（请求时专家一次调用）、bc_requested（同请求BC动作）、bc_arrival（到达时新鲜BC动作，廉价对照）。仅前两来源应跨delay保持hash。bc_arrival有意使用当前观测，不能冒充固定同一请求候选。

原生500步、原目标/动态、不改第三方reward与成功条件，不加人工目标变化。请求checkpoint固定24/96/240，延迟0/4/12；maxdelay12+maxhold12=24，不重叠。只用可观测输入生成动作，分支未来只作标签。

## 数据、指标与选择

每个合格任务：validation seed85000–85007 / test86000–86019；drawer加10000。均与能力/训练数据互斥。每episode9个请求/延迟对；每对reject+3来源×3hold。所有分支共享到达快照和原500步截止，在同一env完整回滚physics/Python/RNG，单次干预后BC继续。无隐藏未来调参。

主要指标：原生success是否在500步内出现；同时报告原生截止success和native dense return/H。收益均相对同到达状态继续BC的分支；不同任务的dense reward不直接横向合并，不与Toy归一化距离效用比较。请求费用为共享沉没量，不重复扣减；不声称实际经济收益。

每个来源在validation预先选固定hold（可reject）；顺序是validation平均success gain，tie时native return，reject作为原点。cheap固定方案可在两个廉价来源之间选择；full固定方案可用三来源。冻结selection与time/hash后才采test。一次干预oracle仅作上界，不是模型效果。

主要描述：专家加入廉价修复族的success与dense portfolio正部增量、raw12与duration选择上界差、验证集固定方案的真实test收益。区间按完整episode配对bootstrap，不能将9个到达状态当独立episode。两任务/多个比较为探索性，不挑选有利seed/时延/指标得显著性。

此外在相同新test种子上运行固定请求闭环：reject、validation-full固定方案、validation-cheap固定方案；按delay分别比较成功和dense return。三个方法每回合相同3个请求，所有候选在到达后才可被采纳，等待期间BC继续。闭环不同历史的后续候选不能声称严格相同；离线配对才是同候选对照。

## 可否证结论和停机

- 若强廉价对照消除专家边际价值，只能在所测外部通道报告来源可替代；若专家仍有价值，修正旧Fetch结论的外推，而非掩盖反例。
- 若validation固定规则没有收益，不宣称学习或可部署方法有效。公开脚本能力通过也不等于固定动作通道通过。
- 当前这仍是测量贡献候选。须与When to Ask for Help等全文已讨论通道替换/修复的工作明确区别，不能把增加任务或普通hold长度选择包装成新算法。
- 先smoke一episode、少量checkpoint；正式采集不改代码、不覆盖旧run。预计双任务数分钟至十几分钟，按实际进度监测。新结果不能用于再选择本轮模型。

## 不改变主分析的精度说明（主run进行时追加）

固定主设计20个独立test episode。零discordance时，经验配对bootstrap可能给[0,0]，不能由此声称总体无效或等效。另做事后描述性Clopper-Pearson正discordance概率上界（每delay/两主对比，并报告固定12比较family Bonferroni上界）；净success提升不超过正discordance概率，因此该上界可用于说明当前无法排除多大改善。它不改原主指标/选择/seed/样本量，不根据结果追加样本找显著性。固定同一请求动作的dense符号翻转与跨delay曲线另标为探索性，不将请求数当独立episode。
