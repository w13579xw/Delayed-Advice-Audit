# 补充实验方案：训练随机性与开发集稳定性

## 1. Objective

Local campaign label: supplementary_campaign_20261104_v1。父结果：arrival_confirmation_20261104_v1；能力参照：act_budget_control_20261103_v1。用户要求投稿前补实验证据。选定写作结构为paper/visual_advice_audit/SUPPLEMENT_OUTLINE.md；主矩阵paper/paper_experiment_matrix.md。研究型追加，不承诺提升或录用。

问题RQ-S1：同一数据、预定容量及优化协议下，本地反馈与3k ACT的能力差是否跨训练种子稳定？问题RQ-S2：零模型选择是否依赖开发集划分及数值容差？这两项都不等于完整15k源/到达选择器多种子验证。

## 2. Boundary and comparability

S1固定200示范/task的180train/20validation划分、冻结编码器、原始损失/优化器/采样。新增training seeds1和2，seed0原模型保留。action heads两容量各20epochs，为代码一致性均训练，但只能评估原开发确定的Push c16/h128和Pick c32/h256；不按新增结果重选容量。ACT固定3000updates/seed/task，每250步离线masked L1选点，slice100固定。本地每native step反馈，两种策略观测频率/架构/历史不同，不作等计算架构排名。无新pose头需要；原主实验pose演员及15k源不改。

新环境池Push65400000..65400049，Pick65500000..65500049，MSAA0，native500，每个训练seed及两种策略共用50环境/task；总600轨迹、300000native steps，但只有100个独立环境seed。运行顺序先写入protocol，再训练/测试。工程smoke使用60400600/60500600，不进入正式结果。每次训练选点都在旧离线validation上完成，第一正式环境开启前冻结全部12个部署模型的权重/config哈希。不得按结果替换训练seed、删轨迹、缩N或扩大调参。

S1主要描述量为每task ACT3k−local的mean native reward差，先对3个训练seed平均，再对50共享环境平均。预先指定20000次crossed hierarchical bootstrap：同时重采样3个训练seed和50个episode，所有策略共享索引；两task分别97.5%近似区间。只有3个训练seed，区间是有限经验分布敏感性，不能当可靠训练总体推断。报告各seed/均值/范围、any-success和cutoff success；success的6个task×trainingseed配对差使用exact McNemar描述，并保留family6阈值.05/6。所有结果仍是补充分析，不重写已有主family。

S2只读取40个已有开发episode，按task各20个完整episode重划分100次（14train/6validation，随机种子2026110420），四个RBF输入/动作格使用原带宽/岭强度网格和零模型。每个split独立重新估计training normalization/距离scale，容差主诊断1e-12，另固定0/1e-14/1e-10/1e-8。报告零模型频率、选择频率和heldout开发reward差的分布，不将100重划分视为100个独立研究，也不读取500确认episode重新选模型。该诊断检验开发集不稳定性，不能证明增加数据后的学习曲线。

## 3. Slice plan

|exp_id|slice_id|priority|type|tier/placement|question|code/assets|
|---|---|---|---|---|---|---|
|V8a|training_replication_3k|1|robustness|main_optional/appendix|RQ-S1训练随机性|旧trainer/evaluator不改；新orchestrator仅outputs，所有资产已在本机|
|V9|development_resplits|2|sensitivity|appendix|RQ-S2开发集与容差|旧40episode，标准NumPy重拟合，无新simulator|

无新增数据集、收费API、真机或外部服务。不会把S1变成15k arrival复现。V8整体仍partial，V8a完成只是其中3k能力层。

## 4. Execution

先冻结本方案与JSON，再验证当前源码fp和资产。ACT做2updates smoke_only（不得用于能力），head做1-demo短overfit工程检查；旧seed0四个任务/角色模型各1episode检验MSAA0评价命令/trace/reward路径。正式新增8个训练run，GPU串行；全部模型冻结后600轨迹串行（一个simulator）。S2 CPU拟合可与GPU训练独立运行，不并行开native环境。

旧测量：4头两task各约140s，action-only预期约70s/task/seed；ACT3k约365s/task；local100episodes约519s、ACT100episodes约58s。合计约55–70分钟，单3090。每分钟检查run.json、日志更新和GPU；更长不因结果中断。ACT>900s或head>480s先检查资源后有据调整；NaN、数据错配、失去输入firewall、模型中途改变则判失败保留日志，用新run修复，不能假成功。专用bash_exec/artifact/memory接口在当前工具清单不可用，继续已验证exec_command/write_stdin与本地JSON/日志记录；无伪造工具artifact。

## 5. Reporting

S1同号各seed和近似区间可增强本批模型能力关系的描述，异号/CI跨0则报告训练敏感；S2分割频率变化说明不能泛化零模型普适性。中性、矛盾与提升均完整保留。每项结束更新CHECKLIST、RUN/RESULT、主矩阵、evidence ledger及选定outline结果表。聚合后再决定正文/补充材料，不更改冻结历史确认。

## 6. Revision log

2026-11-04：原TRAINING_REPLICATION_PLAN的3k训练协议沿用；正式评测改为上述全新50seed/task和MSAA0，修订在所有新增训练和新测试前冻结。旧610/611 MSAA4结果不合并。
