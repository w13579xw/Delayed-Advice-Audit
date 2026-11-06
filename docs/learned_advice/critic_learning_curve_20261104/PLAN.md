# Critic 数据量学习曲线：冻结方案

## Objective / selected outline

Local campaign label critic_learning_curve_20261104_v1。父证据为500episode arrival_confirmation_20261104_v1和已完成supplementary_campaign_20261104。用户授权继续研究。选定paper/visual_advice_audit/LEARNING_CURVE_OUTLINE.md；实验矩阵V11a/V11b/V11c。

RQ-L1：增加独立critic训练episode能否改善固定协议下的部署收益？RQ-L2：原packet-input表面优势是否随训练量变化，零模型解释是否保留？RQ-L3：扩大动作集是否继续引入损害？本轮不提出新kernel架构，不保证正效果。

## Boundary and comparison contract

保持trainingseed0的四个视觉演员、15k ACT源、MSAA0图像、native500、单次强制query、12个既定repair选项、损失native_reward_gain/10、RBF网格带宽.25/1/4与ridge.01/.1/1/10、zero候选、1e-12全局最优容差及正增益采纳阈值。只改变critic独立训练episode数量n=7/14/28。每task同一新28episode训练pool，事前随机rank使三个训练子集嵌套；同一12episode验证pool用于各N和四个策略格。normalization、距离scale、dual和固定动作选择只能使用对应training数据，选点只使用validation。所有模型在正式test前冻结，不读旧500确认结果选新模型。

新开发pool：Push66400000..66400039、Pick66500000..66500039，每task28train/12validation；q均匀16/32/64，d均匀0/16/32，事前调度。新正式test：Push66600000..66600099、Pick66700000..66700099，各100独立成对episode；所有N/四格/refs共享同一请求packet、到达snapshot和prefix，各唯一选中动作计划只执行一次并事前打乱branch顺序。包括prefix已成功episode，不换seed或删失败。

每task12学习策略（3N×2inputs×2actionsets），refs=reject、elapsed_suffix48、n28训练所选fixedcheap/fixedfull。所有选择和输入hash先写入，再执行future分支。source同步计算+合成native-step延迟，不是实测异步或净query价值。

新cohort与旧研究的验证数/调度不同，不能把旧14/6与本轮28/12差异全归因训练量。学习曲线的可比性只在本轮同一validation/test下的n7/14/28之间。只有一个嵌套训练抽样和固定actorseed0，不能泛化所有训练集或模型。

## Primary inference

wholeepisode mean native reward为主。每task四个事前对比：
1. n28 packet/full−reject；
2. n28 packet/full−n14 packet/full；
3. n28 packet/cheap−n28 localinput/cheap；
4. n28 packet/full−n28 packet/cheap。
共8个主对比，50000次wholeepisode pairedbootstrap，各99.375%近似区间（Bonferroni8，总alpha.05），固定rng2026110427。报告全部区间、正/负/零计数(tol1e-10)、极值、选中zero与acceptance频率。它是近似推断，没有有限样本安全保证。delta=.25仍仅为研究定义的奖励全幅2.5%参照，不是已验证实际效用门槛。

判读：n28相对reject和相对n14的区间都在0上才支持该task的有益数据增量；仅相对n14改善而仍低于reject属于减害；区间跨0为不确定；zero结构性拒绝必须明确，不能把正特征格差当运行时专家信息价值。n7/n14其他曲线点及success/cutoff/fixedrefs均次要描述，不另找显著切片。不按partial效果扩N或网格。

## Slice plan / assets

|exp_id|slice_id|class|RQ/design|section_id/item_id|claim_links|paper_role|
|---|---|---|---|---|---|---|
|V11a|new_critic_development|auxiliary|required new80episode labels|critic-data/L1-data|C-L1|appendix|
|V11b|nested_kernel_fitting|auxiliary|samevalidation3N×fourcells|critic-data/L2-models|C-L1,C-L2|appendix|
|V11c|prospective_learning_curve|claim-carrying|200newpairednativeepisodes|critic-data/L3-test|C-L1,C-L2,C-L3|main_optional|

原模型、数据、环境和无费用本地runtime均已存在，无新数据集或外部API。代码仅新增outputs研究入口；scripts/visual_bundle.py有一处迁移兼容修复，按解析后的真实路径匹配演员，已验证仍选Pushhead0/Pickhead1，旧源码备份保留；新fp48ffd6ffff1e3eeb5cc15a475888a844c846ead878292e6137c7f171c5246221。

## Execution / monitoring

先跑2episode开发smoke60400800/60500800(q32,d16)，包括acceptedsource比对和reject bitwise replay；用新fitter在旧40开发数据n14/val6精确重现原8格作为工程parity，不用旧数据训练新正式模型。再80episode正式采集、3N拟合、12格×12val×2task的运行时一致性检查、4个边界test smoke60400900/901与60500900/901(q16d0/q64d32)、最后固定200新test。正式模型与所有normalization JSON/geometry/dual/actor hash均在test前冻结。

旧40episode全12arm采集1775秒，估计新80约60分钟；200test按选中计划去重约45–100分钟，整体2–3小时。先报告实际smoke耗时再启动正式queue，不因结果不理想停止。一个GPU/一个native作业，OMP2/BLAS1/MUJOCO_GL=egl，保持>20GB磁盘余量。日志与run.json每分钟有真实进度；健康运行则等待，不重复启动。若NaN/hash漂移/火墙失败/回放不一致则停止并保留失败，修复后新run。环境专用bash_exec/artifact/memory接口缺失，沿已验证exec_command/write_stdin及本地record，不伪造artifact。

## Reporting / stopping

每slice检查status/exit/checks、原始轨迹reward、schedule与模型hash。完成后同步矩阵、outline、ledger、CURRENT_STATUS与HANDOFF。所有正/负/不确定结果完整保留；未知结果不写论文成功故事。本轮固定采样完成即汇总，不沿同test继续优化。期刊待定，通用Markdown，不套MDPI或投稿。

## Revision log

2026-11-04：首次冻结。所选数据规模是有界的7/14/28学习曲线；未承诺以28episode排除所有small-data解释。下一unchecked项见CHECKLIST.md。
