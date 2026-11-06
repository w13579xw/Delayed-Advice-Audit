# 检查点协议 × 可观测时钟：冻结实验计划

冻结日期：2026-11-02。用户已授权继续未完成实验，路线仍为可信评测/负结果论文。本计划在本轮正式test采集前写入；旧v2结果和调试smoke不作为新test。

## 问题与预先指定对照

v2中45.45%的固定闭环请求晚于采集最大时刻，且模型没有剩余预算输入。因此不能将coverage变化直接归因于自适应采纳。本轮分离两个因素：

|因素|对照|处理|
|---|---|---|
|采集检查点|early：0/10/20/30/40|aligned：0/24/48/72，各延迟仅保留截止前可到达的请求|
|时钟输入|no_clock：36维、末两维置0|with_clock：同36维，末两维为剩余步数/原horizon、原horizon/80|

同协议/训练seed下，两组随机投影、bias、bootstrap随机流和模型容量匹配；当前策略、专家、任务、成功标准、80步原截止、成本0.01、延迟{0,4,12}、动作块12均不改。时钟只来自部署可观测调度器，不含未来目标/分叉信息。

主要对照事前固定：aligned_with_clock减aligned_no_clock；aligned_with_clock减early_with_clock。指标：deadline预测MAE、按episode联合下界coverage、单次候选收益，以及闭环效用差。MAE下降较好；coverage和候选/闭环效用差上升较好。CI跨零则不作明确改善结论。

early每episode15标签，aligned每episode11标签；采集协议因素同时改变检查点分布与样本数，不宣称只改变时刻。aligned的晚期短窗口允许缩短为4/8/20步，显式存储short_window_steps；原截止标签仍使用原80步分母。

## 数据与种子

- 每后端early/aligned各采集80 episode，seed5000–5079。两个协议共享初始seed和外生扰动以便比较；四路固定48/12/12/8，calibration只做残差校准。
- 每协议/特征训练seed{0,1,2,3,4}，总计20个head/后端。所有seed共享同一采集与校准数据；不能称五个独立数据重复。
- 新独立offline test：seed6000–6049，50个episode；shift=.15及clean shift=0，固定请求协议，每场景550标签。各模型在同一候选集评测。
- 闭环：使用上述预先指定的前12个test episode，各延迟/场景/模型相同。aligned下两种特征各五seed，比较veto_only和veto_calibrated；early模型仅作离线因子对照，不冒充已跑闭环。
- 闭环基线local/reject/always/TTL/drift共用seed。TTL/drift沿用v2 validation选择的全拒绝阈值-1，不在新test调优。新的主对照是匹配请求的reject，而非故意弱化的TTL。
- clean/shift场景共享seed，分场景报告，不能当独立任务族重复。closed-loop只有12个episode，不能写成50回合闭环实验。

## 统计与解释

离线先按完整episode聚合，再构造5模型seed×50环境episode矩阵；闭环为5×12矩阵。对照先作配对差，再做交叉bootstrap。报告各seed均值与95% CI，不把250个模型/episode组合当独立样本。该CI条件于共享采集/校准数据，不涵盖任务总体或重新采集的不确定性。

仅aligned/shifted的离线local-only行为、检查点与扰动协议相符；clean是扰动移除对照，闭环接受会改变状态分布，二者不继承离线保证。高coverage还须同时展示接受率、有害率、漏失率和效用，不能用全拒绝包装成功。

正收益oracle只用于离线审计，未用作部署特征或调参。所有反事实分支在原任务截止结束。未出现收益提升或coverage改善同样记录；不换种子、删失败轨迹或增大offset找正结果。

## 执行与预算

配置：`configs/alignment_study_v1.json`；入口：`scripts/alignment_study.py`；核验/统计：`scripts/analyze_alignment.py`。目录必须全新；toy完成后串行Fetch；每阶段独立日志、退出码、协议/源码snapshot。

依据已测Fetch1200条完整截止标签约211秒和v2全流程约512秒，本轮3180标签/后端、20个小型head、1800条闭环记录/后端，预估Fetch约20–30分钟CPU时间；实际耗时以run.json为准，不是GPU推理预算。没有大模型下载、收费API、真机或14周矩阵。

## 明确保留的未完成范围

独立建议源、非reach操纵任务、视觉策略、真实推理延迟、50回合/更多任务的完整闭环主实验仍未由本计划实现。本轮结果只能决定这些方向的优先级，不代表论文已达到投稿验收。
