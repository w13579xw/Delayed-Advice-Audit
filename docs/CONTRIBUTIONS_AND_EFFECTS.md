> 2026-11-04最新视觉确认与贡献以[本轮独立结果](learned_advice/CONFIRMED_RESULTS_20261104.md)为准。本文下方是早期state/alignment阶段历史记录。

# 候选创新点与模型效果清单

2026-11-03视觉阶段最新：同RGB+robot4的专用视觉小模型在独立50回合资格上，Push50/50对原ACT42/50、PickPlace44/50对35/50（+16/+18个百分点，单训练seed0）；这不是新gate效果，也未控制任务训练数据。新的同180/20示范ACT适配和240回合源开发已完成，新的独立50回合匹配对照正在运行。完整受限解释与可复查路径见[视觉阶段结果](learned_advice/VISUAL_RESULTS.md)。此前contact核心40回合已用snapshot-v3复算，严格反转保留；其余旧contact数组仍需逐项复核，见[修正状态](../paper/REPRODUCIBILITY_ALERT_20261103.md)。

2026-11-03最新：外部接触任务及新40回合复核已完成，新增贡献收敛为固定返回内容的到达干预与匹配廉价组合归因；[完整结果](contact_study/RESULTS.md)、[论文初稿](../paper/manuscript.md)。这是受限测量研究，非有效新控制模型。

最新阶段已增加“修复收益是否依赖专家来源”的问题、强廉价源、实际学习选择器及误差归因；见[当前科学定位](SCIENTIFIC_POSITION_AND_NEW_DIRECTION.md)和[新增实验结果](REPAIR_VALUE_RESULTS.md)。下文保留早期阶段结果；最新测试为59通过，新增选择器仍未证实有效。

更新：2026-11-02T17:22:04.863583+08:00。当前定位为评测/负结果研究；不能把已有技术组合直接写成首创。

## 创新点如何表述

|候选论文贡献|已实现/证据|新颖性边界|
|---|---|---|
|对固定同一建议进行到达时间干预，测量收益曲线/有效集合|同请求plan hash跨延迟不变，共享起点/未来扰动；v2截止收益翻转Toy122/400、Fetch55/400组|这是研究对象与测量设计，非“首次反事实干预”；需进一步与最近邻逐项对照|
|分离“有正收益候选”和“求助值得付费”|短窗口/原截止效用同时审计；一次请求oracle收益与成本分解；保留零成本例外|原截止归一化与oracle不等式是测量工具/基本代数，不宣称新理论|
|受控分离检查点错配与剩余预算表示不足|本轮已实现early/aligned × no_clock/with_clock，五训练seed、新50 episode测试池；双后端已完成，完整结果见ALIGNMENT_RESULTS.md|可作为实证诊断贡献；仅加入时钟或四路划分不构成算法新颖性|
|给出门控何时不值得复杂化的可复查证据|强制全拒绝、验证集阈值、同候选离线消融、闭环有害/漏失/coverage、成本敏感性|结论仅覆盖已测控制器与任务；不能泛称慢模块或VLA无效|

随机特征岭回归、bootstrap ensemble、split conformal、按episode划分/重采样均为已有技术；当前没有已经证实的新网络架构或新的闭环安全定理。最可信的潜在贡献在测量问题与系统性实证，而不是给普通回归模型重新命名。

最近邻：[When to Ask for Help](https://arxiv.org/abs/2610.21160)已有慢模块价值/成本调度研究；[RESCUE](https://arxiv.org/abs/2610.13516)已有反事实干预可恢复性评价；[ForkCast](https://arxiv.org/abs/2610.18650)已有候选条件反事实未来比较。上述边界基于2026-11-02重新核对的一手论文，不保证穷尽性新颖性。

## v2模型效果（历史基线，保持不变）

每后端独立offline test为8个episode、120候选；一个训练seed。目标为原episode截止收益。MAE越小越好，候选收益是单次干预后local继续，不能当闭环成功率。

|模型|Toy MAE|Fetch MAE|Toy接受候选/120|Fetch接受候选/120|Toy平均候选收益|Fetch平均候选收益|
|---|---:|---:|---:|---:|---:|---:|
|完整随机特征ensemble|0.0069495|0.0042438|7|1|-0.00019798|-0.00001304|
|线性ridge|0.0108597|0.0045139|4|0|-0.00050961|0.00000000|
|去显式年龄|0.0093454|0.0045926|5|3|-0.00041253|-0.00009443|
|去候选摘要|0.0078178|0.0042594|3|2|-0.00017039|-0.00001926|
|仅当前状态|0.0118976|0.0049022|3|0|0.00003842|0.00000000|

上述MAE最优不等于干预收益最优；不根据小样本test挑选current_only等偶然较好变体。完整模型的校准版本在v2 test全部拒绝，offline episode coverage Toy0.875、Fetch1.0。

闭环v2：校准模型在两场景/三延迟全部拒绝；成本0.01下相对local差-0.04（d0/4）或-0.03（d12），与强制请求全拒绝等效。Fetch未校准模型在d4相对全拒绝约+0.000132，但CI含0，不能声称可靠提升。

v2绝对收益规模：Toy/Fetch候选池单次oracle截止收益均值约0.00011668/0.00002534，远小于设定成本0.01；只约束当前候选分布下的单次强制查询，不约束自适应多次策略。Toy干净、零延迟、零成本always有约+0.001547收益，故不能泛称所有建议无益。

## 本轮继续实验（现已完成）

协议见[冻结计划](ALIGNMENT_PREREGISTRATION.md)。实现和smoke已通过；当前48项测试通过，包含真实Fetch。正式run已完成，见本文件后续小节及ALIGNMENT_RESULTS.md；旧v2保持不变。

本轮只完成优先级最高的检查点/时钟与五模型seed控制实验。独立建议源、非reach任务、视觉策略、真实异步trace、更多闭环测试episode仍是后续缺口，不能改写成已完成。

## Toy先行结果（阶段记录，完整双后端结果见后文）

Toy目录：`outputs/alignment/toy_20261102T091908Z`；status=complete、checks=passed；结果时点2026-11-02T17:26:36.886047+08:00。下面是冻结测试结果，不用来修改正在运行的Fetch配置。

每格为五训练seed × 同一50个新episode的平均；candidate收益不是闭环效用。

|场景|训练协议/特征|MAE|offline episode coverage|点估计接受率|校准接受率|
|---|---|---:|---:|---:|---:|
|shifted|early_no_clock|0.0172331|0.732|3.455%|0.000%|
|shifted|early_with_clock|0.0175923|0.720|5.600%|0.000%|
|shifted|aligned_no_clock|0.0292178|0.844|18.873%|0.327%|
|shifted|aligned_with_clock|0.0299259|0.852|22.691%|0.473%|
|clean|early_no_clock|0.0071757|0.820|2.000%|0.000%|
|clean|early_with_clock|0.0075463|0.820|3.673%|0.000%|
|clean|aligned_no_clock|0.0157104|0.952|16.982%|0.218%|
|clean|aligned_with_clock|0.0163467|0.956|19.636%|0.364%|

Toy扰动场景：matched+clock相对early+clock的coverage差+0.132，95%交叉bootstrap CI[0.012,0.256]，但MAE增加约0.012334。matched中clock相对no_clock的MAE差约+0.000708，CI跨0，未证实时钟降低误差。early/matched除检查点外样本数也不同，不把该对照说成纯时刻效应。
同80个初始episode：early没有terminal_success_gain<0标签，aligned有5条，均来自request48。此为采集协议漏测晚期风险的直接案例，不代表已经证明任何通用规律。

Toy闭环：点估计gate在延迟4和12仍出现有害采纳；加入clock没有稳定改善。calibrated gate几乎全拒绝，但仍有少数有害采纳，不能称安全。完整五seed/十二episode条件表见该run的analysis.md/json。

### 可重放的机制案例

`outputs/alignment/case_20261102T091908Z`：Toy seed5005，第48步请求，第60步到达，严格复现数据标签-1.0177888561008375。目标在46步已变化，在等待/执行期间未再次变化；local等待时已接近目标，旧动作把它推离。接受分支77步才重入容差，截止80步只hold4步，未达原5步标准。最终距离约0.00462，不能将此称为碰撞或现实安全事件。该例按负标签筛选，仅作说明，非额外独立测试样本。

## 本轮最终状态

两个后端均complete/passed，40个模型、6360条双窗口标签、3600条共享seed评测记录；48项测试通过。五模型seed、50个offline-test episode和12个closed-loop episode分别报告，不合并充样本数。创新/效果的最新完整清单见[对齐实验结果](ALIGNMENT_RESULTS.md)，原始索引`outputs/alignment/summary_20261102T091908Z/combined.json`。

## 最新模型效果摘要：五训练seed、50个新episode

以下为shifted场景，MAE越低越好；coverage是episode联合下界覆盖，不是分类准确率。每格为五模型seed与同一50环境episode的平均，训练/校准数据共享。

|采集与时钟|Toy MAE|Toy coverage|Fetch MAE|Fetch coverage|
|---|---:|---:|---:|---:|
|early_no_clock|0.0172331|0.732|0.0057880|0.828|
|early_with_clock|0.0175923|0.720|0.0061494|0.872|
|aligned_no_clock|0.0292178|0.844|0.0052606|0.748|
|aligned_with_clock|0.0299259|0.852|0.0051904|0.772|

with_clock相对no_clock，在aligned/shifted下的MAE差：Toy+0.0007081（95% CI跨0），Fetch-0.0000701（95% CI跨0），没有一致的时钟收益证据。对齐相对early时，Toy覆盖提高但MAE变大；Fetch的MAE改善但覆盖下降。不能把某一指标的单方向变化描述为整体模型更有效。
闭环主比较使用相同请求的reject，成本抵消。模型仍没有跨后端/场景/延迟的一致优势；Toy干净零延迟有小幅正收益，多个较大延迟条件出现负收益。完整效用差、接受/有害/漏失计数和CI见ALIGNMENT_RESULTS.md，不宣称模型已有效或安全。
