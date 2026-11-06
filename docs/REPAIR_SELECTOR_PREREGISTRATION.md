# 从oracle机会到可部署输入的选择器：第二阶段冻结

日期：2026-11-02。在repair_value_pilot_v1的独立12回合test中，两个后端均通过事前规定的启发式机会筛选规则。它只说明指定修复族/廉价来源下存在有待学习的机会，不能称模型已提升。

本阶段使用全新数据，检验是否能真正学到该机会；不会在先行pilot test上训练/调参。

## 研究对象与强对照

固定已返回的建议，选择reject，或选择来源及raw/prefix3/suffix/反馈跟踪。修复算子本身是已知控制思路，不宣称新颖。当前原型是按选项分别拟合的随机特征回归器，亦不宣称新网络架构。

除了原local增益6的名义展开，**在新test前额外固定增益12的廉价名义展开来源**，避免把专家较大的控制增益误当不可替代的信息。原环境中的local仍保持增益6；未削弱或改变基础策略。两廉价来源同样接受所有修复操作，cheap-only选择器可以在它们之间选择。

先行pilot只有一个廉价来源，因此其oracle差不能被描述为超越所有本地策略的“信息价值”。本阶段也只是相对声明的廉价控制族，不是信息论互信息或对任意本地算法的最优性结论。

## 冻结数据与模型

配置：`configs/repair_selector_study_v1.json`。

- train72000–72059：60 episode；validation72100–72114：15；calibration72200–72214：15；独立test73000–73029：30。四池互斥，且不使用旧pilot或alignment的种子。
- 每episode11个固定请求/合法延迟对；每对3来源×5操作的反事实方案，共享到达快照、公共未来与原80步截止。单次操作后回到相同local。
- 5个模型seed0–4，每seed12个选项回归head；输入只含请求/到达观测、候选摘要、已知age和剩余预算。标签、未来分支、隐藏扰动不进入choose接口。
- Head固定为64随机特征、5 bootstrap成员、ridge1。选项score=预测均值-0.5ensemble std；没有测试集超参数搜索、目标裁剪或评估标准变动。
- validation只估计各选项残差RMSE尺度（floor1e-4）和选择最强固定来源/操作；calibration按完整episode对所有合法选项×请求×延迟取标准化最大过估计残差，alpha=.2。
- full与cheap-only都报告point/calibrated。cheap-only使用其自身选项集合的校准offset，不故意用full更宽的界削弱它。过期suffix/track和零延迟重复suffix在选择时排除，此规则不读标签。
- 所有head训练并冻结后才开始test采集。训练/校准共用一次数据抽样，模型seed不是五份独立数据。

## 主比较和统计

主比较：`full learned selector - cheap-only learned selector` 的真实held-out单次候选收益差，先在同一模型seed/同一episode配对，再按5模型seed×30环境episode交叉bootstrap。point/calibrated分开，不挑选某个seed报告。

同时报告：相对reject、相对validation选定固定控制、专家使用次数、有害次数、联合下界coverage及oracle上界。oracle从真实标签选最大值，仅是不可部署上界，不能作为模型效果。

这是**离线local-only候选池**上的单次干预效果，不是反复干预的实际闭环成功率。即使结果正，也不能据此宣称视觉机器人提升、安全保证或论文已完成。若仅能在一个声明的廉价来源之外有收益，但加入更强廉价来源后消失，应按来源归因失败报告。

本阶段不声称取代RTC/DEFLECT/FutureRTC/AQC；是否形成论文创新，仍需严格全文差异审计、独立建议源与非reach任务。不能靠加网络/校准的名称让创新自动成立。

事后术语澄清：原先称“必要机会条件”过强；有符号oracle平均差的正性不是上下文选择有效的普遍必要条件。原协议/阈值/原始输出保持不变，portfolio正部增量另作描述性分析，不据此重新挑选模型。
