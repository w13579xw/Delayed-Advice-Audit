# 当前可信结果与已完成确认

更新于2026-11-04。当前是受限评测研究。500个episode独立确认已完成，完整学习选择器未获提升；新发现是零模型能够解释表面packet特征优势。未宣称新算法有效或论文已达到发表标准。完整最新结果见CONFIRMED_RESULTS_20261104.md。用户目标为先完成可信评测或负结果论文，投稿场所未定。

## 已完成的模型对照

同100个新成对episode/task，native500，同MSAA0观测和source执行slice100，只有训练seed0。cheap和ACT共享冻结视觉编码器与task示范，但架构、优化目标及历史不同。两个pose廉价演员另使用同一示范中的object/goal6D监督，再把预测pose交给公开task脚本控制器；ACT不使用该辅助pose目标。因此相同观测和示范episode并不代表监督/任务先验完全等同。2×2各格对这些廉价演员和预测保持一致，不把差异解释为普遍架构胜负。

|冻结模型|Push any-success|PickPlace any-success|Push mean reward|Pick mean reward|
|---|---:|---:|---:|---:|
|视觉小模型|91/100|90/100|6.97245|6.50994|
|未适配ACT|73/100|52/100|3.27280|1.72242|
|ACT3k预算|69/100|80/100|4.83681|4.79311|
|ACT15k预算|79/100|83/100|5.71400|5.56340|

15k相对3k的Push成功率提高10个百分点，98.75%pairedbootstrap区间[3,18]pp，双侧exactMcNemarp=.001953125，通过预设family4阈值.0125；Pick提高3pp未通过。15k与cheap的主要比较均未通过该exact阈值，不写成等效或普遍架构胜负。训练和离线验证选点全过程保留，前3k训练记录一致。此结果是给定训练与选模协议的条件预算效应，不是收敛证明。

## 价值学习与复现审计

- Snapshot-v3保存完整MjData及native数组别名；31项可选接触测试通过。新视觉回放12个完整剩余轨迹案例中图像、预测、动作、状态、奖励逐元素一致。
- 原生MSAA4EGL存在静止画面1灰度值变化，新视觉反事实明确改用MSAA0。原MSAA4训练数据和旧成绩不改。该观测分布变化作为限制保留。
- 40个新单请求开发episode，20/task，每task14train/6validation，12个相关候选结果。相对事后廉价union的昂贵动作排他any-success机会未观察到；奖励正部机会Push.19545、Pick.16454。不是部署收益或一般无价值证明。
- 标准RBFkernel2×2学习pilot未显示真实专家特征稳定优于reject。小验证集已在两task达到reject6/6成功，不能用它证明一般不可学习。
- 未适配和随机计算特征控制保留。40个15k源packet从归档requestfeatures重算与实际livepacket完全一致。专家输出是同一request视觉/robot4的确定性计算，不是新增Shannon信息。
- 数值规则从v1严格正值，到v2零模型/容差，再到v3明确全局最小MSE合格池，旧输出均保留。v3两个数值回归测试和48项运行时核验通过，预测最大误差2.22e-16，选择完全一致。规则在新确认前冻结。

## 已完成的独立确认

arrival_confirmation_20261104_v1、analysis和逐轨迹audit均complete/checkspassed。500个新成对episode，两task各250；同一冻结协议、模型及完整事前调度，7136.54秒，1667唯一尾轨迹、771308native steps。500条运行时选择复算误差0，奖励均值从原始500步轨迹精确核对，样本与模型未按结果更换。

完整packet-input/full-action选择器相对reject，Push平均奖励差−.364803，99.5833%近似配对区间[−.718754,−.049506]；Pick−.635727，[−1.080270,−.229873]。Pick packet-input/cheap-action对照+ .735043，[.293560,1.216208]，但它是zero model、250/250拒绝，运行时不需要packet，不算专家信息价值。Push动作扩展效应+.139147的区间仍包含δ*.25。独立精确eventbounds都过宽，不宣称有限样本排除。

简单aligned ACTsuffix48相对reject只有次要描述性点估计+.032167/+.048462；不据此声称显著提升，也不能宣称所有源建议无效。事后算术分解表明主要负贡献来自学习策略选中的本地替代动作，未验证新的删臂策略。

这是强制query已到达后的单次干预条件价值，early q16/32/64、d0/16/32；不是净查询价值/真实async/多查询/全部可能调度。只有training seed0。新CONFIRMED_RESULTS_20261104.md包含六个主对比、8个参照与贡献边界。

## 写作路线

这轮协议要求的独立确认现已完成；这不自动意味着整体可发表。模型身份、选模、阈值和稀疏对比的报告修订已落实。广义门控、专家作为特征、动作块对齐均有先行工作。贡献候选是明确条件下的零模型解释、受控负结果与复现审计。新稿使用已完成确认结果，限制单训练seed与两个任务，不主张新算法。

完整历史、失败日志、环境、命令和下一步见`outputs/server/HANDOFF_STATUS.md`。旧内部稿`paper/manuscript.md`不代表本轮新证据已经成稿或验证完成。新的独立目录`paper/visual_advice_audit/`现有约4334词英文完整内部稿、术语表与18份精确源数据副本，全部确认效应、标题和结论按完整结果撰写。它不是已批准或已完成的投稿稿。

全文近邻复核补充：PACT-VLA附录D.2已有固定候选池、选择先于分支和共同continuation，因此这些不能作为本研究的首创贡献。详见`PACT_APPENDIX_DELTA_20261104.md`。本轮贡献仍须以强本地参照、零模型解释和独立precision结果具体支撑。

## 表征、反馈频率和成本范围

本地策略复用了开放ACT权重中的冻结视觉编码器，已获得这部分离线预训练知识。实验不是私有专家新增信息的测试。source全策略资格使用100步chunk，本地反馈每步读图；相同观测变量不等于相同观测频率。单次source预测更复杂也不意味着整回合摊销计算更多，100步chunk减少了推理次数。本文没有等总计算预算实验，不把cheap角色名解释为已测定的总成本优势，也不声称共享encoder后的低查询增益代表专家知识无用。结构参数和这些边界保存于outputs/vision/actor_metadata_20261104/metadata.json。


## 完整稿复现补充

八格验证网格、选中dual、500决策与六主向量从不同工作目录离线重算，4.99秒，全部误差0。脚本为paper/visual_advice_audit/reproduce_offline.py，本地资产1151文件/152322616bytes，已排除Git；这不是native重新执行或外部干净机器复现。PickPlace zero靠事先1e-12容差平局胜出，与最佳非zero验证MSE差1.42955e-13，正文已明确，不能描述为验证误差明确更优。δ*.25只是研究定义的2.5%奖励全幅界限，没有已验证操作意义。


## 补充实验完成与迁移（2026-11-04）

完整工作目录为`/home/ez/DelayedAdviceAudit/code`；原项目内仓库路径是兼容symlink，历史配置、模型路径和venv shebang保留。投稿期刊待定，论文继续通用Markdown，未采用MDPI模板或提交。

- **V8a/训练随机性：完成。** 新增8个训练任务（8个action头、4个ACT3k），固定原180/20示范划分、优化器与预定容量。3个trainingseed×两角色×两task×50全新环境，共600条native500轨迹、300000steps；只有100独立环境，不能按600独立样本计算。12模型测试前冻结，26阶段命令全部exit0，独立audit通过、rawreward及主平均差重算误差0。
- **结果：能力reward排序跨seed反转。** Push ACT3k−local为−1.325、+0.437、−1.542；Pick为−2.269、−1.743、+1.277。跨seed平均分别−.810/−.911，97.5%近似crossedbootstrap区间分别[−2.286,.596]/[−2.924,1.576]，均跨0。仅3trainingseed不能提供可靠训练总体覆盖；不能写成本地普遍更强、等效或source无用。原15k和完整arrival策略多seed仍未运行。
- **V9/开发集敏感性：完成。** 同40开发episode每task100次14/6完整episode重划分×四格×5容差，4000选模配置；原8格精确重现error0，未读500确认结果调参。Pick packet/cheap在1e-12下zero17/100，原零模型选择不跨划分普适。验证reward用于选模，不是新测试收益。
- **V10/调用成本：完成。** 全151500次调用保留，local150000/ACT1500，每episode500/5次。平均worker累计1.862937/.070239秒，RPC3.383862/.084519秒；当前频率下local单次更快但累计更慢。不是等compute、净query收益或真实async测试。

完整协议、逐seed结果、分析边界与后续路线：[补充实验报告](supplementary_campaign_20261104/REPORT.md)。通用英文补充材料：`paper/visual_advice_audit/SUPPLEMENTARY_EXPERIMENTS.md`；主稿约4882词，已加入必要的训练/划分/成本限定，保留修改前快照和冻结材料。所有模型、500确认统计与原18源JSON不变。

迁移后从/tmp按新绝对路径执行离线复现，5.136秒complete/checkspassed，500选择、8格选模、6主向量一致；不是重跑native物理。目前没有后台训练或仿真待完成。V8广义15k/arrival训练随机性和critic数据量学习曲线仍是研究局限，不把本轮写成全部补齐。


## 新一轮critic数据量研究（2026-11-04，运行中）

用户授权继续研究；critic_learning_curve_20261104_v1已冻结并启动。新增80开发episode（每task28train/12validation），嵌套n7/14/28，各N四格原RBF/zero/tie规则，随后每task100个全新test、八个事前reward对比。旧500确认不用于新拟合。此study只在同新验证/test下比较数据规模，不能把原14/6与新28/12差异都归因数据量。smoke2episode120.71s、acceptedsource/bitwise reject回放通过，新fitter精确复现原8格/96grid/dual/choiceserror0，相关3项pytest通过。

迁移后修复scripts/visual_bundle.py的模型路径字符串匹配：改为Path.resolve比较，物理演员仍Pushhead0/Pickhead1，source训练或动作语义不变。旧源码备份outputs/vision/migration_manifest_prepatch_20261104.py。新源码fp48ffd6ffff1e3eeb5cc15a475888a844c846ead878292e6137c7f171c5246221，所有新run记录此fp；旧完成run的fp不追改。正式采集/测试预计整体2–3小时，尚未完成，不宣称效果。


## Latest completed evidence and JAAMAS draft (2026-11-05 CST)
V11criticcurveallcomplete,80development/200test,8primarycontrastsnonestrictlypositive. J1–J4newstatesuitecomplete:3tasks/5criticseeds/126development/360test/18000choices; allaudits passed. Fullreport docs/learned_advice/jaamas_campaign_20261105/REPORT.md. Currentpaper /home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS/Delayed_Advice_Audit_manuscript.pdf (23pages), editablemain.tex,2pageinformation sheet,1pageletter,AUTHOR_CHECKLIST. TargetJAAMAS,notMDPI/Robotica. Completevisual15k/fullarrivalactorseedreplicationstillunrun; no realrobot/async/publicrelease/authorapproval/submissionclaim.
