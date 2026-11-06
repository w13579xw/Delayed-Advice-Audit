# 视觉策略与匹配源：已支持的结论及边界

2026-11-03。尚未证明新的到达价值选择算法有效，尚不能判断已达到目标期刊/会议发表要求。已完成的论文初稿仍是旧接触评测研究的内部草稿。

## 已完成的独立视觉资格

所有部署输入相同：翻转后的corner4 RGB512与robot4；wrist为公开源数据的恒黑占位，不是第二个真实相机。专用小模型只训练head、复用冻结ACT ResNet18。180完整task示范训练、20验证，4候选按新20回合开发的any-success/cutoff/reward顺序冻结，再独立50回合/task测试，全部native500。

|指标|Push小模型|原ACT slice100|PickPlace小模型|原ACT slice20|
|---|---:|---:|---:|---:|
|native any-success|50/50|42/50|44/50|35/50|
|截止成功|33/50|12/50|37/50|2/50|
|mean native reward|7.79052|4.29424|7.53480|2.85781|

主要成功差+16/+18百分点；97.5%成对bootstrap区间[6,28]/[2,34]百分点；一侧exact McNemar p=.00390625/.017578125，均通过事前两task Bonferroni阈值.025。所有失败保留，不以截止成功替换主要指标。outputs/vision/qualification_summary_20261103_v1 complete/checkspassed/exit0，测试seed60200000..49/60300000..49禁止再用于新gate训练或调参。

该结果支持所测任务专用视觉head优于这个公开ACT配置，不能解释为新架构优势或普遍“大模型无用”：task数据和完整500步覆盖不同、只有训练seed0、任务特定与多任务预训练不同。CPU ACT/3090 cheap推理时间不能直接解释为固有速度比。

## 同数据ACT适配：当前控制实验

使用同180/20完整task示范与冻结encoder，原ACT transformer/VAE/action head固定3000updates，seed0；未来100action targets包含原生完整500步、正确padding/x符号/原始归一化。模型核心严格来自固定LeRobot0.3.3源码并保留许可；原安装包未修改。GPUraw/CPU输出最大误差1.36e-5，缓存FP16路径近似单独验收。

Push验证maskedL1 .136559 -> .014337；Pick .109717 -> .003233。两run完整退出0。开发20新回合、slice1/20/100全记录；未适配源选择slice1，匹配适配源选择slice100。Push匹配源19/20成功、reward7.61653；Pick18/20、5.86588。开发结果不是独立测试成绩。slice1为同步新鲜反馈参考，不宣称真实异步实时系统。

新的50回合成对资格61000000..49/61100000..49已完整结束，statuscomplete/checkspassed/exit0；三个冻结配置为原cheap、GPU未适配ACT、GPU匹配适配ACT。4个主要双侧McNemar对照Bonferroni阈值.0125，98.75%成对区间，截止/奖励仅次要描述。选择slice不同，因此比较完整源配置，不能称纯适配因果效应。

## 候选贡献与下一证据门槛

1. 可复查的同观测/强廉价控制评测，检验“昂贵建议是否仍有廉价修复无法替代的增益”；需要新的视觉反事实pilot和实际学习选择器测试。
2. 固定建议的到达干预、严格全尾回放及廉价候选归因。旧核心40回合反转已在snapshot-v3复算后保留，但其新颖性是受限实证设计，不是新定理。
3. 数据适配/反馈频率/任务覆盖的对照，为模型比较提供解释。当前匹配源资格已完成，多训练seed未完成，不能据单次结果主张一般规律。

已有工作覆盖异步chunk对齐、预测到达观测和到达时重新选择完整候选（RTC、REMAC、FutureRTC、PACT-VLA）。简单拼接这些技术或给常规回归重新命名不构成新算法。新机制只有在强对照、独立测试和复现实验中产生额外收益后才可进入论文贡献清单。

## 匹配源独立资格结果（完整50回合/task，训练seed0）

|配置|Push any / 截止|Push reward|PickPlace any / 截止|Pick reward|
|---|---:|---:|---:|---:|
|冻结视觉小模型|50/50 / 28/50|7.68227|48/50 / 34/50|6.95942|
|未适配GPU ACT，slice1|44/50 / 20/50|5.73934|25/50 / 2/50|2.05801|
|同数据适配ACT，slice100|47/50 / 10/50|6.56046|41/50 / 17/50|5.35754|

匹配源配置相对未适配配置，Push成功+6个百分点，98.75%CI[-8,22]，双侧McNemarp=.5078125；Pick+32个百分点，CI[12,52]，p=.0008554459，通过4比较阈值.0125。不同执行slice包含频率变化，支持完整适配源配置在PickPlace上更好，不是纯数据适配效应。

小模型相对匹配源成功差Push+6个百分点，CI[0,16]、p=.25；Pick+14个百分点，CI[0,28]、p=.0390625。均未通过4比较阈值，不把次要reward区间或未经校正p<.05替代主要结论，也不把未显著写成等效。完整artifacts：outputs/vision/matched_qualification_20261103_v1。

预算审计：小模型900000视觉anchor曝光，ACT48000；ACT每anchor监督100未来动作，因而该比例不是完整训练量/算力公平的定义。ACT验证误差仍下降，后续最多一轮预设15k更新源预算控制见ACT_EXPOSURE_PLAN.md；尚未启动或取得效果。

视觉审计首次运行visual_replay_20261103_v1在Push60400000/cut16第一个恢复画面即发现RGB hash不一致，已停止全部机会pilot；同一画面下bundle与接受的单head/source输出误差均为0。正在定位渲染/状态复现原因，未删除断言、未降低图像精度、未产出无效的新视觉反事实结论。

## 训练预算对照已完成（100新成对回合/task，MSAA0）

|固定配置|Push any / 截止|Push mean reward|PickPlace any / 截止|Pick mean reward|
|---|---:|---:|---:|---:|
|原冻结视觉小模型|91/100 / 49/100|6.97245|90/100 / 63/100|6.50994|
|未适配ACT100|73/100 / 17/100|3.27280|52/100 / 0/100|1.72242|
|ACT3k预算100|69/100 / 26/100|4.83681|80/100 / 21/100|4.79311|
|ACT15k预算100|79/100 / 29/100|5.71400|83/100 / 48/100|5.56340|

61600000..99/61700000..99在模型与样本量冻结后才打开。主对照15k−3k：Push+10pp，98.75%CI[3,18]pp，10gain/0loss，双侧exactMcNemarp=.001953125，通过family4阈值.0125；Pick+3pp，CI[-7,13]pp，10gain/7loss，p=.6290588，不通过。Pick截止成功+27pp、meanreward+.77029为次要结果，不能替换主要any-success判断。

15k−cheap：Push−12pp，exactp=.0169005；Pick−7pp，p=.229523，均不通过预设family4。Push pairedbootstrap98.75%CI[-24,-1]pp与离散exact检验阈值判断不同，按预先指定exact主检验报告，不以bootstrap区间挑显著。不能把不显著写成等效。

两15k训练分别1683.495/1675.311秒，所有梯度有限，全部15000updates完成；验证选中Pushstep10750（L1.01226523）、Pickstep5500（.00291059），未按native测试挑checkpoint。两task前3000updates保存的loss/gradient/validation记录与原3k逐元素相同，支持这次规定训练预算的干预对照；未声称中间模型权重逐元素核验。全部800资格测试episode完成，summary outputs/vision/act_budget_control_20261103_v1 complete/checkspassed/exit0。只有训练seed0，训练特征MSAA4/新测试MSAA0，不能与旧50episodeMSAA4成绩直接作因果差。

3k源原视觉24row机会pilot也已完成（4独立episode、12options，不是24独立episode）。主要any-success oracle排他机会两task均0；meanreward排他机会Push.00155949、Pick.34708748；均为hindsight上界、不能写成在线效果。两episode/task不足以证明一般无机会。

下一步只用新训练预算源，采集40个新完整episode的单请求决策数据（14train/6validation per task，无test/calibration声称）。保存完整视觉特征、廉价预测与专家packet，分离输入和反事实结果，以检验后续值预测是否有可学信号。创新候选“建议作为计算特征帮助廉价路由”尚在文献审计，不是已经证实的新算法。
