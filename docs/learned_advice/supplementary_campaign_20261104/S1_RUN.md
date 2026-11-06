# S1：跨训练种子的能力复现

完成时间：2026-11-04 16:29（中国时间）；总3456.93秒，约57.6分钟。新增8个训练run，生成8个action头和4个ACT；仅按原定容量评测。全部26个阶段命令exit0，主run complete/checkspassed。

正式评测：Push65400000..65400049、Pick65500000..65500049，每task50个全新环境。trainingseeds0/1/2、local与ACT3k，共600条native500轨迹/300000steps；独立环境数100，不能把同一环境上的多模型成绩当额外独立样本。

|任务|training seed|本地平均奖励|ACT3k平均奖励|ACT−本地|本地any-success|ACT any-success|
|---|---:|---:|---:|---:|---:|---:|
|push-v3|0|7.005612|5.680520|-1.325092|47/50|39/50|
|push-v3|1|6.458054|6.895477|+0.437423|49/50|46/50|
|push-v3|2|7.281510|5.739795|-1.541715|48/50|40/50|
|pick-place-v3|0|6.582834|4.314005|-2.268829|48/50|41/50|
|pick-place-v3|1|6.102684|4.359789|-1.742895|46/50|40/50|
|pick-place-v3|2|5.064757|6.342085|+1.277328|46/50|46/50|

## 预定统计

每task先对3个训练seed平均、再对50个共享环境平均，20000次crossed训练seed/episode bootstrap，两task主reward差各97.5%近似区间：

- push-v3 ACT3k−local平均奖励差 -0.809795，区间[-2.286401, +0.596499]。
- pick-place-v3 ACT3k−local平均奖励差 -0.911465，区间[-2.923967, +1.575814]。

两任务reward区间均跨0，且各有一个trainingseed的点差反转。3个训练seed不足以可靠估计广泛训练总体不确定性；区间仅是本经验分布的敏感性度量，不能证明等效或普遍强弱。

平均any-success分别为Push本地96.00%、ACT83.33%；Pick本地93.33%、ACT84.67%，属于次要指标。6个task×seed exact McNemar比较只有Pushseed0的p=.0078125通过预定.05/6阈值；其余未通过。完整cutoffsuccess、各seed discordances与区间见analysis.json，不以次要指标替换主reward结论。

## 核验与范围

独立audit检查12个模型哈希、冻结早于正式测试、8个新训练split与原协议一致、600轨迹500步/action范围及reward均值，全部通过；rawmean与聚合主reward差重算最大误差0。训练/推理脚本未修改，旧seed0结果保留。

本轮只补3k ACT/selectedactionhead能力层；不是15k训练预算效应的多seed复现，也不是完整15k到达选择器的多seed复现。后两项仍未运行。未改原500确认的模型、主family或结论。
