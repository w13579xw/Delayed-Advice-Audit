# 外部接触任务能力资格协议

2026-11-02，在正式能力test前冻结。目的先验证独立公开专家来源和具备能力的本地控制，不以弱local制造求助收益。不是新算法成果或Meta-World官方排行榜。

- 独立环境`.venv-metaworld`，Meta-World3.1.1要求MuJoCo3.3.0；原`.venv`的Fetch MuJoCo3.3.7保留。requirements-metaworld.txt固定版本。公开来源Farama-Foundation/Metaworld，原包脚本文件逐run存档。
- 两个预先确定任务push-v3与drawer-open-v3，不根据test挑任务。goal-observable 39D状态，原生500步与原生reward/success，不添加目标扰动。v3任务的reward_function_version实际值在后续环境记录核实，不改第三方源码。
- 官方脚本策略作公开专家；调用前复制观测，避免某些公共策略的视图写入污染状态。专家每步重新计算的能力，只是先决条件，不等于迟到固定动作的能力。
- 本地控制是来自公开专家示范的kNN行为克隆，保持反应式逐步观测，无人工降增益/噪声。特征为hand-object、goal-object、object位置和0.1倍gripper开放度；inverse-distance kNN。
- push示范81000–81199，drawer91000–91199，各200回合全部保留，包括失败；每回合500步。候选训练规模40/100/200递增，k固定候选1/3/5；在validation8回合（push82000–82007/drawer92000–92007）选第一个达到7/8成功的训练规模，再选成功率最高、k最小者。无test调参。
- 模型文件/sha/选择时间保存后，才运行独立20回合test（push83000–83019/drawer93000–93019）。每个任务专家和local均需>=80%成功才可进入下一阶段。validation未通过则不打开test。资格未通过仍完整报告，不以换seed抹除失败。
- 能力通过不证明建议来源有不可替代价值。两策略共享示范来源，不能称独立训练知识来源；只说明脱离原reach手写控制家族，接入外部公开控制与学习式local。
- 后续延迟研究必须另建冻结协议和全新测试池。若使用单次专家动作zero-order hold，应直接写明，不能称VLA动作块预测或完整轨迹规划器。执行期间不能重新调用专家伪装为“修复”。

单回合profiling：push调试seed950000，公开专家500步成功，环境创建+执行0.474秒（只一次粗测）。能力验证约千回合，预计数分钟；先完成回归再正式运行。新快照包含MuJoCo INTEGRATION、Python状态、model数组、原生历史和RNG；接触后回放必须逐步一致。
