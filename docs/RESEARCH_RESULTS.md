# 评测论文推进：本轮实测结果与发表边界

更新：2026-11-02T14:14:33.815483+08:00。用户路线：先完成可信评测/负结果，再选择期刊会议。

本轮完成了实验基础与一轮冻结pilot；**尚不满足投稿所需的跨任务/独立建议源/统计证据**。下面结论不外推到视觉VLA或真实异步服务。

## 实现与验收

- 新增四路episode manifest及checksum，validation/calibration独立；模型训练与归一化不读取validation/test标签，回归测试核验了隔离。
- 同一成对分支同时测24步和原episode截止收益；截止标签采用原80步分母，已在真实toy/Fetch对照实际整回合效用差验证。
- 加入强制请求全拒绝、validation选择TTL/drift、线性ridge和去age/去候选/仅当前状态的离线消融。
- 到达后离线审计接受和拒绝候选，报告有害/漏失分母；另核验固定请求时刻、成本分解和审计不改变动作。
- 最终完整测试 **41 passed / 0 failed / 0 skipped**。实验时冻结源码包含40项测试，均实际通过；额外成本重计测试随后加入。真实Fetch测试执行。
- 新toy/Fetch研究run的run.json均complete、checks.json均passed；每后端1200标签、4路48/12/12/8 episode、两个场景各252闭环记录。旧bootstrap兼容toy流程也complete/passed，仍为600标签、144/72行。

## 运行与可复查材料

基础commit：`712d7e3fdb28d36aec795440c28dd405695834af`，分支main，有本地研究改动，未提交/推送。环境与依赖同本轮服务器记录：Ubuntu22.04.5、Ryzen5900X、31GiB RAM、RTX3090 24GB、独立Python3.12.14；实际包版本和源码快照在各run。

所有命令从仓库根目录执行，设置 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`，tee管道使用pipefail。

```bash
.venv/bin/python scripts/research_pilot.py --backend toy --run-dir outputs/research/toy_20261102T055802Z
.venv/bin/python scripts/research_pilot.py --backend fetch --run-dir outputs/research/fetch_20261102T055802Z
.venv/bin/python scripts/compare_research.py --runs outputs/research/toy_20261102T055802Z outputs/research/fetch_20261102T055802Z --out outputs/research/comparison_support_20261102T055802Z
.venv/bin/python scripts/server_run.py --backend toy --run-dir outputs/server/toy_compatibility_20261102T055802Z
```

上面是已执行命令，旧目录不可重新使用；再次运行须换新run-dir/out。研究run在采集前保存protocol.json、split_manifest.json、源码snapshot/patch/指纹和依赖；完整阶段命令/起止时间/退出码在run.json与logs/。

## 1. 时效性存在，但幅度和价值是不同问题

以下收益比例按epsilon=1e-6，oracle为读真实标签的单次干预上界，只对当前local-only候选池成立；不是自适应多次干预上界。CI按episode聚类。

|后端|标签数|短窗口正收益|截止正收益|截止跨延迟翻转组|平均oracle截止收益|扣一次0.01成本后|耗时秒|
|---|---:|---:|---:|---:|---:|---:|---:|
|toy|1200|122 (10.17%)|122 (10.17%)|122/400|0.00011668|-0.00988332|12.74|
|fetch|1200|57 (4.75%)|58 (4.83%)|55/400|0.00002534|-0.00997466|511.85|

这支持“在这些受控候选上存在延迟相关效应”，不支持“值得付费求助”或“复杂门控必要”。成本0.01是自定义效用单位的设置，不能写为实际服务费用。

## 2. 短窗口失配不是普遍主因

Toy中122个短窗口正收益标签没有任何严格的截止负收益反转。Fetch中57个短窗口正收益标签里4个变成严格截止负收益（4/57≈7.02%），但8个独立offline-test episode内该反转为0。因此不能把“短窗口标签普遍制造假收益”写成主结论。新数据在截止指标下仍只有微弱正收益。
两后端所有1200条单次干预标签的terminal_success_gain均为0；收益主要反映距离proxy。与此同时Toy repeated-always在delay12的12个闭环episode均失败。该差异不能直接归为短视窗口或单独归为序列交互：行为继续规则与检查点范围也不同，必须做匹配对照后才能归因。

## 3. 强简单基线与学习门控

两后端独立闭环validation均选择TTL=-1、drift=-1，即请求后全部拒绝。主比较基线依validation而定，平局取TTL，不按test挑对照。

在固定成本0.01下，校准gate在两场景、三延迟均不接受，效用与强制请求全拒绝相同，比不请求的local低0.04（delay0/4）或0.03（delay12）。这是成本损失，不是门控性能提升。
Fetch点估计gate只在delay4接受极少数建议，相对TTL的平均提升约0.000132，12 episode配对CI下限为0；没有稳健优于强简单基线的证据。Toy扰动场景的点估计gate效用差在三延迟均不优于全拒绝。

离线消融全部预先固定并保留，见各run的analysis.md/json。由于只有一个训练seed、8个offline-test episode，不按test宣称某个特征组或线性/非线性模型胜出。

## 4. 零成本对照避免过度负结论

成本敏感性保存成本{0,1e-5,1e-4,1e-3,.01,.05}，固定动作和原validation阈值。当前固定请求/到达门控不依赖查询成本，故可以准确重计；这不是每个成本重新优化后的最优前沿。

Toy干净场景的零延迟always在零成本下有小幅正收益（约0.001547，95%配对CI约[0.001377,0.001707]）。因此“所有建议都无益”是错误表述。Fetch的always在本轮两个场景、三延迟，即使零成本仍为负。具体表和原始分解见comparison_support目录。

## 5. 覆盖率解读中发现的混杂因素

|后端|offline test联合coverage|扰动闭环经验联合coverage|干净闭环经验联合coverage|校准gate接受数|
|---|---:|---:|---:|---:|
|toy|0.875|0.583|0.917|0|
|fetch|1.000|1.000|1.000|0|

采集请求点为0/10/20/30/40；固定闭环请求包含0/24/48/72。每场景132次校准gate请求里60次（45.45%）超出采集最大请求时刻40，36次到达超出采集最大到达时刻52。校准gate实际接受0次，因此不能把Toy覆盖下降归为“采纳引起的分布偏移”。这里还存在不同检查点分布和较短剩余预算。
当前34维特征也没有显式原截止时刻/剩余预算。deadline target虽正确归一化，模型仍可能缺少预测该目标的关键信息。这是本轮发现的设计限制，不能通过挑选test或增大offset掩盖，也不能据此宣称conformal理论失效。论文主张coverage前必须预先对齐检查点，加入可部署的时钟特征，再用新test做受控对照。

## 下一阶段优先级

1. **先做协议对齐对照**：采集与闭环相同固定请求点，允许截止附近合法到达、明确截短窗口；加入可观测剩余预算，在新manifest上对照有/无clock，分离检查点偏移和模型误差。旧结果不覆盖，新实验在看新test前冻结。
2. **再做外部可迁移性检验**：保持可用快策略，接入独立建议源与至少一个非reach任务，验证当前发现是否超出弱手写专家/成功率天花板。负结果论文也不能只证明一个人为弱通道不工作。
3. 冻结主比较、最小有意义效应和样本量后运行五训练seed与更多独立episode；若涉及真实异步性能，另采本机服务trace。当前pilot不能代替该阶段。
4. 按证据填充paper/measurement_outline.md。现有结果更适合一项有范围限制的测量研究；尚不足以负责任地宣布已达到某期刊/会议录用标准。

## 保留的失败与修复

- 首个toy_smoke_v2在tests阶段退出2：输出源码快照被pytest递归收集，造成模块名冲突。增加pyproject的testpaths=[tests]，不删测试、不屏蔽断言；原run及日志保留。toy_smoke_v2_attempt2完整通过。
- 增加成本重计回归测试时，首次patch把另一测试的尾部误置到新函数，引发NameError（40 passed/1 failed）。已恢复正确函数边界，保留失败日志；最终41 passed/0 failed/0 skipped。不是实验失败，也未更改评测标准。
- 独立状态查看曾误用系统Python导入numpy失败，后改用项目.venv；未影响已运行子进程或实验产物。
- 当前源代码与旧baseline不同，各研究run保存其冻结时快照。成本/支持范围后处理在run完成后加入，不修改这些已完成run。

详细运行与交接记录见outputs/server/HANDOFF_STATUS.md；新完整改动补丁和文件哈希位于outputs/server_logs/。
