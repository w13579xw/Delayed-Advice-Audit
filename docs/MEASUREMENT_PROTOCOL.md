# 延迟建议评测协议 v2

本协议服务于评测或负结果论文的先行实验。补齐测量缺口，不预设学习门控有效，也不预设短窗口一定误导。

## 冻结与样本

配置：`configs/state_pilot_v2.json`。每次运行在采集前保存协议与checksum、四路分组manifest、源码快照/指纹/patch、依赖版本；目录必须为空。toy与真实MuJoCo Fetch依次运行，不改控制器、任务或成功标准。

- 标签采集：80个episode，seed2000–2079；每episode五个请求检查点、三档延迟，共1200条。同请求的动作块hash在所有延迟一致。
- 四路数据：48 train、12 validation、12 calibration、8 offline test，全部请求/延迟/分支随episode分组。归一化与模型只拟合train；校准只用calibration。独立validation已具备，当前固定模型参数不搜索。
- 闭环阈值选择：另用seed4000–4007的8个validation episode，TTL网格{-1,0,4,12}，drift网格{-1,0,.01,.025,.05,.1,.2}。按三延迟等权平均净效用选择，平局优先少接受再按网格顺序。-1表示全拒绝。
- 闭环test：seed22000–22011，12个episode、delay{0,4,12}、7方法，共252条/场景。包含shift=.15与shift=0干净对照；两个场景共享初始seed，不当作独立任务或独立总体重复。
- 与旧Linux复现、早期smoke种子分开。全部当前实验仍是pilot，不满足五训练seed、跨任务正式论文规模。

## 两种收益对象

固定请求时同一候选，基础控制器在延迟期间继续运动。到达时复制完整快照，Accept执行12步候选后回到local，Continue全程local，共享未来扰动和原任务截止。

旧短窗口 `gain`：未来24步平均负距离 + 该窗口末连续hold成功，两分支之差。窗口末成功不是原episode最终成功。不同延迟的窗口截止时刻不同，因此是同长度窗口，不是同绝对截止。

新 `gain_deadline`：

```text
(sum_future_rewards_accept - sum_future_rewards_continue) / original_episode_horizon
    + terminal_success_accept - terminal_success_continue
```

公共过去抵消，故等于整episode效用 `mean(all_rewards)+terminal_success` 的单次干预差。不同延迟都在原episode第80步截止；不能把分母换成剩余步数后仍称与整回合效用一致。

两者均为一次未来实现、本次候选后local-only继续，不是多次干预自适应策略的Q值。短/截止标签同时存储，审计标签不进入线上特征。报告符号一致与不一致，不为验证预期而调窗口。

## 一次请求的价值可行性

对固定候选分布，单次干预收益G和任意二元接受决策a满足 `a*G <= max(G,0)`。一次强制发生的请求成本c下，平均净收益不超过 `E[max(G,0)]-c`。用episode-cluster bootstrap报告此乐观可行性检查的区间。

这是直接不等式，不宣称新理论。oracle读未来标签、现实不可部署；候选来自local-only检查点，不能跨时间相加，也不能据此否定改变后续分布的自适应多次干预。0.01是诊断效用单位下的假设成本，非实测API费用，需另看零成本表现和敏感性。

## 基线与消融

闭环固定请求：local（不请求、成本0）、reject（相同请求但全拒绝）、always、validation选定TTL/drift、veto_only、veto_calibrated。period24满足 `period >= max_delay + chunk`。非local各方法的请求时刻一致，但历史干预会改变后续候选，不能称后续内容完全一致。

独立offline test用同候选集比较完整随机特征ensemble、线性ridge（无随机特征、无bootstrap）、去显式age、去候选摘要、仅当前状态、TTL/drift与标签oracle。全部预先声明，不据test挑赢家。去age仍保留运动/历史，不能称彻底去除了时间信息。离线消融不冒充各模型的闭环性能。

## 审计与统计

到达时先做接受决策，再离线分叉。短窗口与截止标签分别统计有害接受、正收益候选、漏失正收益；epsilon=1e-6，保留epsilon=0有害计数。没有接受或没有正收益候选时对应比例为null，不填0。

coverage分开报告offline episode联合下界覆盖、闭环点经验覆盖、闭环episode×预设delay经验联合覆盖。闭环分布不继承离线conformal保证，group-max校准也不是选择后有害比例控制。

标签CI按episode聚类，不将1200条相关标签当独立样本。闭环CI在固定scenario/delay内重采样配对环境seed，单训练seed不含训练方差。8个offline-test episode、12个闭环test episode的CI只作探索，不能用于强等价/非劣结论或事后显著性挑选。

## 运行与产物

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q
.venv/bin/python scripts/research_pilot.py --backend toy --run-dir outputs/research/NEW_TOY_ID
.venv/bin/python scripts/research_pilot.py --backend fetch --run-dir outputs/research/NEW_FETCH_ID
```

先toy后Fetch；使用tee时加 `set -o pipefail`。目录必须全新，无自动恢复/覆盖，不下载模型、不调用API、不自动长扫。

产物：`protocol.json`、`split_manifest.json`、`environment.json`、`source.patch`、`source_snapshot/`、`pairs.npz`、`models/`、`validation/`、`selection.json`、`test/`、`test_clean/`、`analysis.json/md`、`checks.json`、`run.json`和`logs/`。研究checks由`analyze_research.py`产生，不与旧bootstrap的三路/144/72验收混用。

可报告明确控制器、包装环境、扰动、延迟、成本与样本量下的发现。不能写视觉VLA效果、官方Fetch提升、GPU并发性能、普遍泛化或现实安全保证。两个后端仍是同类reach控制，不是两种操纵任务家族。
