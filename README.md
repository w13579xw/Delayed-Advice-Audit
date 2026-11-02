# DACI：迟到的专家建议，机器人还应该采纳吗？

**交付定位：可运行的研究原型 + 单 RTX 3090 纯仿真研究方案。没有训练好的视觉机器人模型，没有顶会录用保证。**

拟定论文题目：*Should a Robot Obey Advice That Arrives Late? Counterfactual Acceptance of Delayed Interventions*。

核心假设：建议在请求时有用，不代表它在到达时仍有用。机器人继续运动、目标变化和专家模型误差都可能改变干预收益。研究对象是具体建议在不同延迟下的收益曲线及采纳决策。允许曲线非单调，不能简单把它压成一个固定过期时间。

这不是“首次反事实干预”或“首次自适应调用”。2026年10月的 When to Ask for Help、RESCUE、ForkCast 已经非常接近，见 [新颖性审计](docs/相关工作与新颖性审计.md)。

## 先阅读

- **服务器模型从 [AGENTS.md](AGENTS.md) 开始，再按 [服务器执行指南](docs/SERVER_AGENT_GUIDE.md) 操作。**
- [可直接交给服务器模型的任务书](SERVER_TASK.md)：本次执行范围、交付物、停止条件。
- [研究计划](docs/研究计划.md)：候选贡献、单卡预算、分阶段否证、投稿条件。
- [实验协议](docs/实验协议.md)：数据划分、强基线、成对统计、OOD 与泄漏控制。
- [代码与算法说明](docs/代码与算法说明.md)：准确描述哪些已经实现、哪些仍需开展。
- [本机验证记录](docs/验证记录.md)：实际跑过的命令与结果；不把 smoke test 写成论文实验。

## 环境

建议在服务器上使用 Ubuntu 22.04/24.04 + Python 3.11/3.12。Fetch 的上游主要支持 Linux/macOS；Windows 本地验证情况见验证记录。源代码核心兼容 Python 3.8，但完整依赖请用 3.11/3.12。

```bash
git clone git@github.com:w13579xw/daci-robotic-agent.git
cd daci-robotic-agent
bash scripts/setup_linux.sh
source .venv/bin/activate
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
python -u scripts/server_run.py --backend toy --run-dir "outputs/server/${RUN_TAG}_toy"
python -u scripts/server_run.py --backend fetch --run-dir "outputs/server/${RUN_TAG}_fetch"
```

私有仓库要求服务器已有授权 SSH key；也可以使用已登录 GitHub CLI 的 HTTPS 凭据。不要把 access token 写入命令、URL、日志或发给模型。网络仅用于首次安装开源依赖；运行时不调用收费 API、不下载模型权重。

新入口为每次运行保留日志、命令、源码指纹和验收结果；同参数加 `--resume` 可接着失败阶段执行。默认每个后端采集40个episode、测试4个episode，完整12方法评估144行、固定查询6方法评估72行。新入口默认不做昂贵的逐次采纳反事实审计，要分析 harmful accepts 时从一开始加 `--audit`。第一次复现完成只说明工程链路工作，不要求DACI优于基线。旧 `run_smoke.py`、`sweep.py` 和手动CLI命令可能覆盖同名输出；自动接手优先使用新入口。

源码、研究方案和测试会随Git同步；虚拟环境、实验输出和模型文件留在各自机器。`requirements-tested-windows.txt` 仅为本地验证记录，不是Linux依赖锁文件。

Windows 激活命令：`.venv\Scripts\Activate.ps1`。也可始终用 `.venv\Scripts\python.exe`。
服务器没有显示器也可以运行当前 state 模式，不需要 OpenGL 渲染。若只运行 toy：`pip install -e '.[dev,plot]'`。

## 分步运行

```bash
python -m daci.cli collect --backend toy --episodes 100 --delays 0 2 6 12 --out outputs/toy/pairs.npz
python -m daci.cli train --data outputs/toy/pairs.npz --out outputs/toy/model --alpha 0.2
python -m daci.cli evaluate --backend toy --model-dir outputs/toy/model --out outputs/toy/eval --episodes 30 --delays 0 6 12 24 --audit
python -m daci.cli report --dir outputs/toy/eval --plot
python -m daci.cli curves --data outputs/toy/pairs.npz --out outputs/toy/temporal_curves.png
```

切换真实 MuJoCo FetchReach 后端时，把 `--backend toy` 改为 `--backend fetch`，并重新采集、训练，使用新的输出目录。禁止混用 backend 的 gate。

```bash
python scripts/sweep.py --backend toy --train-episodes 100 --test-episodes 30 --seeds 0 1 2 3 4
python scripts/aggregate_seeds.py --dirs outputs/sweep/toy/seed_0_id outputs/sweep/toy/seed_1_id outputs/sweep/toy/seed_2_id outputs/sweep/toy/seed_3_id outputs/sweep/toy/seed_4_id --method daci --delay 12 --out outputs/sweep/toy/cross_seed.json
```

这个 sweep 是开发期诊断。训练 seed 改变分组划分、随机特征和bootstrap；采集数据共享。正式论文还需要数据种子与任务家族变化、真实视觉策略、独立专家和公平调参。

## 已实现

1. **两种后端**：toy 三维惯性到达；Gymnasium-Robotics 的真实 MuJoCo FetchReach-v4。二者均为状态观测诊断，不是视觉操作 benchmark。
2. **成对标签**：请求时固定建议，环境继续推进 D 步；从同一到达状态分叉“执行建议/继续本地控制”，共享未来扰动及 RNG。
3. **小模型**：bootstrap 随机特征 ridge 回归，预测到达收益；另有辅助 query gate。
4. **到达门控**：均值、ensemble保守项、可选按episode最大残差的一侧 split conformal。
5. **12种诊断配置**：local、always、periodic、random、risk、ttl、drift、query_only、veto_only、veto_calibrated、daci、daci_calibrated。risk 是距离阈值启发式，不是已发表风险检测算法的复现。
6. **记录与分析**：请求/接收/拒绝事件，成对episode bootstrap，跨训练seed统计，延迟收益曲线与PNG图。

## 必须保留的边界

- 当前“专家”是手写名义动力学开环规划器；本地策略是手写反馈servo。**没有调用LLM、没有VLA、没有学习视觉表征。**它们用于检验数据和执行逻辑，不能支持大模型机器人结论。
- 当前延迟是确定性的环境步注入；环境在这段时间内持续step。没有真实并发模型服务，也没有测过3090推理时延。
- 当前 query target 使用正收益部分的期望，是知道如何完美拒绝负建议时的乐观代理；闭环 gate 有误差，论文需做 policy-consistent 更新。到达拒绝主线可用 `veto_only` 在固定查询条件下检验。
- 当前训练分支来自local-only轨迹，部署gate改变状态分布；不能直接继承离线覆盖保证。保形offset样本不足时是无穷大，策略会拒绝全部，属于预期行为。
- 统计脚本不会自动完成公平预算配对或强基线调参。总请求上限相同不等于实际计算成本相同，正式实验需扫描预算并比较相同成本截面。
- 标准 FetchReach 的goal与reward/终止已包装为命令变化诊断任务，结果不能叫官方 FetchReach leaderboard 成绩。

## 论文主实验仍需完成

至少两个操纵任务家族，冻结预训练/BC视觉动作策略 + 独立建议源；同样的观测与动作接口、真实profiling延迟重放、跨延迟/扰动/任务测试；对照 RTC/REMAC、VoI、RESCUE/ForkCast 能公开复现的部分及强的陈旧建议拒绝启发式。详细接入验收见研究计划。

不需要付费API即可执行本项目。项目不会连接或操作任何真机。

## 固定请求时刻，隔离到达采纳

```bash
python -m daci.cli evaluate --backend toy --model-dir outputs/toy/model --out outputs/toy/fixed --episodes 30 --delays 0 6 12 --fixed-schedule --period 36 --methods local always ttl drift veto_only veto_calibrated --audit
python -m daci.cli report --dir outputs/toy/fixed --plot
```

固定模式要求 period >= delay+chunk，保证每条建议及动作段结束后才到下一请求槽。always/ttl/drift/veto_only/veto_calibrated 具有相同请求时刻和数量；多个干预后状态可不同，因此后续建议也可能不同，这是闭环后果。更严格的同候选比较使用离线成对标签。local不调用专家，是系统成本基线。
