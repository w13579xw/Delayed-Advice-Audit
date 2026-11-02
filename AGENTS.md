# 服务器模型执行入口

本文件写给在 Linux 服务器接手本仓库的编码/研究模型。用户后续明确指令优先于本文件。请先读完，再执行。

## 任务与已确认条件

- 用户设备：Linux、单张 RTX 3090 24GB；没有真机；默认零付费 API。
- 研究问题：一条专家建议在等待期间可能过时，机器人如何估计其到达后的反事实增益并决定是否采纳。
- **本次默认任务是服务器接通与诊断复现**：安装环境、通过测试、运行 toy 与 Fetch 流程、核对产物、记录结果。完成后交付运行报告。用户没有要求在第一次接手时自动跑完14周研究。
- 当前策略为手写反馈控制器与手写开环建议器，gate 是小型回归模型。没有已接入的视觉 VLA、没有 LLM/API 推理、没有真实 GPU 并发服务。当前流程可在 CPU 上运行，3090 暂时闲置是正常的。
- 项目在 Windows 上做过单元与端到端验证；**Linux 服务器尚需你实际验证**。不要把历史验证记录写成当前服务器成功记录。

## 阅读顺序

1. 本文件。
2. [服务器执行指南](docs/SERVER_AGENT_GUIDE.md)：命令、产物、恢复与验收。
3. [README](README.md)：当前功能与限制。
4. [代码与算法说明](docs/代码与算法说明.md)：标签、时钟、成本、数据划分。
5. [验证记录](docs/验证记录.md)：已有运行及负结果。
6. 做科研扩展前再读 [研究计划](docs/研究计划.md)、[实验协议](docs/实验协议.md)、[新颖性审计](docs/相关工作与新颖性审计.md)。

执行命令前检查本地版本：`git status --short`、`git rev-parse HEAD`、`python3.11 --version` 或 `python3.12 --version`。所有命令从仓库根目录运行。不要照搬文档中的 Windows 绝对路径。

## 第一阶段执行顺序

使用项目 `.venv`，不要向系统 Python 安装依赖，不要升级 CUDA/驱动或其他项目环境。

```bash
# 自动选择已有 Python 3.11/3.12；亦可显式指定 DACI_PYTHON=python3.12
bash scripts/setup_linux.sh
.venv/bin/python scripts/server_run.py --help
.venv/bin/python scripts/server_run.py --backend toy --run-dir outputs/server/toy_initial
.venv/bin/python scripts/server_run.py --backend fetch --run-dir outputs/server/fetch_initial
```

`toy_initial`、`fetch_initial` 仅可在目录尚不存在或为空时使用。已有结果则改用唯一 run ID；恢复步骤见指南。runner默认40个采集episode、4个测试episode，并自动运行固定时刻对照。依次运行，不要同时开多个 Fetch 分叉作业。不要默认直接启动 `scripts/sweep.py` 的全矩阵。

每个步骤都检查退出码及实际产物。使用 `tee` 时必须启用 Bash `set -o pipefail`，否则 Python 失败可能被日志命令掩盖。日志空白不代表进程卡死；先检查进程、CPU 与已有阶段日志。

## 不可破坏的实验语义

- 等待期间环境继续推进；候选在到达前不得被 gate 读取。
- 同一请求的各延迟条件使用同一条建议；成对分支共享起始快照及未来外生扰动。
- 快照和未来分支结果仅用于离线标签/评测审计，不得作为部署特征。
- train/calibration/test 按完整 episode 分组；禁止为了提高分数更换测试 seed、删失败轨迹或读取 test 调参。
- query cost 每次请求只计一次，到达采纳阶段不得重复扣费。
- 固定请求模式要求 `period >= delay + chunk`，避免策略因提前拒绝获得更多查询机会。
- 小校准集导致 `conformal_offset=null` 表示数学上的正无穷，全部拒绝属于设计行为。不要把 null 填0来提高接受率。
- 当前 state 模式、合成延迟及自定义 Fetch 包装任务必须保留标记；不称为视觉模型、真实异步服务或官方 benchmark 成绩。

## 修复与执行边界

- 常规环境问题和可逆代码修复可以直接处理；先保留错误日志，再定位最小原因。
- 不通过删除断言、skip 真正失败测试、降低成功标准、缩短未说明的评测预算来“跑通”。环境缺依赖导致 Fetch 跳过，不算完整验收。
- 不调用收费 API；本次 bootstrap 不需要任何 API key。不要索要、打印或提交 token、SSH key、`.env`。
- 不删除已有输出，不覆盖已完成 run，不提交 `.venv`、缓存、数据和模型大文件。所有新实验使用独立目录。
- 不修改上游第三方源码掩盖版本问题；如确有兼容修复，保留补丁、依赖版本与复现说明。
- 不因一项 smoke test 通过就启动长时间多种子训练。先给出单回合/分叉耗时与容量估算，按用户当前目标开展下一阶段。

## 进度与交接记录

按照 [SERVER_TASK.md](SERVER_TASK.md)，维护 `outputs/server/HANDOFF_STATUS.md`，每次运行追加独立小节，至少写清：

```text
仓库commit、分支、是否有本地修改：
服务器OS、CPU、RAM、GPU、Python与依赖版本：
本次run目录、开始/结束时间：
已执行命令及退出码：
测试通过/失败/跳过数量：
数据行数、训练/测试seed、关键产物位置：
实际结果（包括不提升、全拒绝、覆盖不足等）：
修复内容与验证：
尚未完成/失败原因：
下一条可直接执行的命令：
```

记录文件放在各run目录之外，避免首次运行前写入文件导致runner拒绝非空目录。run完成后可以在该目录内另存报告快照。若进程中断，先检查运行状态和产物完整性；不要把半写入CSV当成功结果。`server_run.py --resume`只允许相同代码指纹和相同配置，保留各阶段attempt日志，跳过已有完整产物的完成阶段。修改代码或参数后必须创建新run。不要给其他不支持恢复的CLI凭空添加参数。

## 本阶段完成标准

完整测试通过且 Fetch 测试实际执行；toy 与 Fetch 的采集、训练、普通/固定时刻评测、报告流程退出码为0；各自`run.json`的`status`为`complete`、`checks.json`的`checks`为`passed`；结果在服务器报告中有明确记录。结果不优于local也可以算工程复现完成，但不能算研究假设已成立。

完成后报告：运行目录、核心指标、负结果、尚未验证事项和建议下一步。后续视觉策略接入、第二任务环境、实测3090推理延迟、完整五种子实验应作为独立研究阶段，按研究计划设立验收条件。
