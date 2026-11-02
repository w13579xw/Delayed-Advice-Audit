# Linux 服务器执行与模型交接指南

更新：2026-11-02。适用设备：Linux + 单 RTX 3090 24GB，无真机，零付费 API。以下命令均从 Git 仓库根目录执行。

## 0. 当前接手目标

先完成“安装 → 测试 → toy诊断 → MuJoCo Fetch诊断 → 产物核验 → 报告”，让服务器具备可复现实验基础。本次默认范围不是自动完成14周研究或立即训练视觉大模型。

当前代码的两个控制器是手写策略，收益预测器是小型随机特征岭回归ensemble。流程主要使用CPU；没有下载预训练模型、没有API调用、没有PyTorch/CUDA训练。此阶段GPU利用率为0不代表失败，也不需要为此重装驱动。

历史验证是在Windows上完成的。以本服务器实际命令、退出码和输出作为Linux是否成功的证据。第一次运行可能暴露依赖或操作系统差异；不要声称已在Linux验证。

## 1. 拉取后预检

仓库地址使用用户提供的GitHub地址；私有仓库需服务器已配置GitHub认证。不要把token拼入命令或写入文档。进入clone目录后：

```bash
pwd
git rev-parse --show-toplevel
git status --short
git rev-parse HEAD
uname -a
free -h
df -h .
nvidia-smi
command -v python3.11 || command -v python3.12
```

确认这是本项目目录，至少有 `pyproject.toml`、`src/daci`、`scripts`、`tests`。记录commit与已有修改，避免在其他项目中安装。

建议首轮至少保留数GB磁盘空间；不要把研究计划未来视觉数据所需100–200GB误认为当前诊断的硬性要求。记录服务器实际CPU/RAM；此前尚未获知这些配置。首轮只开一个Python任务。

## 2. 安装隔离环境

支持路径：Python 3.11或3.12。推荐使用服务器已有版本，不为本项目更新系统默认Python。

```bash
# 自动优先选择Python3.11，其次3.12，再检查python3是否为这两个版本之一
bash scripts/setup_linux.sh
```

脚本在项目下创建 `.venv`，安装 `.[sim,dev,plot]` 并运行测试。若需要指定独立Python解释器，可设置`DACI_PYTHON`；通常不需要手工选择版本。若已存在Linux `.venv`，脚本检查并复用；不会删除不兼容的Windows环境。

随后核对解释器与依赖：

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/python -c "import sys,numpy,gymnasium,gymnasium_robotics,mujoco; print(sys.executable); print('numpy',numpy.__version__); print('gymnasium',gymnasium.__version__); print('robotics',gymnasium_robotics.__version__); print('mujoco',mujoco.__version__)"
.venv/bin/python -m pytest -q
```

当前完整套件为25项，后续仓库可能继续新增测试，不要为了匹配某个计数删除测试。要求完整套件通过；Fetch测试不能因缺包而skip。`requirements-tested-windows.txt` 仅记录历史机器，不是Linux安装入口；Linux以 `pyproject.toml` 的依赖为准。

当前Fetch只读取状态，不渲染图像；不需要显示器、Xvfb、`DISPLAY`或EGL配置。只有未来接入RGB视觉轨道时才配置渲染后端。

## 3. 首轮服务器运行

先读取参数说明：

```bash
.venv/bin/python scripts/server_run.py --help
```

使用唯一目录。建议新建一次Bash会话，执行：

```bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
toy_run="outputs/server/toy_${run_stamp}"
fetch_run="outputs/server/fetch_${run_stamp}"
mkdir -p outputs/server_logs

.venv/bin/python scripts/server_run.py --backend toy --run-dir "$toy_run" \
  2>&1 | tee "outputs/server_logs/toy_${run_stamp}.log"

.venv/bin/python scripts/server_run.py --backend fetch --run-dir "$fetch_run" \
  2>&1 | tee "outputs/server_logs/fetch_${run_stamp}.log"
```

先完成toy再启动Fetch。不要把两个命令放到后台同时运行。限制BLAS线程只是控制本轮CPU资源，之后可根据实测调整。日志目录和run目录分开，runner自行创建run目录，并拒绝没有`--resume`的非空目录。

默认配置为 `--train-episodes 40 --test-episodes 4`，horizon80、采集seed100起、评测seed10000起、延迟{0,4,12}。默认不启用`--audit`，以控制首轮耗时。runner会执行：profile、tests、collect、train、十二方法普通评测、六方法固定请求评测、两份报告、时延曲线、有效性分析、完整性检查。首次接通无需另跑多seed sweep。

`set -o pipefail`不可省略：有了它，Python失败时整条 `python | tee` 管道仍返回失败，后续步骤不会误判成功。

如果预计SSH会断开，可先启动tmux：

```bash
tmux new -s daci-bootstrap
# 在tmux内进入仓库，然后运行上面的命令。
# Ctrl-b 后按 d 脱离；重新连接后使用：
tmux attach -t daci-bootstrap
```

若没有tmux，优先保持当前会话执行小规模诊断；不要自行申请付费云资源。断开后恢复方法见第9节。

## 4. 分阶段检查产物

runner是统一入口，当前目录布局如下；将来代码变更时以其输出和 `scripts/server_run.py` 为准：

| 阶段 | 应出现的核心产物 | 核验点 |
|---|---|---|
| 环境/版本记录 | `run.json`、`profile.json`、`logs/tests_attempt_*.log` | 与当前服务器一致，非复制Windows记录 |
| collect | `pairs.npz` | 标签数、episode数、metadata与配置匹配，数值有限 |
| train | `model/query.npz`、`model/arrival.npz`、`model/training.json` | backend正确，episode切分互斥，校准组数量记录 |
| evaluate | `eval/episodes.csv`、`eval/traces.jsonl`、`eval/manifest.json` | 评测seed与采集seed不重叠，所有计划条件都有记录 |
| report | `eval/summary.json`、`eval/summary.md`、`eval/diagnostic.png` | 图和表对应同一CSV，不是旧目录结果 |
| fixed/fixed_report | `fixed/`下对应CSV、trace、manifest及报告 | 六方法，非local方法请求数在相同seed/delay下相等 |
| 曲线/分析 | `temporal_curves.png`、`validity_analysis.json` | 同一请求、同一候选跨延迟比较，注明有限窗口 |
| verify | `checks.json` | `checks`为`passed`，不要求结果为正 |

用 `find "$toy_run" -maxdepth 3 -type f` 和对应Fetch路径查看实际产物；不要仅看到图片就认定全流程完成。若runner额外生成状态文件或阶段日志，应一并保留。

默认40个采集episode、80步horizon、延迟{0,4,12}对应600条标签；十二方法×三延迟×四评测episode生成144条普通评测记录，六方法固定时刻对照生成72条。若显式更改episode数，按manifest重新计算预期行数，不应强行改结果凑数。全部阶段完成后，`run.json`的`status`应为`complete`；`checks.json`只表示完整性检查通过，不是科学有效性证明。

## 5. 怎样读结果

先检查：任务成功率、实际请求数、接受/拒绝数、相对local的配对效用差、区间估计，再检查保形覆盖。指标口径见 `docs/代码与算法说明.md`。

- 方法无提升或比local差，仍应保留结果。工程流程可以验收通过，研究假设则尚未得到支持。
- gate从不请求/从不接受，报告其请求率与收益，不能写成“高可靠性且性能最优”。
- `conformal_offset`为null时，当前实现把它解释为+∞，通常因为校准组不足。它不是数据损坏。不要把null替换为0；需要有限边界时，用新run增加预先设定的采集episode。
- `--audit`启用额外反事实分叉，运行变慢正常。它用于评测有害采纳，不可参与线上决策，其host runtime不是模型推理时延。
- 当前成功率是本项目包装Fetch任务的终点hold标准，不是官方FetchReach leaderboard分数。
- 离线覆盖不自动迁移到接受建议后改变分布的闭环，也不自动覆盖未见延迟。

## 6. 第二阶段：解读固定请求对照，可选增加审计

runner已自动生成小规模固定时刻对照，先读 `$toy_run/fixed/summary.md` 和 `$fetch_run/fixed/summary.md`，无需重复启动。此时不要立即启动五seed大矩阵。

若下一步需要审计有害采纳，下面是独立的toy审计例子（首轮验收不强制运行）。`$toy_run/model` 是本次模型路径；重连shell后应先将变量重新设为已有run的实际路径。未启用`--audit`的首轮输出不能用来声称实测有害采纳率为0。

```bash
set -euo pipefail
fixed_run="outputs/server/toy_fixed_$(date -u +%Y%m%dT%H%M%SZ)_$$"
test ! -e "$fixed_run"

.venv/bin/python -m daci.cli evaluate \
  --backend toy --model-dir "$toy_run/model" --out "$fixed_run" \
  --episodes 4 --horizon 80 --seed 10000 --delays 0 4 12 \
  --fixed-schedule --period 24 --chunk 12 \
  --methods local always ttl drift veto_only veto_calibrated --audit

.venv/bin/python -m daci.cli report --dir "$fixed_run" --plot
```

参数约束：24 >= max(delay=12)+chunk=12。训练时chunk若不是12，必须匹配；backend也必须匹配。不同后端分别训练gate，不能拿toy模型跑Fetch。

此对照固定请求时刻与次数，以观察到达采纳规则的作用。闭环执行不同会产生不同后续状态和建议，因此不是所有方法后续候选内容完全相同；严格的同候选比较应依赖离线成对数据。

若只是按本次“服务器接通”目标工作，完成本阶段或给出明确下一步后即可交付，不自动开始长期研究。后续用户要求继续时保留已有授权与实验记录，自主完成其指定范围。

## 7. 服务器性能测量与研究扩展

runner已经生成`profile.json`。若后续需要单独重测CPU仿真profiling：

```bash
.venv/bin/python scripts/profile_server.py --backend fetch \
  --out "outputs/server/fetch_profile_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
```

这里记录GPU清单，但测量的是状态仿真与成对分叉，不能写成3090 VLA吞吐。

后续独立阶段的验收顺序：

1. 明确新增研究贡献并复核近邻论文；若简单TTL已解释全部效果，先缩小题目。
2. 接入一个已有可用checkpoint的冻结视觉策略，在干净任务上复现基础能力；明确数据/模型来源和版本。
3. 完整恢复视觉策略历史与动作队列，验证分叉轨迹重演；不让特权状态进入视觉gate。
4. 接入独立建议源、至少两类操作任务，采集真实batch=1时延trace；区分合成延迟、trace重放、同卡并发。
5. 冻结训练/验证/校准/测试切分与协议，再做强基线、跨任务/OOD、五训练seed与统计。

现有仓库没有实现上述视觉主实验。不要凭借文件名或论文计划假设它已实现。需要新依赖或模型权重时先核对许可证、磁盘与显存，选择本地免费路径。大型下载/长跑根据用户当次范围与资源预算执行。

## 8. 常见故障及处理顺序

| 现象 | 先检查/处理 | 不应采用的处理 |
|---|---|---|
| `python3.11: command not found` | 检查已有python3.12，设置`DACI_PYTHON=python3.12`；都没有时记录缺失项，并使用用户认可的环境管理方式 | 改系统默认Python、乱用旧3.8装完整依赖 |
| `ensurepip`/`venv`不可用 | 检查该Python是否配套venv模块；必要时请管理员安装对应版本venv包，或使用现有独立Conda环境的解释器 | `sudo pip install`到系统环境 |
| `ModuleNotFoundError: daci` | 确认仓库根目录、`.venv/bin/python`、`pip install -e '.[sim,dev,plot]'`成功 | 手工复制源码到系统site-packages |
| 网络/PyPI失败 | 保留错误和index信息；重试可靠官方源或服务器已配置的可信镜像，记录实际依赖版本 | 使用来历不明whl或关闭校验 |
| Fetch测试skip | 确认gymnasium-robotics、mujoco在当前venv成功导入 | 把skip当作全部通过 |
| XML/asset文件找不到 | 检查上游包完整性、依赖版本、路径；使用英文目录重新建立venv可避开跨平台Unicode问题 | 屏蔽异常或把Fetch改成toy冒充通过 |
| OpenGL/EGL报错 | 检查是否意外启用了render；当前state模式无需渲染 | 为state任务安装整套桌面环境 |
| 输出一段时间不增长 | 查看进程CPU/内存与日志；Fetch反事实分叉会重建仿真实例，可能较慢 | 看到静默就启动第二份相同任务 |
| 出现NaN/物理错误 | 保存seed与命令；先用最小单episode复现，检查版本和输入 | 删除该episode、只保留成功样本 |
| 学习gate未优于local | 如实报告，检查专家是否提供可用正收益、阈值基线是否已足够 | 故意削弱local、调test seed找正例 |

## 9. 中断、更新与日志保留

不要自动覆盖任何存在的run目录。先确认旧进程是否还在运行；只操作本项目自己启动的PID，不终止服务器其他作业。

runner支持恢复，例如：

```bash
# 此例原始运行使用默认40/4且未启用audit。
# 在同一个shell里，fetch_run已指向原来的唯一run目录。
# 重连后先将fetch_run设为该目录的真实路径。
.venv/bin/python scripts/server_run.py --backend fetch \
  --run-dir "$fetch_run" --train-episodes 40 --test-episodes 4 --resume
```

恢复时backend、train/test episode数、audit配置必须与原始命令完全相同；如原始运行带`--audit`，恢复时也必须带。runner还核验源码指纹。参数或代码改变时，它会拒绝恢复，应创建新run并关联原目录。环境或依赖版本改变时也建议新run，不混合两套环境结果。

恢复跳过已完成阶段，但要求该阶段预期产物仍存在；缺产物会报错，不能删除状态字段强行绕过。失败/中断阶段会重新执行，它可能重写该阶段未完成的输出，但旧attempt日志保留为独立文件；重新执行前可复制失败产物用于排查。不要在原进程仍运行时并发恢复同一run。

底层CLI本身未提供事务式恢复，可能覆盖旧文件；不要把“脚本可重复执行”理解成“允许覆盖已完成结果”。

更新仓库前执行 `git status --short`；有本地修改时先记录并保存，不用 `git reset --hard`清理。干净checkout可使用 `git pull --ff-only`。代码/依赖改变后，新run记录新的commit与版本，旧结果保持原样。

按 `SERVER_TASK.md` 在 `outputs/server/HANDOFF_STATUS.md`追加每次运行记录；首次运行前不要在目标run目录预写进度文件，否则非空检查会拒绝启动。run完成后可另存报告快照。不要把密钥、完整环境变量或SSH配置放进日志。大型CSV、模型和图像保留在输出目录，是否回传GitHub由用户后续任务决定。

## 10. 交付给用户或下一位模型的摘要

```text
运行状态：完成 / 失败 / 仍在运行（注明PID与日志）
代码commit：
服务器与Python版本：
测试：通过__ / 失败__ / 跳过__，Fetch是否真实执行：
toy run目录与退出码：
Fetch run目录与退出码：
固定时刻对照目录（如运行）：
数据与评测行数：
关键结果和负结果：
修复及实际改动文件：
未验证事项：
下一条具体可执行命令：
```

工程跑通不等于算法优胜，更不等于顶会成果。下一位模型应能从该摘要和完整日志继续工作，不必猜测前面是否真的执行过命令。
