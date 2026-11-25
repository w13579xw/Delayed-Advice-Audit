<p align="center">
  <strong>简体中文</strong> | <a href="./README_EN.md">English</a>
</p>

<div align="center">

# Delayed Advice Audit（延迟建议审计）

**面向机器人仿真环境的拟合策略性能、运行时消息依赖性与延迟消息通道价值审计研究软件与分析数据集**

![Version](https://img.shields.io/badge/Version-1.0.1-6366f1?style=flat-square)
![Python](https://img.shields.io/badge/Python-%3E%3D3.8-3776AB?style=flat-square&logo=python&logoColor=white)
![Code License](https://img.shields.io/badge/Code-MIT-10b981?style=flat-square)
![Data License](https://img.shields.io/badge/Data-CC_BY_4.0-0ea5e9?style=flat-square)

</div>

---

## 项目概述

**Delayed Advice Audit** 是一套面向机器人仿真（MuJoCo / Gymnasium-Robotics / MetaWorld）的端到端研究软件与数值分析数据集，围绕三大核心问题提供严格的可复现审计框架：

1. **拟合策略性能评估**：在状态空间（State）与视觉空间（Vision / ACT）操控任务上，系统对比本地/低成本策略、消息包增强选择器、全动作执行与保守拒绝（`reject`）基线。
2. **运行时消息依赖性审计**：跨多个独立训练随机种子与独立测试环境，量化运行时门控与动作选择对延迟建议消息包（delayed advice packet）的实际依赖。
3. **预定义消息通道与解码器对照**：在原生仿真环境中对比陈旧通道（Stale, `S`）、刷新通道（Fresh, `F`）、控制通道（Elapsed, `E`）以及时间敏感解码器（`time` vs. `legacy`）的真实回报与成功率差异。

---

## 仓库结构

```text
Delayed-Advice-Audit/
├── src/advice_audit/                # 核心 Python 包（通过 pyproject.toml 安装）
│   ├── envs.py, metaworld_adapter.py # 仿真环境封装与 MetaWorld 适配器
│   ├── policies.py, vision.py        # 策略接口与视觉特征管线
│   ├── act_core.py, act_bridge.py,   # ACT (Action Chunking with Transformers) 桥接与适配
│   │   act_adaptation.py
│   ├── value_kernel.py, model.py,    # 价值核拟合、校准、修复策略与选择器模型
│   │   calibration.py, repair.py,
│   │   repair_selector.py, features.py
│   ├── protocol.py, evaluation.py,   # 审计协议、反事实评估与结果报告生成
│   │   counterfactual.py, report.py
│   └── cli.py                        # 命令行入口（`advice-audit` / `python -m advice_audit.cli`）
│
├── experiments/                     # 专题实验驱动与统计分析脚本
│   ├── state/                        # 状态空间主实验（drawer / door / button-press）
│   ├── visual_fit/                   # 视觉价值曲线拟合与到达确认实验
│   ├── visual_pipeline/              # 多随机种子视觉复现与通道 v2 实验管线
│   ├── visual_channel/               # S/F/E 视觉通道干预与时间动态分析
│   └── decoder_control/              # Legacy 与时间敏感解码器对照及回放审计
│
├── scripts/                         # 独立 Worker、参数扫描、ACT/视觉工具与基线脚本
├── configs/                         # 实验配置 JSON（alignment、repair selector、pilot 等）
├── data/analyzed/                   # 23 份完整分析/审计 JSON 与 NPZ 数据及 manifest.json
├── reproduction/                    # 轻量级 NumPy 离线复现套件（8 单元、500 组冻结决策）
│   ├── cached_distances/             # 归一化核距离缓存（pick-place-v3.npz, push-v3.npz）
│   ├── choices/                      # 500 份冻结运行时决策记录 JSON
│   └── reproduce_offline.py          # 免仿真 NumPy 一键离线核验脚本
│
├── tests/                           # 核心单元与集成测试
├── tests_metaworld/                 # MetaWorld 环境专项测试套件
├── tests_vision/                    # 视觉管线专项测试套件
├── pyproject.toml                   # 包构建配置与可选依赖组（sim / dev / plot）
├── requirements-metaworld.txt       # MetaWorld 实验独立依赖
├── requirements-vision.txt          # 视觉管线独立依赖
├── CITATION.cff                     # 规范引用元数据（v1.0.1）
├── LICENSE & DATA_LICENSE.md        # MIT（代码）与 CC BY 4.0（数值数据）协议
└── THIRD_PARTY_NOTICES.md           # 第三方组件版权声明
```

---

## 核心实验结果与审计校验

以下数据均直接提取自仓库 `data/analyzed/` 中经过校验的真实分析记录。

### 1. 数据与回放完整性审计（零误差校验）

| 审计记录文件 | 覆盖规模 | 核心核验项 | 最大回报重构误差 |
| :--- | :--- | :--- | :---: |
| `state_checks.json` | 360 案例、126 开发组 | 回放 18,000 次决策、2,899 唯一叶节点，模型哈希校验通过 | **0** |
| `visual_replication_audit.json` | 300 逻辑案例（3 种子 × 100 环境） | 1,200 次决策、1,584 原生分支尾轨、300 源轨迹完整校验 | **已通过** |
| `visual_channel_audit.json` | 60 主环境 + 6 恒等 + 2 冒烟环境 | 816 次决策、199 分支、136 源消息包核验 | **0.0** |
| `decoder_audit.json` | 68 案例（60 主 + 6 恒等 + 2 冒烟） | 544 次决策、320 唯一路径、204 索引与 136 哈希校验 | **0.0** |

### 2. 状态空间操控主实验（`data/analyzed/state.json`）

在 MetaWorld 状态任务上对比 `reject` 基线、`fixed_expert32` 及三组配对回报差值（均值与置信区间）：

| 任务 | `reject` 回报（成功率） | `fixed_expert32` 回报（成功率） | `packet_cheap − local_cheap` | `packet_full − packet_cheap` | `packet_full − reject` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drawer-open-v3` | **8.0539** (100.0%) | 3.1336 (30.8%) | `0.0000` `[0.0000, 0.0000]` | `-0.4092` `[-0.7100, -0.1653]` | `-0.4092` `[-0.7022, -0.1620]` |
| `door-open-v3` | **8.5962** (94.2%) | 6.0932 (80.0%) | `+0.0051` `[-0.0125, 0.0541]` | `-0.1915` `[-0.3225, -0.0809]` | `-0.1933` `[-0.3220, -0.0747]` |
| `button-press-v3` | **1.1248** (100.0%) | 0.7551 (75.0%) | `0.0000` `[0.0000, 0.0000]` | `0.0000` `[0.0000, 0.0000]` | `0.0000` `[0.0000, 0.0000]` |

> **核心发现**：在状态任务中，未经保护地执行延迟全动作消息包（`packet_full`）在 `drawer-open-v3` 与 `door-open-v3` 上均会导致回报显著低于 `packet_cheap` 与 `reject`，而 `fixed_expert32` 的性能退化尤为明显。

### 3. 视觉管线多随机种子复现（`data/analyzed/visual_replication_analysis.json`）

3 套独立训练的视觉系统在 100 个独立测试环境上的复现结果（每任务 300 组逻辑到达案例）：

| 任务 | `reject` 回报（成功率） | `with_packet / cheap_actions` | `with_packet / all_actions` | `packet_cheap − local_cheap` | `packet_full − reject` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `push-v3` | 7.4215 (97.3%) | 7.2835 (94.7%) | **7.4893** (96.7%) | `-0.0094` `[-0.4274, 0.2602]` | `+0.0678` `[-0.5931, 0.7221]` |
| `pick-place-v3` | **6.1985** (93.3%) | 6.1774 (**95.3%**) | 5.8161 (92.0%) | `+0.4257` `[-0.3671, 1.5998]` | `-0.3824` `[-1.5000, 0.3193]` |

### 4. 视觉通道干预与解码器对照实验（`n_env = 60` 主队列）

- **视觉通道干预（`visual_channel_analysis.json`）**：
  - 在 `with_packet / cheap_actions` 条件下，陈旧（`S`）、刷新（`F`）与控制（`E`）三组回报均为 **6.5569**（`any_success = 88.3%`，`terminal_success = 58.3%`）。
  - 在 `with_packet / all_actions` 条件下，`S` 组回报为 **6.0389**（`83.3%`），`F` 组为 **5.8962**（`80.0%`），主对比 `refresh_minus_stale` 为 **`-0.1428`**（`CI [-0.4932, 0.2285]`）。
- **解码器控制（`decoder_analysis.json`）**：
  - 基线 `R`：回报 **6.9003**（`any_success = 91.7%`，`terminal_success = 66.7%`）。
  - 固定通道：`fixed_S_legacy` **6.8716**（`93.3%`）、`fixed_F_legacy` **6.4413**（`83.3%`）、**`fixed_F_time` 7.0204**（`88.3%`，`terminal_success = 71.7%`）。
  - 门控通道：`gate_S_legacy` **6.4493**、`gate_F_legacy` **6.2909**、`gate_F_time` **6.2611**（共观测到 `24` 次时间敏感源选择、`24` 次解码器动作变更及 `10` 次陈旧/新鲜门控选择变更）。

---

## 快速开始

### 1. 一键离线复现审计（仅需 NumPy，秒级完成）

无需安装 MuJoCo 或 MetaWorld，直接基于缓存核距离核验 8 组视觉拟合/选择单元、500 组冻结运行时决策与 6 组核心配对结果向量：

```bash
python -m pip install 'numpy>=1.24,<3'
python reproduction/reproduce_offline.py --out outputs/offline_check
```

> 注：指定的输出目录（如 `outputs/offline_check`）需为尚不存在的新目录。详见 [`reproduction/README.md`](reproduction/README.md)。

### 2. 核心包安装与单元测试

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,sim,plot]'
python -m pytest -q tests
python -m advice_audit.cli --help
```

### 3. MetaWorld 与视觉测试环境

为避免依赖冲突，MetaWorld 与视觉管线使用独立依赖文件，建议分别创建虚拟环境：

```bash
# MetaWorld 测试套件
python -m pip install -r requirements-metaworld.txt
python -m pytest -q tests_metaworld

# 视觉管线测试套件
python -m pip install -r requirements-vision.txt
python -m pytest -q tests_vision
```

> 原生仿真驱动需自行准备演示数据、Actor 检查点及仿真器依赖；每次新实验请使用独立的新输出目录。

---

## 发布范围、许可证与引用

- **发布范围**：包含源代码、23 份分析数值记录（`data/analyzed/manifest.json`）、紧凑离线复现数据与许可声明；不包含第三方演示数据、预训练网络权重、完整原生轨迹归档及论文投稿文件。
- **许可证**：原创代码采用 **MIT** 协议（[`LICENSE`](LICENSE)），数值分析数据采用 **CC BY 4.0** 协议（[`DATA_LICENSE.md`](DATA_LICENSE.md)），第三方声明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
- **引用格式**（详见 [`CITATION.cff`](CITATION.cff)）：

```bibtex
@software{wang_su_2026_delayed_advice_audit,
  author  = {Wang, Junxiang and Su, Jiaoyang},
  title   = {Delayed Advice Audit: Code and Analyzed Data},
  version = {1.0.1},
  year    = {2026},
  license = {MIT},
  url     = {https://github.com/w13579xw/Delayed-Advice-Audit}
}
```
