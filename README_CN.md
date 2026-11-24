<p align="center">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

# Delayed Advice Audit（延迟建议审计）

用于在机器人仿真环境中审计拟合策略性能、运行时消息依赖性以及预定义消息通道价值的研究软件与分析数据集。

## 目录结构

- `src/advice_audit/`：仿真环境、策略实现、数据采集、拟合与评估核心模块。
- `scripts/`：可执行研究工作脚本与基线工具。
- `experiments/`：状态管线、视觉管线、固定通道及解码器控制实验驱动脚本。
- `data/analyzed/`：20 份数值分析/审计 JSON 记录、3 个 NPZ 数组及文件清单。
- `reproduction/`：轻量级 NumPy 离线复现审计脚本，覆盖 8 个拟合/选择单元、500 组冻结决策与 6 组核心配对结果向量。
- `tests/`、`tests_metaworld/`、`tests_vision/`：分别面向基础、MetaWorld 与视觉环境的自动化测试套件。

## 基础安装与自检

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,sim,plot]'
python -m pytest -q tests
python -m advice_audit.cli --help
```

MetaWorld 与视觉相关依赖列在各自独立的 `requirements-*.txt` 文件中，建议使用独立虚拟环境。原生驱动需要自行准备演示数据、Actor 检查点及相关仿真依赖；记录中的数据/模型路径均相对于解压后的项目根目录，每次新实验请指定新的输出目录。

## 离线复现审计

```sh
python -m pip install 'numpy>=1.24,<3'
python reproduction/reproduce_offline.py --out outputs/offline_check
```

该离线审计从缓存的归一化核距离与开发集标签出发，验证拟合/选择逻辑、冻结决策与配对结果计算（不独立验证视觉特征提取或距离构建，也不执行 ACT 或原生仿真）。详见 `reproduction/README.md`。

## 发布范围与许可证

本发行包包含源代码、数值分析数据、最小使用文档及许可声明；不包含第三方演示数据、预训练神经网络权重、完整原生轨迹归档及论文投稿文件。

原创代码采用 **MIT** 协议（`LICENSE`），数值数据采用 **CC BY 4.0** 协议（`DATA_LICENSE.md`），第三方组件声明见 `THIRD_PARTY_NOTICES.md`，引用元数据见 `CITATION.cff`。当前版本：v1.0.1。
