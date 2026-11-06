# Delayed Advice Audit

**Auditing the Value of Delayed Advice in Autonomous Agents**

当前仓库：[w13579xw/Delayed-Advice-Audit](https://github.com/w13579xw/Delayed-Advice-Audit)。

本项目研究如何区分三个不能互相替代的评估结果：独立训练策略的性能差异、冻结接收器对消息的依赖，以及预先定义的消息通道干预带来的实际轨迹收益。目标期刊为 Autonomous Agents and Multi-Agent Systems；尚未投稿。

[统一命名检查](docs/release_consistency_20261106/REPORT.md) · [分析数据归档](archives/Delayed_Advice_Audit_data_20261106.zip)

## 当前证据与论文

- PickPlace 的 packet-input 策略相对 local-input 策略奖励增加 +0.735，但胜出的零预测器在 250 个测试案例中全部拒绝干预。这证明该策略间差异不能归因于新返回的专家信息。
- 已完成 Push/PickPlace 视觉任务及 Drawer/Door/Button 状态任务研究、冻结策略 S/F/absence 配对仿真、候选动作 headroom、选择规则敏感性与三个完整视觉训练系统复现。没有建立跨种子的完整到达策略正收益。
- 独立视觉解码实验中，预定义固定 48 步新消息按生成时间解码，相比旧解码提高 +0.579061，98.75% CI [+0.172635,+0.997332]；它不证明刷新相对陈旧消息或门控策略相对本地继续执行有正收益。
- [主论文 PDF](manuscript/JAAMAS/Delayed_Advice_Audit_manuscript.pdf)（28 页）、[Online Resource 1](manuscript/JAAMAS/supplement/Advice_Audit_Online_Resource_1.pdf)（19 页）、[LaTeX 源码与构建](manuscript/JAAMAS/README.md)。作者身份、批准与投稿行政信息仍待作者完成。

当前证据支持仿真评估与归因方法；没有真实机器人、人类教师、真实异步服务或独立私有专家观测验证。合成环境步延迟不等于实测服务延迟。

## 使用

项目分发名 `delayed-advice-audit`，Python 包 `advice_audit`，命令 `advice-audit`。从当前仓库目录执行：

```bash
bash scripts/setup_linux.sh
.venv/bin/python -m pytest -q tests
.venv/bin/python -m advice_audit.cli --help
.venv/bin/python scripts/server_run.py --backend toy --run-dir outputs/server/toy_new_run
.venv/bin/python scripts/server_run.py --backend fetch --run-dir outputs/server/fetch_new_run
```

每次使用新的输出目录；Fetch 使用 MuJoCo。toy/Fetch 是早期状态诊断，不是主论文的视觉任务。MetaWorld 与视觉依赖分别见 `requirements-metaworld.txt`、`requirements-vision.txt`。完整训练复现需要未纳入 Git 的数据、权重及冻结运行记录；本文源码包不能代替这些资产。

## 复现与历史

- [服务器执行约束](AGENTS.md)、[原型服务器指南](docs/SERVER_AGENT_GUIDE.md)。这些引导文档的早期阶段描述不代表当前研究状态。
- [命名迁移与兼容性](docs/NAME_MIGRATION.md)、[本地目录](WORKSPACE_LAYOUT.md)。2026-11-06 更名验证：基础/Fetch 65 项、MetaWorld 31 项、视觉 1 项全部通过；固定 toy/Fetch/Push/PickPlace 样本改名前后逐位相同。
- `docs/learned_advice/`、`paper/` 为按日期保留的研究记录与旧稿；当前投稿稿件以 `manuscript/JAAMAS/` 发布副本及本地 Springer 作者目录为准。

当前发布仅提供统一命名的命令与导入接口；原始运行记录另存于私有溯源备份。数值结果、模型和推断结论保持不变。代码、论文和记录均不构成录用保证。
