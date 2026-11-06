# 补充实验已完成：证据增量与结论边界

所有代码、模型、数据、日志、论文与Git仓库已迁至/home/ez/DelayedAdviceAudit/code。原路径为兼容symlink，历史绝对路径/venv shebang/冻结manifest不改。迁移为同一文件系统rename，目录inode保留。投稿期刊待定，未采用MDPI模板或进行投稿。

## 本轮完成

1. S1/V8a：新增8个训练任务（8个action头、4个3k ACT），固定原数据与选模协议；3个trainingseed×两task×两角色×50全新环境，600完整轨迹、300000原生步，100独立环境，57.6分钟。独立trace/model/split/time审计全通过，reward重算误差0。
2. S2/V9：同40开发episode每task100次完整episode14/6重划分×四格×5容差，共4000选模配置。原8格精确复现；没有读取500确认数据调参。
3. S3/V10：汇总600轨迹全部151500次预测的worker/RPC耗时，无额外native轨迹。

## 主要发现

- 训练随机性影响能力排序：ACT3k−local reward差Push按seed为−1.325、+0.437、−1.542；Pick为−2.269、−1.743、+1.277。两个任务均出现符号反转。跨seed平均分别−0.810/−0.911，但97.5%近似crossedbootstrap区间均跨0。因此不支持一般的架构优劣或稳定reward差。
- 原零模型结论有划分边界：Pick packet/cheap在100重划分中仅17次选zero（tol1e-12），不是每次都选零。原冻结策略在250新确认episode全拒绝的反例仍成立，不能推广为专家没有价值。
- 计算口径有调用频率边界：本地单次推理更快，但每回合500次；ACT每回合5次，记录到的平均worker累计为1.863秒与0.070秒。没有等compute或真实async结论。

本轮增加了可审计工作量和反例的适用边界，没有验证新的到达策略提升，也没有保证可发表。原15k源和完整到达选择器仍只有trainingseed0；critic训练量学习曲线、真实async和外部cleanmachine复现仍未完成。

## 当前路线

将本轮完整结果纳入通用Markdown研究记录与论文补充材料，保留训练反转/划分敏感性。投稿期刊待定。若后续扩大到可泛化方法贡献，优先另立“更多独立critic训练数据+新未触碰测试池”的协议；不可在已完成500确认上反复调参。

证据：S1_RUN.md/S1_RESULT.json、S2_RUN.md/S2_RESULT.json、S3_RUN.md/S3_RESULT.json；原始轨迹与统计在outputs/vision对应run。迁移后从/tmp运行paper/visual_advice_audit/reproduce_offline.py完成5.14秒复现，全部500决策、8格选模和6向量一致；这不等于重新执行native物理。
