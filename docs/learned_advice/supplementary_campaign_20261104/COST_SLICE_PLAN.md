# V10：反馈频率下的实际推理开销描述

追加冻结时间UTC：2026-11-04T07:43:15.848958+00:00
当前phase：train_push_act3k_2，正式评价尚未开始。S1原PLAN/protocol不修改，不改变训练、控制或主要比较。

RQ-S3：在当前同一3090运行环境下，每native step本地反馈与ACT100步chunk各自产生多少次预测、每次和每episode记录到的worker/RPC耗时？这是成本口径的描述性补充，不是等计算预算试验、付费query成本或真实异步延迟测试。

只读S1本来就保存的latency.json和run.json，不增添计时instrument，不加模拟环境。不挑快样本、不过滤失败/暖机调用，保留全部成功正式轨迹上的调用；每个local episode应恰500调用，ACT100应5调用（总150000与1500）。报告每task/训练seed/角色的调用次数、每调用worker与RPC耗时median/p90、每episode耗时sum的均值/median/p10/p90、评价run walltime（含启动、渲染、physics、输出I/O，不能解释为机器人执行延迟）。

Worker计时包含图像预处理、模型、GPU到CPU结果传输；RPC另包含文件序列化/通信，但图像渲染在RPC外。CPU/GPU热状态、首次调用、测量噪声和运行顺序均保留；不作环境推广的显著性检验。架构/观察频率/历史不同，不将单次模型更大等同于整个轨迹更贵。

exp_id=V10；slice_id=observed_inference_cost；section_id=compute-boundary；item_id=S3-observed-cost；claim_links=[C-S3]；paper_role=appendix；tier=appendix。完成条件：全部600轨迹151500个调用核验与原latency精确汇总；缺日志则标未完成，不补估数。
