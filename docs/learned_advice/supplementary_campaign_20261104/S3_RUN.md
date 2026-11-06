# S3：实际推理调用成本

正式评价开始前追加冻结的描述性方案；只读取S1已记录latency，不添加模拟轨迹。全部151500次调用保留（150000local/1500ACT），600轨迹中的每个local恰500调用、每个ACT恰5调用，exit0/checkspassed。

|策略|调用/episode|平均worker秒/episode|平均RPC秒/episode|
|---|---:|---:|---:|
|local|500|1.862937|3.383862|
|ACT3k|5|0.070239|0.084519|

两task×三seed共6个run/角色，各50episode等权。每cell单次worker耗时median范围local约3.38–3.52ms，ACT约13.52–13.96ms。虽然单次ACT推理更慢，当前100步chunk使其累计调用耗时更小；不能把cheap角色名称理解为已验证的整回合总计算更低。

Worker计时包含图像预处理、模型和GPU到CPU传输；RPC另包含序列化/通信，渲染和物理步在其外。评价walltime另保留（含进程启动/渲染/输出），不能叫真实机器人延迟。温度、首次调用和硬件噪声不滤除。这不是等计算预算、真实async或净query收益实验。
