# JAAMAS supplementary campaign: completed

结论：补充实验没有建立干预控制器性能提升。它加强了评测贡献：策略间收益、运行时消息依赖和指定通道的价值是不同问题。

3个新增状态接触任务；126个开发episode（每任务30train/12validation）；60个拟合变体（每格5initializations）；360个独立测试环境；12种delay/frequency/sensornoise条件；18000次决策重算和2899条独立保存路径均通过，reward reconstruction error0。真实执行1118988native test steps。完整queue819.156秒，6阶段exit0。

|Task|Contrast|Mean|99.4444% approximate interval|
|---|---|---:|---|
|drawer-open-v3|packet_cheap_minus_local_cheap|+0.000000|[+0.000000,+0.000000]|
|drawer-open-v3|packet_full_minus_packet_cheap|-0.409175|[-0.709988,-0.165329]|
|drawer-open-v3|packet_full_minus_reject|-0.409175|[-0.702225,-0.161971]|
|door-open-v3|packet_cheap_minus_local_cheap|+0.005082|[-0.012540,+0.054098]|
|door-open-v3|packet_full_minus_packet_cheap|-0.191465|[-0.322515,-0.080944]|
|door-open-v3|packet_full_minus_reject|-0.193315|[-0.321979,-0.074699]|
|button-press-v3|packet_cheap_minus_local_cheap|+0.000000|[+0.000000,+0.000000]|
|button-press-v3|packet_full_minus_packet_cheap|+0.000000|[+0.000000,+0.000000]|
|button-press-v3|packet_full_minus_reject|+0.000000|[+0.000000,+0.000000]|

Drawer两种local-action输入变体各5seed均zero；Button全部20变体zero；Door全部20nonzero。Door packet/local-action用零替换消息在1500次loggedchoices中改变1063次，但原packet-inputgain区间跨0。该probe未重跑outcome，可能off-support，不是因果信息价值估计。

资格验证：Drawer public/local20/20，Door18/20，Button20/20。均选择40whole demonstrations，k1/5/1；没有淘汰任务。额外完整性检查：7entryhash、60train-onlynormalizations、60globaltieparity、2899noise streams、3600null-policyepisodeidentities均通过。

边界：五seed仅是新状态critic；3kvisualactors有3seed，15k/fullvisualarrival仍actorseed0。无真机、无真实async、无直接语义噪声。Newstate与visual改变source/representation/actionfamily，分组报告不池化原始reward。No positive primary lower bound不是普遍无效/等效证明。

后续：写作已转向Springer/JAAMAS regular研究稿，作者确认与公开release仍需人类完成；不能宣称已投稿或可保证录用。
