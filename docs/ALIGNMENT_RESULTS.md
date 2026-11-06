# 检查点、时钟与五种子模型效果

完成记录：2026-11-02T17:47:53.176806+08:00。这是状态控制的受控评测，不是视觉VLA实验。

## 本轮完成范围

两个后端都完成2种采集协议×2种时钟设置×5个训练seed，共40个arrival模型。每后端3180条双窗口标签、1800条闭环评测记录；合计6360条标签、3600条记录。记录共享环境seed，不能当作3600个独立样本。
独立offline test为50个episode；闭环为预先指定的前12个episode/延迟，五训练seed。训练/校准数据在模型seed间共享，CI只覆盖模型seed与测试episode变异。
完整测试48 passed、0 failed、0 skipped，包含真实Fetch；每run的status=complete、checks=passed。无失败run、无覆盖旧结果。

## 运行索引

|后端|目录|开始UTC|结束UTC|秒|模型|标签|每场景闭环行数|
|---|---|---|---|---:|---:|---:|---:|
|toy|`outputs/alignment/toy_20261102T091908Z`|2026-11-02T09:19:08.877104+00:00|2026-11-02T09:19:47.977591+00:00|39.10|20|3180|900|
|fetch|`outputs/alignment/fetch_20261102T091908Z`|2026-11-02T09:19:48.099233+00:00|2026-11-02T09:44:45.008767+00:00|1496.91|20|3180|900|

基础提交712d7e3fdb28d36aec795440c28dd405695834af，main，有本地未提交研究改动；源码冻结snapshot、fingerprint、配置/分组checksum、依赖和全部阶段命令/退出码在各run。未推送GitHub。环境沿用本机Ubuntu22.04.5/Python3.12.14/RTX3090，未改CUDA或系统环境。

```bash
.venv/bin/python scripts/alignment_study.py --backend toy --run-dir outputs/alignment/toy_20261102T091908Z
.venv/bin/python scripts/alignment_study.py --backend fetch --run-dir outputs/alignment/fetch_20261102T091908Z
```

以上是已执行命令；再运行必须换新目录。OMP/OPENBLAS/MKL线程均为1，使用tee时启用pipefail。

## 模型效果：固定测试池上的回归与校准

MAE是原episode截止收益的误差，越小越好；coverage是每episode内全部预定请求×合法延迟的联合覆盖，不能当准确率或安全概率。均值跨5模型seed与50环境episode。

|后端|场景|采集/时钟|MAE|coverage|点估计接受率|校准接受率|
|---|---|---|---:|---:|---:|---:|
|toy|shifted|early_no_clock|0.0172331|0.732|3.455%|0.000%|
|toy|shifted|early_with_clock|0.0175923|0.720|5.600%|0.000%|
|toy|shifted|aligned_no_clock|0.0292178|0.844|18.873%|0.327%|
|toy|shifted|aligned_with_clock|0.0299259|0.852|22.691%|0.473%|
|toy|clean|early_no_clock|0.0071757|0.820|2.000%|0.000%|
|toy|clean|early_with_clock|0.0075463|0.820|3.673%|0.000%|
|toy|clean|aligned_no_clock|0.0157104|0.952|16.982%|0.218%|
|toy|clean|aligned_with_clock|0.0163467|0.956|19.636%|0.364%|
|fetch|shifted|early_no_clock|0.0057880|0.828|1.127%|0.000%|
|fetch|shifted|early_with_clock|0.0061494|0.872|1.382%|0.000%|
|fetch|shifted|aligned_no_clock|0.0052606|0.748|2.800%|0.036%|
|fetch|shifted|aligned_with_clock|0.0051904|0.772|5.709%|0.036%|
|fetch|clean|early_no_clock|0.0034596|1.000|0.618%|0.000%|
|fetch|clean|early_with_clock|0.0039996|1.000|0.764%|0.000%|
|fetch|clean|aligned_no_clock|0.0026496|1.000|1.055%|0.000%|
|fetch|clean|aligned_with_clock|0.0027407|1.000|3.745%|0.000%|

## 事前指定对照与区间

MAE差负值较好；coverage、候选收益差正值较好。下列是名义95%交叉bootstrap区间，多指标/多场景属探索性诊断，不挑选显著结果作普遍结论。

|后端/场景|对照|指标|平均差|95% CI|
|---|---|---|---:|---|
|toy/shifted|aligned_with_clock minus aligned_no_clock|mae|0.0007081|[-0.0003164987209376692, 0.0019235182034755829]|
|toy/shifted|aligned_with_clock minus aligned_no_clock|coverage|0.0080000|[0.0, 0.032]|
|toy/shifted|aligned_with_clock minus aligned_no_clock|point_value|-0.0015943|[-0.0045327683842034654, 2.4216126938098647e-05]|
|toy/shifted|aligned_with_clock minus aligned_no_clock|calibrated_value|-0.0000539|[-0.00026370275038760106, 4.969690649508574e-06]|
|toy/shifted|aligned_with_clock minus early_with_clock|mae|0.0123336|[0.007514474729902572, 0.017715208188657287]|
|toy/shifted|aligned_with_clock minus early_with_clock|coverage|0.1320000|[0.012, 0.256]|
|toy/shifted|aligned_with_clock minus early_with_clock|point_value|-0.0018079|[-0.006685564601042576, 0.0024841643643983306]|
|toy/shifted|aligned_with_clock minus early_with_clock|calibrated_value|-0.0001093|[-0.00045265420708104527, 0.0]|
|toy/clean|aligned_with_clock minus aligned_no_clock|mae|0.0006363|[-0.0003280771529374185, 0.0016823472887486613]|
|toy/clean|aligned_with_clock minus aligned_no_clock|coverage|0.0040000|[0.0, 0.024]|
|toy/clean|aligned_with_clock minus aligned_no_clock|point_value|0.0000188|[-0.0002145207585361736, 0.0003174418201475567]|
|toy/clean|aligned_with_clock minus aligned_no_clock|calibrated_value|-0.0000578|[-0.00026865544173324755, 1.2901227160211963e-06]|
|toy/clean|aligned_with_clock minus early_with_clock|mae|0.0088004|[0.0054234023099969594, 0.012517269582099554]|
|toy/clean|aligned_with_clock minus early_with_clock|coverage|0.1360000|[0.032, 0.244]|
|toy/clean|aligned_with_clock minus early_with_clock|point_value|-0.0004417|[-0.0011691129162375055, -1.1508919893687095e-05]|
|toy/clean|aligned_with_clock minus early_with_clock|calibrated_value|-0.0001119|[-0.00047138023082932595, 0.0]|
|fetch/shifted|aligned_with_clock minus aligned_no_clock|mae|-0.0000701|[-0.0002165927739974653, 7.3573930819942e-05]|
|fetch/shifted|aligned_with_clock minus aligned_no_clock|coverage|0.0240000|[-0.04, 0.092]|
|fetch/shifted|aligned_with_clock minus aligned_no_clock|point_value|-0.0000114|[-6.773585253887843e-05, 3.623624480054556e-05]|
|fetch/shifted|aligned_with_clock minus aligned_no_clock|calibrated_value|0.0000002|[-7.519100836361678e-06, 8.894202624850007e-06]|
|fetch/shifted|aligned_with_clock minus early_with_clock|mae|-0.0009589|[-0.0011786624475215856, -0.0007275981757501615]|
|fetch/shifted|aligned_with_clock minus early_with_clock|coverage|-0.1000000|[-0.212, -0.008]|
|fetch/shifted|aligned_with_clock minus early_with_clock|point_value|-0.0000288|[-7.685154186405658e-05, 9.6817444778938e-06]|
|fetch/shifted|aligned_with_clock minus early_with_clock|calibrated_value|-0.0000015|[-9.258652738803442e-06, 0.0]|
|fetch/clean|aligned_with_clock minus aligned_no_clock|mae|0.0000910|[-5.260757027303229e-06, 0.00018794378757801145]|
|fetch/clean|aligned_with_clock minus aligned_no_clock|coverage|0.0000000|[0.0, 0.0]|
|fetch/clean|aligned_with_clock minus aligned_no_clock|point_value|0.0000055|[-1.883955855204993e-05, 3.564709892343307e-05]|
|fetch/clean|aligned_with_clock minus aligned_no_clock|calibrated_value|0.0000000|[0.0, 0.0]|
|fetch/clean|aligned_with_clock minus early_with_clock|mae|-0.0012589|[-0.001466299760795974, -0.00106452586239869]|
|fetch/clean|aligned_with_clock minus early_with_clock|coverage|0.0000000|[0.0, 0.0]|
|fetch/clean|aligned_with_clock minus early_with_clock|point_value|-0.0000000|[-2.1552948625275893e-05, 1.8796058396262158e-05]|
|fetch/clean|aligned_with_clock minus early_with_clock|calibrated_value|0.0000000|[0.0, 0.0]|

## 闭环收益：匹配请求的全拒绝对照

这里两侧都按相同固定时刻查询，查询成本抵消，所以效果不足不能仅归咎于0.01的成本选择。另有local零查询系统对照，完整表在各run analysis.md。

|后端/场景|时钟|方法/延迟|成功率|相对reject效用|95% CI|接受|有害|漏失比例|
|---|---|---|---:|---:|---|---:|---:|---:|
|toy/shifted|no_clock|veto_only,delay=0|1.000|-0.0012919|[-0.0037359878739300573, 0.0002938081290361193]|68|35|0.7027027027027027|
|toy/shifted|no_clock|veto_only,delay=4|1.000|-0.0093374|[-0.01910027443272254, -0.00259046112973864]|38|32|0.8|
|toy/shifted|no_clock|veto_only,delay=12|1.000|-0.0041893|[-0.011947564052283628, -0.00011951335356465605]|13|13|None|
|toy/shifted|no_clock|veto_calibrated,delay=0|1.000|0.0000615|[0.0, 0.00027020769663441135]|2|0|0.9733333333333334|
|toy/shifted|no_clock|veto_calibrated,delay=4|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|toy/shifted|no_clock|veto_calibrated,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|toy/shifted|with_clock|veto_only,delay=0|1.000|-0.0009614|[-0.0034416575530487373, 0.0005345321176088672]|84|38|0.6575342465753424|
|toy/shifted|with_clock|veto_only,delay=4|1.000|-0.0069480|[-0.01498615396224976, -0.0019454305095208302]|43|33|0.6363636363636364|
|toy/shifted|with_clock|veto_only,delay=12|0.983|-0.0220686|[-0.10652582842413423, -0.00013994057030143918]|14|14|None|
|toy/shifted|with_clock|veto_calibrated,delay=0|1.000|0.0000615|[0.0, 0.00027020769663441135]|2|0|0.9733333333333334|
|toy/shifted|with_clock|veto_calibrated,delay=4|1.000|-0.0005888|[-0.0035325752663547872, 0.0]|1|1|None|
|toy/shifted|with_clock|veto_calibrated,delay=12|1.000|-0.0013813|[-0.008287788248205941, 0.0]|1|1|None|
|toy/clean|no_clock|veto_only,delay=0|1.000|0.0005151|[0.00019994720264293744, 0.0008558640825557197]|73|17|0.6875|
|toy/clean|no_clock|veto_only,delay=4|1.000|-0.0055532|[-0.013092320185866037, -0.0005779650072765337]|37|22|0.0|
|toy/clean|no_clock|veto_only,delay=12|1.000|-0.0031537|[-0.010505911414200474, 0.0]|4|4|None|
|toy/clean|no_clock|veto_calibrated,delay=0|1.000|0.0000616|[0.0, 0.0002702684457469684]|2|0|0.9692307692307692|
|toy/clean|no_clock|veto_calibrated,delay=4|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|toy/clean|no_clock|veto_calibrated,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|toy/clean|with_clock|veto_only,delay=0|1.000|0.0005733|[0.0002517505134577827, 0.0009330469084832572]|88|10|0.6615384615384615|
|toy/clean|with_clock|veto_only,delay=4|1.000|-0.0031806|[-0.008637913462442304, -3.150969624360633e-05]|36|18|None|
|toy/clean|with_clock|veto_only,delay=12|1.000|-0.0034522|[-0.010188368572242144, 0.0]|7|7|None|
|toy/clean|with_clock|veto_calibrated,delay=0|1.000|0.0000616|[0.0, 0.0002702684457469684]|2|0|0.9692307692307692|
|toy/clean|with_clock|veto_calibrated,delay=4|1.000|-0.0006170|[-0.0037018974655435135, 0.0]|1|1|None|
|toy/clean|with_clock|veto_calibrated,delay=12|1.000|-0.0014352|[-0.008611476947230334, 0.0]|1|1|None|
|fetch/shifted|no_clock|veto_only,delay=0|1.000|-0.0003457|[-0.000994432322473277, -6.821692901818702e-07]|11|11|None|
|fetch/shifted|no_clock|veto_only,delay=4|1.000|-0.0000151|[-0.00022261988129079017, 8.157473154824592e-05]|4|1|0.926829268292683|
|fetch/shifted|no_clock|veto_only,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/shifted|no_clock|veto_calibrated,delay=0|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/shifted|no_clock|veto_calibrated,delay=4|1.000|0.0000000|[0.0, 0.0]|0|0|1.0|
|fetch/shifted|no_clock|veto_calibrated,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/shifted|with_clock|veto_only,delay=0|1.000|-0.0003535|[-0.0009217811502448945, -7.463818794652458e-06]|21|21|None|
|fetch/shifted|with_clock|veto_only,delay=4|1.000|-0.0001943|[-0.00136710471937158, 0.00012415160948877396]|15|3|0.8571428571428571|
|fetch/shifted|with_clock|veto_only,delay=12|1.000|-0.0000093|[-5.597804236459059e-05, 0.0]|1|1|None|
|fetch/shifted|with_clock|veto_calibrated,delay=0|1.000|-0.0000707|[-0.0004243549171951577, 0.0]|1|1|None|
|fetch/shifted|with_clock|veto_calibrated,delay=4|1.000|0.0000000|[0.0, 0.0]|0|0|1.0|
|fetch/shifted|with_clock|veto_calibrated,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/clean|no_clock|veto_only,delay=0|1.000|-0.0000974|[-0.00041866117672238463, 0.0]|3|3|None|
|fetch/clean|no_clock|veto_only,delay=4|1.000|-0.0000101|[-0.00023328980503720305, 0.00012588265337398763]|3|1|0.875|
|fetch/clean|no_clock|veto_only,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/clean|no_clock|veto_calibrated,delay=0|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/clean|no_clock|veto_calibrated,delay=4|1.000|0.0000000|[0.0, 0.0]|0|0|1.0|
|fetch/clean|no_clock|veto_calibrated,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/clean|with_clock|veto_only,delay=0|1.000|-0.0000837|[-0.00047223724701733684, -1.2068021411903995e-06]|16|16|None|
|fetch/clean|with_clock|veto_only,delay=4|1.000|0.0000326|[-1.2677614965807735e-07, 0.00012581601016988005]|23|0|0.8666666666666667|
|fetch/clean|with_clock|veto_only,delay=12|1.000|-0.0000120|[-7.185554481110578e-05, 0.0]|1|1|None|
|fetch/clean|with_clock|veto_calibrated,delay=0|1.000|0.0000000|[0.0, 0.0]|0|0|None|
|fetch/clean|with_clock|veto_calibrated,delay=4|1.000|0.0000000|[0.0, 0.0]|0|0|1.0|
|fetch/clean|with_clock|veto_calibrated,delay=12|1.000|0.0000000|[0.0, 0.0]|0|0|None|

## 机制、反例与解释边界

1. 对齐不等于模型更强。Toy在扰动测试中，aligned+clock比early+clock的coverage均值高0.132，但MAE更大。Fetch方向不同，应按上表逐项报告，不能写成对齐必然提高coverage。
2. 时钟不是已验证的算法改进。其预先指定MAE/收益对照没有一致优势，不能只选择某个后端、某个seed宣传提升。
3. 采集协议会漏测风险。下面计数来自原始标签，不含模型预测：

|后端|数据池|标签数|截止正收益|终点hold损失|终点hold提升|
|---|---|---:|---:|---:|---:|
|toy|early|1200|118|0|0|
|toy|aligned|880|103|5|0|
|toy|test_shifted|550|60|5|0|
|toy|test_clean|550|52|0|0|
|fetch|early|1200|53|0|0|
|fetch|aligned|880|43|0|0|
|fetch|test_shifted|550|32|1|0|
|fetch|test_clean|550|16|0|0|

Toy的aligned训练池有5个终点hold损失，early为0；Fetch训练池均为0，shifted独立test池出现1个。该有限样本现象不等于普遍规律，也不能把单次干预标签直接相加来预测反复干预。
4. 可重放案例Toy seed5005：request48、arrival60，goal已在46步变化，在请求后不再变化。接受旧动作后在77步才重入容差，截止80步只hold4步，原标准要求5步。标签-1.0177888561008375被精确重放；接受分支最终距离约0.00462，不是碰撞或现实安全事件。案例按负标签筛选，只作机制说明，不增加独立测试样本。
案例文件：`outputs/alignment/case_20261102T091908Z/case.json`、`trajectories.csv`、`case.md`；重放入口`outputs/server_logs/replay_alignment_case.py`。
5. 12个calibration组只是一份固定校准样本。名义conformal结论对交换性下随机calibration/test的边际分布成立，不保证固定这12组后每次实测coverage必达0.8；五模型seed共享同一校准集，没有重复采集的不确定性。
6. early每episode15标签，aligned每episode11标签；因素同时改变检查点分布与样本数量。仅aligned/shifted的离线local-only协议匹配；clean和自适应闭环不继承离线保证。

## 附加误差参照（事后探索，非主比较）

`outputs/server_logs/alignment_error_reference.py`保存了零预测器、只用train拟合的常数均值、以及各模型的MAE/RMSE。它在看到初步结果后添加，仅检查误差量级，不参与选择模型/参数。Ridge拟合平方损失，不能把MAE排序直接当优化目标胜负。

|后端/场景|预测器|MAE|RMSE|
|---|---|---:|---:|
|toy/shifted|zero_predictor|0.0207687|0.0997812|
|toy/shifted|early_train_mean|0.0235248|0.0981418|
|toy/shifted|early_no_clock|0.0172331|0.0961539|
|toy/shifted|early_with_clock|0.0175923|0.0965316|
|toy/shifted|aligned_train_mean|0.0253759|0.0978037|
|toy/shifted|aligned_no_clock|0.0292178|0.0991351|
|toy/shifted|aligned_with_clock|0.0299259|0.0992480|
|toy/clean|zero_predictor|0.0090215|0.0225646|
|toy/clean|early_train_mean|0.0157792|0.0208915|
|toy/clean|early_no_clock|0.0071757|0.0100823|
|toy/clean|early_with_clock|0.0075463|0.0102693|
|toy/clean|aligned_train_mean|0.0185839|0.0217479|
|toy/clean|aligned_no_clock|0.0157104|0.0279502|
|toy/clean|aligned_with_clock|0.0163467|0.0275428|
|fetch/shifted|zero_predictor|0.0057600|0.0438818|
|fetch/shifted|early_train_mean|0.0063463|0.0435226|
|fetch/shifted|early_no_clock|0.0057880|0.0430663|
|fetch/shifted|early_with_clock|0.0061494|0.0429176|
|fetch/shifted|aligned_train_mean|0.0061003|0.0435435|
|fetch/shifted|aligned_no_clock|0.0052606|0.0430958|
|fetch/shifted|aligned_with_clock|0.0051904|0.0430797|
|fetch/clean|zero_predictor|0.0017531|0.0040771|
|fetch/clean|early_train_mean|0.0042147|0.0046613|
|fetch/clean|early_no_clock|0.0034596|0.0038818|
|fetch/clean|early_with_clock|0.0039996|0.0044765|
|fetch/clean|aligned_train_mean|0.0037509|0.0043117|
|fetch/clean|aligned_no_clock|0.0026496|0.0029950|
|fetch/clean|aligned_with_clock|0.0027407|0.0032483|

## 论文贡献与剩余实验

可以争取的贡献是固定同建议的到达干预测量、时窗/截止/成本的分离审计、以及协议/表示因素的受控边界检验。当前没有已证实的新架构、新安全定理或一致优于强基线的模型。
本轮完成了检查点/时钟对照与五训练seed重复，不能再把它们列为未运行。仍未完成：独立建议源、非reach操纵任务、视觉策略、真实服务延迟、50个环境episode的完整闭环主实验。两个后端都是reach，不能称两种操纵任务家族。
下一步应优先验证独立干预通道和非reach任务，而非继续在本test上调gate求正结果。若要对比动作块执行方式，应加入延迟补偿/动作对齐强对照；不能把简单截掉动作前缀称为完整RTC复现。
状态研究结果可以写入范围明确的测量草稿，但还不支持承诺任何期刊会议录用。
