# S2：开发集划分与零模型选择敏感性

完成：200个完整episode重划分（每task100），每次四格×5个预定容差，共4000个选模配置。它们来自同40个旧开发episode，不是4000个独立实验。原14/6划分的8格验证MSE、dual及选择精确复现，最大误差0；没有访问500个确认episode。15.70秒完成，exit0/checkspassed。

主诊断容差1e-12：

|task|输入/动作格|选zero次数/100|六个验证episode全拒绝次数/100|
|---|---|---:|---:|
|push-v3|cheap_inputs/cheap_actions|3|15|
|push-v3|cheap_inputs/all_actions|4|14|
|push-v3|with_packet/cheap_actions|2|12|
|push-v3|with_packet/all_actions|2|9|
|pick-place-v3|cheap_inputs/cheap_actions|11|11|
|pick-place-v3|cheap_inputs/all_actions|7|7|
|pick-place-v3|with_packet/cheap_actions|17|17|
|pick-place-v3|with_packet/all_actions|9|9|

PickPlace packet/cheap选zero仅17/100；容差从0、1e-14、1e-12、1e-10到1e-8时依次13、14、17、19、19/100。说明原零模型选择不是对所有开发划分稳定成立。Push可出现非zero模型在六个验证输入上仍全拒绝，因此“选zero”和“行为全拒绝”必须分别报告。

这进一步收窄结论：原冻结策略在250个独立确认episode上全拒绝、表面正特征差不等于运行时信息价值的反例仍成立，但不能泛化为每次训练都会选零模型或一般专家无价值。

选模使用这些验证标签，因此其验证reward存在选模偏倚，只作为开发诊断；不把跨split平均reward当独立测试收益、学习曲线或显著提升。量化分布见analysis.json；原seed与分割表见records.json/protocol.json。
