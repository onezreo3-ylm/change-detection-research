# Valais B0：完整扰动验证与下一匹配对照

日期：2026-10-02。B0评分于北京时间22:09:52完成。此记录更新早先“仍在运行”
的事实，不改写已经seal的旧续接文档、配置或结果。原全FP32训练外层完成标记
缺失的历史仍保留；该clean基线由独立完成审计与受守卫独立验证恢复确认。

## 已核验结果

同一epoch53 checkpoint、Sierre1715 tile、45条件/77175实例；clean F1精确复现，
status/summary一致，guard完成且summary/state哈希吻合。全部条件有相同源覆盖，
每条件源混淆计数之和与总计一致，mF1从45个condition计数复核；清单、实例、
checkpoint、acceptance哈希与原记录相同。此次分析没有重新推理或打开test。

| 指标 | 结果 |
| --- | ---: |
| clean F1 | 0.4162248161980205 |
| clean Precision / Recall | 0.4261829556 / 0.4067214129 |
| 45条件corruption mF1 | 0.37004352259315076 |
| clean减corruption mF1 | 0.04618129360486972 |
| pooled corruption Recall | 0.3422552867 |
| clean / pooled corruption背景FPR | 0.0014926838 / 0.0013496604 |
| corruption相对clean重复45次的净FP | -721409 |
| 同一比较的净FN | +886338 |

275个有变化clean tile中176个TP=0，不能将其误写成176个tile完全没有正预测；
其中可能仍有FP。1440个空标签tile中39个产生正预测，合计60498个FP。
强度L1..L5的跨九类mF1为0.407053、0.390470、0.373962、0.351909、0.326823。
color_bias family mF1=0.280821、shadow=0.320402，是当前较大退化来源；本批
不据这些分数调整强度、抽掉类型或打开test。

当前clean检出不足并存，扰动造成的整体损失主要表现为新增漏检而非汇总FP增加。
不同family的FP/FN方向不相同；汇总不能证明所有实例的因果损伤。方向/时相/
global-local分层以**同一批源的clean计数**作参考，避免把分组标签比例不同混为
扰动效果。Sierre单城且没有可靠空间源ID，不做tile独立bootstrap。

结构化数据及CSV在 `runs/valais_matched_study_v1_20261002/baseline_analysis`。

## 固定train诊断

按train-only标签分层与独立SHA种子固定8张，实际原尺寸查看了全部8个A/B/label/
TP-FP-FN面板。8002.tif的102个正像素是左边界薄片，模型全漏检；3173.tif仅有
边界x255、y120..121两个正像素，不能认定具有可恢复的语义变化。9371/3270较大
标签上检出部分目标，同时有明显未标屋顶区域预测；外观差异可能是屋顶修缮、
足迹标注定义或漏标，仅凭图像不能确定。4张空标签样例均无正预测，也不能用来
估计总体误报率。未观察到这些样例的整体尺寸错配，但不宣称全数据标签正确。
原数据、标签、采样、损失、阈值和强度均保持。

图像PNG总字节2341834（约2.23MiB），每次工具最多2张；未加载旧大型QA联系表。
视觉记录、图像/来源哈希与守卫完成在该研究目录的train_diagnostic及
diagnostic_review.json中。后续对话优先读文本，不重复加载这8张图。

## 下一项已准备的实验

[匹配协议](VALAIS_MATCHED_BASELINES_PROTOCOL_2026-10-02.md)与独立B1/B0-E YAML
已建立。原模型、训练器、算子和守卫代码保持；单列wrapper增加实际来源顺序与
峰值分配记录。267项unittest通过（67.891秒）；两项真实CUDA FP32双视图smoke
均成功，source batch4/8pair views、一次成功更新、一次BN更新，初始张量SHA均为
eb4fa2550e47d732c5b21acb6899ba5539856bee35b532da58006f56108001a8。
该smoke仅工程检查，不能称为研究成绩。完整训练状态以新研究目录status为准。
