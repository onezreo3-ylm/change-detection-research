# Valais 匹配对照最终结果、误差归因与下一项普通基线

日期：2026-10-04。回应“是否取得预期结果，执行下一步”。

## 结论

**本批训练和验证已正常完成，B1 未取得预先要求的整体扰动收益。**
恢复队列于北京时间 **13:05:30** 完成，任务正常退出。独立完成审计于 **13:36:05**
通过；随后已执行 B0/B1/B0-E 的完整保存计数归因。这是 seed42、Sierre 单城的
描述性负结果，不能据此断言所有增广方案均无效。

| 模型 | Clean F1 | Corruption mF1 | Clean Recall | Pooled corruption Recall |
| --- | ---: | ---: | ---: | ---: |
| B0 | 41.62% | 37.00% | 40.67% | 34.23% |
| B1 | 39.27% | 36.70% | 38.72% | 35.09% |
| B0-E | 42.06% | 37.55% | 42.51% | 36.31% |

主要归因对照 B1−B0-E：clean F1 **−2.78 个百分点**、corruption mF1
**−0.86 个百分点**、clean Recall **−3.80 个百分点**、corruption Recall
**−1.21 个百分点**。相对 B0 的 mF1 也下降 **0.31 个百分点**。

冻结门要求：相对 B0 与 B0-E 分别达到 mF1 增益至少 0.5 个百分点，clean F1
下降不超过 0.5 个百分点，clean/corruption Recall 下降各不超过 1 个百分点，
pooled corruption 空标签 tile FPR 增加不超过 0.0002。B1 对 B0 未通过前三项；
对 B0-E 未通过前四项。空标签 FPR 保护门均通过，不能抵消其他门的失败。

原结果：[comparison.json](../runs/valais_matched_study_v1_20261002/comparison.json)。
原筛选门和阈值保持原样；下面的普通基线提案不回溯修改本批结论。

## 独立完成核验

[完成审计](../runs/valais_matched_completion_audit_20261004/audit.json)核对了
**5,436 个不同文件**：原 59 个静态封存输入、恢复授权的 140 个封存输入、两项
完成记录、训练历史/预算/选模、真实 source 顺序、guard 及评价 provenance。
对 provenance 所绑定的 Sierre 原文件仅计算字节哈希，没有新增图像解码或推理。
训练数据使用已封存的完整 train/val 审计及 checkpoint 证据，未重新扫描原 train。

| 核验项 | B1 | B0-E |
| --- | ---: | ---: |
| 完成训练轮数 | 100 | 100 |
| Best epoch（首次最高 clean-val F1） | 47 | 53 |
| 成功更新 | 275,700 | 275,700 |
| AMP 跳步 | 0 | 0 |
| Source draws | 1,102,500 | 1,102,500 |
| Pair views | 2,205,000 | 2,205,000 |
| 实际 source 顺序 | 相同 | 相同 |
| 初始化张量摘要 | 相同 | 相同 |
| 有效完成 guard | completed / ok | completed / ok |

B0-E 的完成守卫为 `temperature_guard_resumes/session_0001` 内的记录；根目录旧
手动暂停快照保留。恢复前 31 轮的 loader RNG/顺序重建证据和恢复后的实际顺序均已核验。
两项训练完成守卫记录最高观测温度 83°C、传感器失败 0；这仍是批次之间的软件监测。
B0-E 时间/峰值记录只覆盖恢复后的阶段，不能与 B1 全程时间直接作效率比较。

三项评价均完成相同 45 条件；summary/status/guard 哈希一致。最终完成摘要、恢复
镜像、原比较得分与筛选门重算一致。Martigny test 未访问。

## 已执行的下一步：保存计数误差归因

独立工具：[analyze_valais_matched_errors.py](../scripts/analyze_valais_matched_errors.py)。
配置：[error review YAML](../configs/valais_matched_error_review_20261004.yaml)。
结果：[report.json](../runs/valais_matched_error_review_20261004/report.json)。

已覆盖三项模型的 **135 个条件报告、765 个同源分组**，包括 9 类×5 级、空/非空
tile、clean 标注面积、direction、target date 和 global/local。每个条件各分组轴均
恢复原始总计数及同一批 source 的 clean 计数，共 405 项轴划分恢复检查通过。
本轮累计函数复用 10 月 3 日更正工具的一次遍历实现，不使用旧 generator 分组结果。

导出包括 [conditions.csv](../runs/valais_matched_error_review_20261004/conditions.csv)、
[families.csv](../runs/valais_matched_error_review_20261004/families.csv)、
[levels.csv](../runs/valais_matched_error_review_20261004/levels.csv)、
[subgroups.csv](../runs/valais_matched_error_review_20261004/subgroups.csv)、
[subgroup_totals.csv](../runs/valais_matched_error_review_20261004/subgroup_totals.csv)、
[clean_size_groups.csv](../runs/valais_matched_error_review_20261004/clean_size_groups.csv)、
[clean_by_source.csv](../runs/valais_matched_error_review_20261004/clean_by_source.csv) 和
[comparisons.csv](../runs/valais_matched_error_review_20261004/comparisons.csv)。

### 1. Clean 检出不足，而且 B1 的净损失主要集中于前景面积较大的 tile

| Clean 计数 | B0 | B1 | B0-E |
| --- | ---: | ---: | ---: |
| TP | 124,266 | 118,287 | 129,883 |
| FN | 181,265 | 187,244 | 175,648 |
| FP | 167,313 | 178,539 | 182,209 |
| 有变化但 TP=0 的 tile / 275 | 176 | 150 | 157 |
| 有变化且完全无正预测的 tile | 164 | 132 | 142 |
| 空标签且有正预测的 tile / 1,440 | 39 | 73 | 52 |

B1 相对 B0-E 的 TP 净减少/FN 净增加 **11,596 像素**，FP 净减少 3,670。
B1 在 11 个原 TP=0 的正 tile 上开始有 TP，同时 4 个原有 TP 的正 tile 变成 TP=0；
因此 TP=0 tile 数下降仍可伴随总体像素 Recall 下降。

| 单 tile 标注前景像素数 | Val tile 数 | 标注正像素 | B1 FN | B0-E FN | B1−B0-E FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–64 | 23 | 471 | 436 | 440 | −4 |
| 65–256 | 60 | 9,772 | 8,341 | 8,613 | −272 |
| 257–1,024 | 119 | 67,480 | 44,310 | 44,652 | −342 |
| 1,025–4,096 | 59 | 106,152 | 63,483 | 62,546 | +937 |
| >4,096 | 14 | 121,656 | 70,674 | 59,397 | **+11,277** |

最后一层贡献了 clean 净增 FN 的 **97.25%**。这里的面积是每个 tile 的标注像素总量，
不等于单栋建筑大小，也不能由此推断具体建筑的视觉可恢复性。小于等于 256 像素
的 tile 召回仍弱，但其正像素仅占总量 3.35%；B0-E 在该层的 FN 占全部 clean FN
5.15%。少量细边缘标签不能解释总体低分或 B1 本次主要退化。

固定八张 Sion train 诊断继续沿用既有审阅。没有改标签、剔除困难样本或重新选
“好看的”样例。空标签 clean FP 在 B1 为 72,383、B0-E 为 63,711；两者总体 FP 的
方向不同，说明空/非空分层必须保留。

### 2. 扰动下仍有新增漏检，B1 的局部收益不足以通过总体门

下面净值均相对**各模型自己的 clean 计数重复 45 次**；不能把两模型净退化差直接
当成两模型扰动后绝对差。

| 45 条件累计 | B0 | B1 | B0-E |
| --- | ---: | ---: | ---: |
| FN | 9,043,263 | 8,923,827 | 8,757,195 |
| FP | 6,807,676 | 7,700,624 | 8,125,419 |
| 相对自身重复 clean 的净 FN | +886,338 | +497,847 | +853,035 |
| 相对自身重复 clean 的净 FP | −721,409 | −333,631 | −73,986 |

B1 的新增 FN 少于 B0-E，但其扰动后绝对 FN 仍多 **166,632**，绝对 FP 少 **424,795**。
clean 起点下降造成了部分“分差缩小”，必须结合绝对 mF1/Recall 和 clean 保护门解释。

| 扰动族 | B0 mF1 | B1 mF1 | B0-E mF1 | B1−B0-E（百分点） |
| --- | ---: | ---: | ---: | ---: |
| Brightness | 41.23% | 38.52% | 41.58% | −3.06 |
| Contrast | 35.69% | 36.12% | 36.69% | −0.56 |
| Gamma | 35.72% | 36.31% | 37.46% | −1.15 |
| Color bias | 28.08% | 30.13% | 25.84% | **+4.29** |
| Haze | 36.73% | 35.68% | 36.03% | −0.35 |
| Shadow | 32.04% | 36.19% | 34.54% | **+1.65** |
| Noise | 41.49% | 39.51% | 41.71% | −2.20 |
| Blur | 40.49% | 38.58% | 42.14% | −3.56 |
| JPEG | 41.57% | 39.23% | 41.99% | −2.76 |

B1 在 2/9 族、11/45 条件上高于 B0-E。按级别均值，L1–L4 均落后，L5 高约
1.16 个百分点。完整清单继续保留；不根据本次得分事后删族、降验证强度或改阈值。

### 3. 同源分组揭示差异，但不能替代总体评价

global/local 下 B1 与 B0-E 的同一验证实例源集合相同。池化后：global F1 为
34.72% / 34.04%，Recall 为 32.15% / 31.80%；local F1 为 38.71% / 40.15%，
Recall 为 38.18% / 41.04%。这表示本次 global 有局部收益、local 有损害。
各方向和时相的同源 clean 参考、净 FP/FN 均已保存，不能使用总体 clean F1
替代每个子组参考，亦不能把不同方向间的源分布差异解释为扰动因果效应。

保存混淆计数给出的是净变化，无法恢复全部像素“错变对/对变错”交换；这些结果
不是校正动作的收益证据。45 次出现的同一 tile 不成为 45 个独立样本。本批没有
tile bootstrap、空间独立区间或统计显著性声明。

## 下一项具体候选：B0-E-S，普通 clean 分层采样对照

基于以上归因，下一实验候选优先检查 **clean 正类曝光**。Sion train 有变化 tile
1,046/11,025（9.49%）；原均匀、无放回 batch=4 的全空批次概率为 67.11%。这是
选择普通采样对照的依据，并未证明曝光不足是低分的唯一原因或保证采样会提高分数。

这项候选作为本次负结果后的新提案单列。9 月 30 日旧方案的自动触发条件是
100 轮 best TP=0/F1=0；当前结果不满足该条件。

提案：[valais_b0es_candidate_proposal_20261004.yaml](../configs/valais_b0es_candidate_proposal_20261004.yaml)。
它绑定本次误差报告、原 train 标签统计、完整数据审计、B0-E 评价与原 base YAML。

- 主对照为当前 B0-E；保留 seed42 随机初始化、原模型/损失、FP32、阈值 0.5、
  clean-val 首次最高 F1 选模及 100 轮预算。
- 每 source 仍为两个相同 clean view，以一次联合前向维持当前 BN 更新方式。
- 唯一拟改变项为 train source 采样：前 2,756 个完整 batch 各 1 正/3 空；末尾
  单例 batch 为正。每轮 2,757 次正类抽取、8,268 次空类抽取，共 11,025 source，
  2,757 更新、22,050 pair views。
- 正类完整列表独立打乱并循环取样（每个正 source 每轮出现 2 或 3 次）；空类每轮
  无放回抽取。仅从 train 标签定义正/空，不用 val 分数选择样本。精确种子模板、
  cycle/batch 索引、批内与批间顺序见 YAML；采样使用局部 RNG。
- 总 source draws、更新及 pair views 与 B0-E 相同，但 source 顺序、正类占比和
  source 重复次数有意不同，必须单列而不能声称与均匀对照 source 完全匹配。
- 为本候选起草的描述性门：相对 B0-E clean F1 至少 +0.01、Recall 至少 +0.02，
  corruption mF1 下降不超过 0.005、corruption Recall 下降不超过 0.01，clean 与
  corruption 空 tile FPR 分别增加不超过 0.0002。这些是**本次结果之后的提案门**，
  尚未用于选模或评价新运行，也不改变已完成 B1 的冻结门。

当前已完成误差归因和可审阅采样协议提案。采样器及新训练尚未启动；按 10 月 3 日
条件分支，本阶段先提交普通 clean 基线候选。进入其实施批次前，需要核对采样流/
重复数和初始化，完成全量测试及真实 CUDA 双 clean view smoke，再建立独立输入
封存、温度守卫和启动记录。若普通基线确有收益，随后单列独立 seed 确认及重新匹配
增广控制；当前结果不足以推进校正/门控机制。

## 工程验证与保存

[verification.json](../runs/valais_matched_error_verification_20261004/verification.json)：
`python -m unittest discover -s tests -v` **290 项全部通过、0 跳过，67.933 秒**。
八项新增回归检查覆盖 TP=0 与无正预测的区别、同源 label/coverage、净计数与 tile
检出交换、面积划分、macro/pooled F1 区别、重复 clean 参考、实例覆盖/重复、输入
哈希变化及原规范的 exposure 摘要编码。

新归因 CLI 在三项真实保存验证计数上完成 CPU smoke：135 条件、765 分组均通过。
完成审计 CLI 也已实际执行。本轮新增统计工具、提案和报告，原训练数学/代码、封存
配置、checkpoint、暂停备份、完成结果均保持；验证后原 59/恢复 140 个封存文件再次
核对一致。新报告各输入及九个输出文件哈希已核验，产物保存于独立目录。
