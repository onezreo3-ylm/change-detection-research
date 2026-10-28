# Valais 匹配对照：阶段结果与后续步骤

日期：2026-10-03。进度证据时间为北京时间 **10:01:28**，随后日志确认第60轮正常推进。
本记录响应用户“是否取得预期成果，继续开展下一步”。当前批次仍按2026-10-02冻结协议执行。

## 是否取得预期成果

**尚不能确认扰动监督取得预期收益。** B1尚未完成100轮，正式45条件验证和B0-E对照
尚未开始。完成的工程检查与B0研究结果可确认；B1中途best不能作为最终比较。

| 项目 | 已核对结果 | 当前状态 |
| --- | ---: | --- |
| B0 clean F1 | 0.4162248161980205（41.62%） | 已完成100轮、独立验证及45条件评分 |
| B0 corruption mF1 | 0.37004352259315076（37.00%） | 45条件等权平均，非pooled F1 |
| B1 completed epochs | 59/100 | 第60轮训练中 |
| B1 best clean F1 so far | 0.3927471582466876（39.27%），epoch47 | 中途best，不是最终成绩 |
| B1 best对应clean Recall | 0.38715220386801996 | 中途值 |
| B1 corruption mF1 | 待评分 | 训练完成后执行 |
| B0-E | 待启动 | B1正式验证完成后执行 |
| 完整比较门 | 待判定 | comparison.json / completion.json尚未产生 |

B1当前best clean F1比B0低 **2.35个百分点**；对应Recall低 **1.96个百分点**。
若最终最佳仍停在该水平，将不满足冻结的clean F1和Recall保护门。后续epoch仍可能
改变best，因此目前不能宣告本批成功或最终失败，也不能仅据损失下降判断研究收益。

最新进度快照与完整性检查：
[audit.json](../runs/valais_matched_progress_audit_20261003/refresh_20261003_100128/audit.json)。
该目录保留实际读取的history、guard、queue status、exposure及进程快照；活动文件按顺序
复制，属于带时间戳的阶段审计，不是整批完成证明。早先09:48审计保持原样。

## 当前执行核验

- 59个静态封存文件SHA256全部一致，授权文件SHA256仍为
  `5006e7e680e80d7dfae2297973e2fcddd22e1999e8af08ff27704c421cc07062`。
- Windows任务为Running；队列及B1进程的PID、创建时间、父进程和完整命令行与启动记录一致。
- 温度守卫为monitoring / ok；快照80°C、历史观测最高83°C，传感器失败0。
- B1正式扰动验证、B0-E及完整比较尚未开始或完成。没有新模型推理或test访问。
- status.json记录阶段切换，训练期间其时间戳不逐轮更新；进度以history、日志及guard为据。

## 本轮完成的下一项工作：更正保存计数的分组分析

审计发现旧分析脚本的`pooled_counts`对输入按tn、fp、fn、tp分别遍历。当传入一次性
generator时，tn遍历消耗全部记录，后三个字段错误归零。旧analysis.json的全部255个
方向、时相及global/local分组因而给出错误的F1/Recall/Precision/FPR和净FP/FN。

受影响范围为旧报告的`subgroups`派生字段。clean、每条件、family/level、总体mF1、
pooled corruption和空/非空分层使用可重复遍历的容器，已从原保存计数重新核对。
队列的筛选门以独立逐字段累计读取原评价计数，也不使用旧subgroups。

独立更正工具：[analyze_valais_saved_subgroups.py](../scripts/analyze_valais_saved_subgroups.py)。
路径配置：[audit YAML](../configs/valais_saved_subgroup_audit_20261003.yaml)。
产物：[report.json](../runs/valais_saved_subgroup_audit_20261003/report.json)和
[subgroups.csv](../runs/valais_saved_subgroup_audit_20261003/subgroups.csv)。

新工具对每个条件的三个分组轴检查：源覆盖一致，标签像素一致，组计数之和精确恢复
对应condition与同一批source的clean计数。45条件、255分组全部通过；255个旧分组均更正。
总体clean F1、corruption mF1和pooled指标与旧记录一致（浮点汇总允许1e-15误差）。
旧脚本、旧封存分析及已有结果文件保留；后续分组解释使用本次更正报告。

更正后的例子均以**同一组source**的clean为参考：

| 条件及分组 | 同组clean F1 | Perturbed F1 | 净FP | 净FN |
| --- | ---: | ---: | ---: | ---: |
| color_bias L5 / global | 0.447478 | 0.107660 | -50243 | +62535 |
| shadow L5 / global | 0.390704 | 0.104593 | -74044 | +55321 |

这些保存计数显示上述条件伴随明显漏检增加。它们不证明某个校正动作能恢复变化，也不
用于事后删条件、降强度或改阈值；全套45条件继续保留。

验证记录：[verification.json](../runs/valais_saved_subgroup_verification_20261003/verification.json)。
全量命令`python -m unittest discover -s tests -v`完成273项测试：270通过，3项CUDA测试
因测试进程显式设置`CUDA_VISIBLE_DEVICES=-1`而跳过，耗时41.166秒。六项新回归检查覆盖
一次性迭代器、非法计数、分组恢复、源覆盖、标签变化及guard不一致。随后新工具在真实
已保存验证计数上完成CPU CLI smoke；这验证统计工具，不增加模型研究结果。
原启动前267项全量测试和真实CUDA双视图smoke证据保持。

## 已授权队列继续执行的顺序

1. B1完成原定100轮，按首次最高clean-val F1选best。
2. B1重放该best的clean Sierre指标，再评同一45条件/77175实例。
3. B0-E从匹配初始化独立训练100轮，再执行相同clean重放和扰动验证。
4. 核对两项训练的真实source顺序、初始化、1102500个source、275700次成功更新、
   2205000个pair views、零AMP跳步，以及guard/数据审计/评价哈希。
5. 读取自动生成的comparison.json、RESULT.md、completion.json，并核对完成哈希。

当前GPU由该队列使用。本轮新增内容为独立统计工具、测试、保存计数报告及此文档。
本轮不改变现有训练/评分数学或封存协议。若技术失败或守卫停止，保留现场并先审计，
缺失完成产物不能解释成方法效果的负结果。

## 完整结果的判定规则

以下门在B1新得分前已冻结，B1须对B0和B0-E**分别全部满足**：

| 指标差值（B1−reference） | 接受条件 |
| --- | ---: |
| corruption mF1 | ≥ +0.005 |
| clean F1 | ≥ -0.005 |
| clean Recall | ≥ -0.01 |
| pooled corruption Recall | ≥ -0.01 |
| pooled corruption空标签tile背景FPR | ≤ +0.0002 |

F1提高、误报降低或clean-corruption差距缩小，均不能替代Recall和clean保护门。
本批仅seed42、Sierre单城；缺少可信空间源ID，不做tile bootstrap或统计显著性声明。

## 完整门出来后的条件分支（准备方案）

### 全部门通过

下一候选是seed43、44各一对独立的B1/B0-E确认，加已有seed42形成三组匹配运行。
主归因对照是每个seed的B1−B0-E；现有B0只作为seed42问题参照，不能冒充三seed匹配B0。
逐seed报告效果与损害，禁止用均值掩盖单seed明显损害。

保持模型、数据切分、归一化、阈值、损失、采样、100轮预算、双view前向、BN行为、
选模规则及45条件验证清单不变。新增实验seed及其影响须单列，训练扰动stream设置
也须在复验协议中明确；任何stream变化要有新的兼容性核验及review sidecar。
当前wrapper硬性要求seed42，不能仅改YAML便宣称可复用。待本批结束后再独立准备
复验入口、配置和新封存证据，并完成全量测试和真实CUDA smoke。
本文件是条件方案，不启动四项新长训练。

### 任一效果或保护门未通过

先用完整B0/B1/B0-E保存计数作误差归因：clean检出不足、扰动新增FN、净FP变化、
空/非空tile、family/level及已更正的同源方向/时相/global-local分组。
汇总净变化不能揭示个体像素由错变对与由对变错的全部交换；像素动作收益需要单独证据。
训练诊断继续复用固定8张train记录；新增可视化仍只在train并遵守图像传输预算。

再根据归因单列普通clean基线改进候选。类别平衡只是可能的普通对照；2026-09-30
旧方案的自动触发情形是100轮best仍TP=0/F1=0，当前B0不符合该情形。
不得把当前clean偏弱误写成已满足旧全背景失败门，也不直接升级B2或校正/门控模型。

### 证据或完整性核验不通过

先恢复可解释的运行/数据/计数证据，记录异常和影响范围，再决定后续实验。
Martigny test保持封闭，已使用的LEVIR test继续退出开发。当前准备与条件方案均不触发test评价。
