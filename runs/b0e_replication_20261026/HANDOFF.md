# 当前论文证据任务

总目标active，用户要求持续改进至可支撑投稿论文。最新规范docs/RESEARCH_SPEC.md；
docs/PUBLICATION_EVIDENCE_PROGRESS_2026-10-26.md说明当前实际状态与下一步骤。

当前正式B0-E43→44队列正在运行；只用scripts.b0e_replication_supervision monitor --mode train
获取紧凑状态。不要启动第二个GPUworker，不编辑已封存源码或配置，不重复正常检查。
无新自动化调度。526项完整测试、修正CPUsmoke及一次真实pilot已经通过。
原CPUsmoke失败和准备快照保留，修正见cpu_smoke_correction.json。

训练完成后独立核对保存训练、初始化/200轮流、保护、退出与账本；再封存原45条件开发
评价的新身份适配，串行评分、三seed配对分析及论文表格。未完成部分保持pending。
不能据已有单seed失败复活旧候选，不能把三seed基线闭环当整篇论文贡献成立。
原48小时共享账本仍生效，所有开销累计，无自动延长；test/Martigny关闭。

当前续接已完成CPU分析准备：scripts/analyze_b0e_replication.py、
configs/b0e_replication_analysis_20261026.yaml；九项新CPU测试和七历史结果重放通过，
规则与原七格封存在runs/b0e_replication_analysis_20261026，不再编辑这些已封存分析文件。
其新代码全量测试仍pending，等CUDA真正idle后与评价适配一起跑；不得并发526项旧全量测试。
新评价实现已准备，详见docs/B0E_FIXED_VALIDATION_ENGINEERING_2026-10-26.md；全量测试仍pending，
尤其区分外部派生guard证明与原status、不改旧checker；两行新completion必须包含actual
evaluation_session进程记录，不能仅成功标题。当前11项CPU检查及原架构CPU smoke通过；
GPU空闲后按工程runbook完成实际全量测试和一次固定评价。训练期间可推进论文与独立证据工作。

本轮新工作：runs/manuscript_working_draft_20261026/manuscript.md及五张机读/Markdown表已生成，
缺失B0-E43/44保留pending；均值SD/效用/转换算术经独立方式核对，无新推理或重采样。
文稿明确356源actor与445源基线不可直接作选择器归因，Phase7参照按原B*规则取更优常数，
而不是预先固定G0身份。该稿不是投稿就绪认证，旧稿与冻结结果保留。

独立数据元数据审查已结束：docs/INDEPENDENT_DATA_METADATA_SCREEN_2026-10-26.md。
S2Looking因全球来源/侧视配准范围暂不优先；SECOND列为后续发布说明、标签schema与来源候选，
尚未准入。一次HEAD得到200 HTML2466字节，不是ZIP大小；未读body/下载/解压/看raw图或标签。
后续若继续：先冻结有界公开发布页/元数据读取，核对building编码、独立change-mask/同类拆建、
数据使用说明、包身份与官方split；不把语义不等式/XOR自动当建筑变化，不提前打开test。
不重复旧文献/正常工程检查，不加新GPU任务。当前15:28UTC快照seed43已提交16轮、GPU76°C。

2026-10-27 00:11更新：四个原创建身份仍active，seed43完成25轮、正推进后续，GPU77°C，最高78°C。
新工作包runs/publication_evidence_audit_20261027_r3：两张已保存结果图、20项事后实验族登记、
原始CSV/JSON/图注与SECOND归档失败审查。不要重复画图或重跑正常检查。
SECOND目录读取成功但训练RAR分段修正第4段遭quota-exceeded；partial保留、最终校验未完成，
新数据未准入。严格不继续同一归档传输，不解压partial、不打开嵌套test或解码标签。
详见docs/SECOND_QUARANTINE_OUTCOME_2026-10-27.md。公开building编码和数据使用条件仍unknown。
本轮五项归档CPU测试通过；旧526项全量不作新代码验收，当前全套须等CUDA真正idle后执行。
546为评价验收最低要求，不是实际最新总测试数量；原评价runbook及一次正式评分顺序保持。
训练源/配置、原分析规则、共享账本均未被并发修改；本轮CPU工作计入重叠owner墙钟。

用户后续对路线问题明确回答“优先提出并验证新方法”。以docs/METHOD_FIRST_CONTINUATION_2026-10-27.md
为新主线，不以继续润色负结果稿替代目标。B0-E控制仍照常完成；旧失败路线不重开。
已实际完成train-gate空间硬动作余量筛查，71/18两组资源门通过，但全部是真值知情诊断，
不是新方法分数。原8项CPU测试、合成smoke与138项独立有理数断点核对通过；成果
runs/spatial_action_feasibility_20261027，不重复计算。
当前唯一机制候选为无动作参考统计重放的局部feature动作。独立pcdc/local_feature_replay.py
及完整FPN64/8组CPU合成原型通过9项CPU检查和smoke；旧model.py/phase5_models.py及活动49项
训练绑定未更改。普通输出掩膜同样消除支持外变化，故当前不声称任务改善或创新成立。
结果runs/local_feature_replay_cpu_20261027；GPU/FP16、真实数据样本绑定和最近邻全文/代码核查
均pending。不要将此CPU原型直接接入生产训练，不启动第二GPUworker。原全量测试需GPU idle后执行。
