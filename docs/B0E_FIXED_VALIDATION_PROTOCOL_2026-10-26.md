# B0-E补充复现的固定开发验证协议

2026-10-26，Asia/Shanghai。承接B0E_REPLICATION_PROTOCOL_2026-10-26.md；两行训练进行中，
本文件固定其后适配范围与验收，尚未实现或执行新评价，不代表已有新分数。

## 训练身份先于评价

必须等43/44各100轮完成，且原CPUowner正常收尾、worker实际退出码0、独立guard/watchdog
均正常结束。只读核对原status/history、200轮精确clean流、初始化、首次最高cleanF1选模、
178000尝试和1424000pair views/行及AMP跳步算术。原checkpoint使用weights_only方式在CPU加载，
best/last的history前缀、配置、训练签名、11个原核心源码hash、数据内容与val清单逐项匹配。
新CPU工程修正与原prepared快照/最终training_authorization链保持可追溯。

原train_robust输出status与checkpoint不修改。外部监督证明使用**另外的、明确标注为派生证明的**
guarded_status与completion JSON。它们绑定原status字节、实际owner/worker/guard/watchdog记录及
原授权，不能把外部保护伪装为旧in-process wrapper。原matched_robust_baseline_v1 flavor保持。
可复用旧CandidateMetadata与严格_validate_checkpoints，并显式将其训练协议视图设为已核验的
同seed协议；仍须证明相对seed42只差声明的seed/身份，相对同seedB1只差第二clean view。
不得修改旧评价器、强度验收、旧身份封存或新checkpoint来通过检查。

## 输入、算术与运行

仅原开发val64/1024个256tile、原FP16/channels_last/batch32/0.5阈值、原45条件manifest/instances。
原train-only强度视觉验收继承，不制造新视觉验收。使用未修改的run_condition、
FrozenInstanceDataset、summarize_conditions与原计数定义；每条件先pooled像素再F1，45个F1等权均值。
原val文件字节与benchmark及新checkpoint保存身份一致，不能借评价读取oldtest或Martigny。

先在独立新文件实现CPU检查和fixture smoke，完整unittest在当前CUDA队列真正结束后运行。
新增代码尚未通过全量测试时明确pending，不能将训练前526项测试冒充其验收。
若需要真实CUDA兼容smoke，只用原固定首train样本，每行加载其真实选中checkpoint，
检查重复/时相交换、状态不变与计数守恒；无优化更新，不扩大train诊断或选择新样例。
它与完整工程/评价均计入原账本和本次3600秒评价预留，不能另加预算。

评价顺序43→44，一次执行。每行先在新checkpoint上精确重放其选中clean计数和五指标，
再执行全部45条件，保留每条件、每source、全部joint strata。任一条件覆盖、
source16tile覆盖、像素守恒、标签支持、分层或clean重放不一致就结束并保留失败，不循环重试。
不按第43行的效果决定是否执行44，不改选checkpoint或阈值。

独立CPUowner、温控guard、watchdog与物理GPUUUID/PID创建身份绑定，guard/watchdog就绪后才导入CUDA；
温度85°C三次确认/86°C立即、传感器3次失败、30秒保护心跳、900秒进展停止线保持。
全评价含smoke、启动、CPU、测试、I/O与分析，预留最多3600秒，且受原172800秒总账余额限制；
扣除先行实际工程费用后才分配worker墙钟。没有可用余额不启动，不重置账本。
本3600秒内另留120秒用于实际评分后证书与统计分析；这是内部额度分配，不增加总预算。

## 输出与完成证明

新目录runs/b0e_replication_evaluation_20261026，不覆写任何旧run。
原始外部进程证据保存在evaluation_session/：owner_state.json、worker_result.json、
guard_state.json和watchdog_state.json；必须绑定实际创建身份及执行源码/配置。
两个seed各自保存clean_metrics.json、conditions/、conditions.json、summary.json、status.json和provenance。

新summary可与旧受审计分析schema兼容，但其guard标记必须明确为独立外部监督的**派生摘要**。
保存原始外部状态；独立收尾后才写guard_completion、temperature_guard_state派生证明与
evaluation_completion.json。最后者绑定原始进程记录和本行结果，记录seed、test_accessed=false，
status=completed_independent_external_supervision_audit及roles/bound_sha256。
不能仅凭成功标题、锁或PID存在认证完成：要读实际worker退出、owner状态、两保护角色状态及温度证据。
如果OS不再提供owner退出码，区分“owner记录正常收尾/进程已退出”与“独立观测到owner退出码0”。

统计规则已封存在runs/b0e_replication_analysis_20261026/analysis_rules_seal.json，
实现为scripts/analyze_b0e_replication.py；历史7结果快照固定，43/44的B0-E是唯一新增两格。
分析manifest必须保留原7格逐字身份，绑定两项真实新评价完成证明，缺任一格拒绝三seed主结论。
检验B1−B0-E的原门、全部条件与三seed，并报告相同source抽样的20000次区间。
该区间条件于现有训练seed与开发源，不是训练seed总体或地理独立区间。
无论门是否通过，本批停止追加训练；论文总目标仍需单独审计贡献、覆盖与独立证据。
