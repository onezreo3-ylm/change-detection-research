# 固定开发验证适配工程状态

2026-10-26，Asia/Shanghai。两个独立新文件已实现评价适配与外部保护，尚未开始真实验证。
当前seed43→44训练、原pcdc核心和已封存分析规则保持不变。

scripts/evaluate_b0e_replications.py复用原严格CandidateMetadata/checkpoint检查、run_condition、
FrozenInstanceDataset与汇总算术。完成训练后才允许CPU准备；为外部监督建立明确标注的
派生status/completion sidecar，不修改原status/checkpoint、旧checker或真实训练flavor。
四角色证据必须有正确源码/runtime、不同PID、创建时间、可执行文件、完整命令与实际结束状态。
clean五指标/计数精确重放后，才按原45条件评价；逐源像素、标签、joint-strata、计数聚合和
模型状态不变均受检查，完成时重新从保存条件文件汇总并绑定实际外部证据。

scripts/fixed_development_evaluation_supervision.py只有固定evaluate模式，原物理GPU、
85/86°C温控、独立guard/watchdog和共享锁保持。全评价含工程/测试/失败/CPU/I/O最多3600秒，
并从该额度内留120秒用于评分后证书与分析，不扩大原48小时总账。
engineering_costs.json记录实际CPU阶段，seal/launch前settle进原共享账本；费用不能重复或清零。
最终结算超过子预算或共享预算即报错，保留实际已发生费用。

工程证据在runs/b0e_fixed_validation_engineering_20261026：最终11项CPU检查通过，
原共享ResNet18/FPN显式CPU FP32 synthetic forward/count smoke通过，0优化更新、2048像素守恒。
新身份fixture曾发现字符串Path类型错误，已在新适配内修复；未触发真实评价。
smoke模型实现随后未改，仅收紧监督器预算和全量日志验收，最终11项测试已覆盖这些变化。
verification_final.json绑定最终源码、配置、测试日志及先前smoke证据，不把工程结果写成研究效果。

**新增代码的全项目测试仍pending**。当前有真实CUDA训练，不并发旧全量测试。
未来实际全项目日志必须包含两个新模块且当前至少546项；旧526项日志不构成验收，会被入口拒绝。

## CUDA空闲后的执行

所有命令在项目根目录使用E:/Download/anaconda3/envs/geosa-basa/python.exe -X utf8执行。

1. 先确认原训练两个100轮及实际owner/worker/guard/watchdog正常终止，核对保存训练和原账本。
2. `-m scripts.evaluate_b0e_replications prepare`在CPU验证best/last全身份并生成sidecar。
   活动或失败的训练会被拒绝；不为查看准备状态重复加载checkpoint。
3. CUDA确认idle后运行`-m unittest discover -s tests -v`，保存UTF-8完整日志与实际子进程墙钟。
   再运行`-m scripts.evaluate_b0e_replications smoke`提交正式准备目录的显式CPU smoke。
4. 新监督入口`record-cost --phase full_tests --seconds <实际墙钟> --basis <测量来源>`记录测试开销。
   prepare/smoke CLI各自记录CPU阶段和启动余量；`settle-costs`只在无活动reservation时执行。
5. 新监督入口`seal --checks <实际全量日志>`后，只执行一次`launch --mode evaluate`。
   launch会结算seal费用并在原额度内扣除120秒收尾余量。运行中仅`monitor --mode evaluate`。
6. 两行45条件完成，实际四角色inactive且正常收尾后，执行评价适配的`certify`，生成明确的
   外部派生证明和runs/b0e_replication_evaluation_20261026/analysis_manifest.json。
7. 执行`-m scripts.analyze_b0e_replication analyze --manifest runs/b0e_replication_evaluation_20261026/analysis_manifest.json`。
   外层测量实际分析CPU墙钟，通过监督入口`record-cost --phase analysis ...`及`settle-costs`入账，
   最终同时核对3600秒子预算和172800秒共享总账，不因已经有分数跳过收尾费用。

失败保存现场后诊断，不循环重试、不覆盖完成结果。不访问test/Martigny，不添加模型、扰动或
阈值/强度搜索。本闭环不自动认证论文可投稿；总目标仍需贡献、模型覆盖和独立证据的完整审计。
