# P1训练完成后的固定评价与残余诊断

2026-10-19，承接用户正式训练、恢复并继续及托管授权。C-S/A-S均已完成100轮。
本批只执行原P1_MANAGED_PROTOCOL_2026-10-14.md规定的45条件val及固定train残余诊断。
保持原数据、实例、强度、0.5阈值、FP32/CUDA、模型、首次最高clean-val F1选模和旧成果。
不训练新模型，不修改或伪装checkpoint字段，不开放Martigny或旧LEVIR test。

新入口configs/valais_p1_posttraining_v1_20261019.yaml，独立scripts.valais_p1_posttraining
及scripts.valais_p1_residual。原pcdc、trainer、旧evaluator和封存文件不修改。
先重新核验原封存链、两行100轮/275700成功更新/2205000 pair views、guard/status哈希；
从原train审计的排序源名和原标签分区重新生成100轮采样计划，与封存计划、两行实际流及
checkpoint history逐轮相等。验证新flavor、训练/采样签名、初始张量、原源码、完整数据
身份、配置、曝光、首次clean选模、有限权重/优化器。训练完成字符串不足以通过本门。

新val入口显式接受stratified_two_view_baseline_v1。复用原CandidateMetadata、bundle与
run_condition/summarize_conditions数学，独立验证新训练身份；不调用旧flavor的检查器来
强行接受新checkpoint。对原45条件完整覆盖，clean结果必须精确复现选中checkpoint的五项
指标和像素混淆计数；逐条件/源/strata计数审计后才接受汇总。验证文件与原acceptance、
实例、baseline、checkpoint/status/guard/exposure/采样/适配源码均封存哈希。

残余诊断直接引用原P0 manifest及visual_review，不重选样本、方向、强度、QA或用分数
再验收。新checkpoint另由本批封存链绑定，原P0 manifest及其旧模型绑定保持不动。
只读原64个Sion train样本、四族、等级1/3/5；保留相同u/v及所有方向正反结果。
每个新模型3136主预测+2重复/时相交换自检；合计6272主预测、6276总预测、1536条记录。
检查四格像素边际、退化四格相同、真实TP/FN/TN/FP交换、a/b/i及输出尺度，冻结门仍为
两模型各至少两个族绝对D/O F1差>=0.005并有非零有害交换。保存结果用独立计数重算审计。
富集train诊断不代表地域独立显著性，不从非零i宣称因果性或新算法。

启动准入：完整unittest，两个新模型在同一原P0首train样本的真实FP32 CUDA
重复/时相交换及四格退化/边际smoke。smoke为工程检查，研究评分全部单独完整执行。
GPU严格串行C-S val→A-S val→train残余诊断；不重复已完成行，不自动重试失败。
Val继续原85°C确认/86°C立即guard，残余诊断保持原P0 >=85°C立即停止合同。
每模型val固定6小时上限，残余诊断独立6小时上限；上限只是失败界限，不授权重复延长。

原P1实用筛查门完整写入本批YAML，未据新扰动结果修改。比较分别为C-S对原B0-E、
A-S对同采样C-S；clean/扰动空tile FPR分别核对，不混用总背景FPR。
已知clean门的负面结果如实保留，其他未算项不补作通过。完成后记录普通采样/增强的
收益边界及残余现象；无已准入下一实验时保存报告并暂停托管。P2仍需独立文献、机制、
计算pilot、同信息强对照及有限预算冻结，不由本批直接启动。

有效训练预算匹配不代表物理开销匹配：保留C-S95轮、A-S早前暂停如有未提交步骤和
A-S97轮至少1100步重算、恢复smoke的额外开销披露；精确废弃步数未知。
