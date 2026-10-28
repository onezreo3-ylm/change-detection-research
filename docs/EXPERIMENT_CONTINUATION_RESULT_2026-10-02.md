# 本对话续接的执行结果（2026-10-02）

用户要求已落实到当前对话 `续接中断的实验进度`。详细故障证据与执行边界见[续接记录](EXPERIMENT_CONTINUATION_2026-10-02.md)。旧对话的最后失败为本机 `cockpit-cliproxy` 对 `localhost:65467/v1/responses` 的上游stream-open等待超时，返回504；其后重连被中断。本机证据还不能确定更深层的上游故障，当前对话可用也不代表代理超时已永久修复。

Valais全FP32 clean基线已经完成100轮，并经独立完成审计与受守卫验证精确复现epoch53 Sierre F1@0.5=0.4162248161980205。外层PowerShell完成门误判是另一个已经审计恢复的问题；原失败现场、旧AMP运行与缺失原guard_completion保持原样。

本次从扰动协议预检后的准确中断点继续，完成了训练样例的实际视觉复核、输入/输出像素哈希核对、validation候选清单生成、独立acceptance和评估配置。v1含1715个Sierre验证tile、45条件、77175个实例；train-only视觉证据和小目标/L5信息损失限制见[验收与评估协议](VALAIS_CORRUPTION_V1_REVIEW_AND_EVALUATION_2026-10-02.md)。准备产物仍标为candidate，评分许可由绑定相同哈希的sidecar单独提供。

重新执行的257项单测全部通过（64.111秒），真实全FP32 CUDA smoke也通过；smoke只作工程检查。预检成功后，[authorization](../runs/valais_corruption_protocol_preflight_20261002/authorization_resumed_20261002.json)冻结了输入、预检、视觉证据、独立审计和当前源码/守卫身份。

2026-10-02 **21:56（Asia/Shanghai）**，单次Windows任务 `Codex-Valais-Baseline-Corruption-ValV1-20261002` 启动已有模型的受守卫validation评分，无trigger、无重试，重复运行请求采用IgnoreNew；通过隐藏PowerShell将既有评估入口输出写入[运行日志](../runs/valais_corruption_protocol_preflight_20261002/scheduled_evaluation_resumed_20261002.log)。任务注册文件及XML、[实际进程身份](../runs/valais_corruption_protocol_preflight_20261002/execution_started_resumed_20261002.json)均已保存。观测到evaluator PID19068、父进程22304；这是启动时身份，后续必须核对创建时间和命令行，不能只用PID判断存活。

启动复核时进入`evaluating_corruptions`，已生成首批condition结果，GPU温度约62°C；正式结果尚未完成。本次续接已经启动计算，不将部分条件宣称为最终mF1。实时状态见[status.json](../runs/valais_baseline_corruption_val_v1_seed42_20261002/status.json)。只有45条件完整、clean F1精确复现、温度守卫完成且summary/guard_state哈希与guard_completion一致，才可报告正式完成。

原`levir-cd`小时heartbeat已通过app工具迁移至当前对话，保留频率并更新为本次validation检查点；健康或无实质变化时静默，完成/失败/重要新证据时在此通知。后续只继续该评分及必要结果分析、准备单列B1匹配协议。本批未启动B1长训或校正/门控模型；Martigny test未访问，已用LEVIR test不重开。
