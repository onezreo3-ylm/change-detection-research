# Valais 全 FP32 训练的独立完成审计

日期：2026-10-02。预注册只读恢复审计；只可在训练器和守卫退出后执行。不修改原 `runs/valais_clean_full_fp32_seed42_20261002`，不补写其 `guard_completion.json`，不执行模型推理或访问 Martigny test。

训练器已报告完整 100 轮、test=false，`pcdc train` 与 PID 温度守卫均以 0 退出，但外层 Task Scheduler wrapper 未写入持久完成标记。任务结果记录为 `launcher_exit_code=0`、`status=failed`，其报错是 `Training/guard/history did not complete under the frozen protocol`。Windows PowerShell 5.1 的 `@(Get-Content ... | ConvertFrom-Json)` 在该 `history.json` 顶层数组上得到单个嵌套数组；已在同一解释器只读重现 `Count=1` 而内层含 1–100 轮。这解释外层完成门的误判，不作为研究有效性的替代证据。

[审计器](../scripts/audit_valais_full_fp32_completion.py)必须独立核对：冻结配置/授权/任务 XML 的哈希与无 trigger/无重试设置；无残余 launcher、trainer、guard 进程；训练 status=completed/100/test=false；`history.json` 和 `last.pt` 依次为 1–100 轮，最佳首次 F1、best.pt、status 完全一致且模型张量有限；checkpoint 配置签名/源码快照/完整 train+val 输入审计一致；守卫 PID/创建时间/命令行绑定、最终 `training_process_exited`、温度 decision=ok 且无 lock/pause；训练/守卫日志显示正常退出且没有 traceback。确认原 durable marker 不存在，并绑定原始产物与审计脚本哈希。

全部通过后，只在新目录写 `certificate.json`，状态明确为 `audit_recovered_guarded_completion_only`，记录 `original_launcher_completed_normally=false` 与 `test_accessed=false`。如任一门不通过，拒绝签证。该证书只允许后续单列的受守卫 clean-val 工程复核使用；旧验证器若要求原 marker，仍不得绕过其门。第 53 轮开发验证 F1 是 checkpoint 选择结果，只有独立复核后才可作为完成基线报告；单城 Sierre 不提供跨地区泛化证据。
