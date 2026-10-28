# Valais FP32-val 第 27 轮断点：外部进程消失后的恢复协议

日期：2026-10-01。状态：**恢复前协议，不更改模型/训练数值设置**。原独立新 run 为 runs/valais_clean_fp32val_seed42_relaunch_20261001；目前 status.json 仍是 stale running/epoch27，而 launcher、训练和守卫均已消失。第 28 轮仅有半轮日志，不能保存或计入完成。完整 last.pt 为 format2/epoch27，SHA256 92e0d14acF9bad87cc6bfea61e36dc9d92a5aacc9c95a89e7b7b87e68b6bb607（大小写不影响字节值）；best.pt 来自 epoch26，中间 clean-val F1 0.3339481876。该分数不是 100 轮完成结果。

## 中断证据与边界

[21 项只读中断审计](../runs/valais_fp32val_relaunch_interrupt_20261001/audit.json) SHA256 5a7567f2265dede5f08f62260893d7f0a828f1f323294fd3d06de7881234d8c5，另存 stale status、守卫状态、训练日志与 Windows System Event XML。最后训练日志为 epoch28/train step1375，守卫最后在本地16:13:31记录77°C/decision ok；没有 traceback、高温暂停或 guard_completion。系统没有重启（LastBootUpTime 当日14:01:16）。

System Event 153/nvlddmkm 在16:13:39.333记录 GPU 驱动错误（\Device\Video3，GPUID:100），Codex 主进程在16:13:44建立。驱动事件、应用重启和训练进程树消失时间接近，但不能从现有记录证明哪个是首因；**不得写成模型数值失败或温度守卫触发**。若同类驱动错误再现，停止恢复并重新评估硬件/运行环境，不用不断重试掩盖故障。

## 恢复门

1. 复核审计绑定的21项文件原字节哈希、last.pt 的 epoch27/有限权重/RNG/优化器/GradScaler/调度器、新研究 fingerprint、全部源码快照及实时 train/val 数据审计；原 run 须无匹配的训练、launcher、守卫进程。Martigny test 未访问，旧 LEVIR held-out 不再读取。
2. 把 stale running **行政归类**为 interrupted，须先保存原 status 字节与 SHA（已在审计目录），另写[恢复记录](../runs/valais_fp32val_relaunch_interrupt_20261001/recovery_authorization.json)，记录原状态 SHA、新状态 SHA、审计 SHA、checkpoint SHA、进程缺席和唯一转换理由。该状态变更不是训练“自然报告 interrupted”，也不使第28轮部分训练有效。任何其它原文件禁止修改。
3. 只使用原[relaunch YAML](../configs/valais_clean_fp32val_seed42_relaunch_20261001.yaml)及原冻结 scripts/pcdc 源码，通过严格 --resume 从完整 epoch27 恢复第28轮。代码已经允许 interrupted 状态，但 checkpoint fingerprint、source snapshot、审计和数据任何一项不符即拒绝。新的温度守卫在独立 resume session 下绑定新 PID/创建时间/命令行、相同 GPU UUID 和85/86°C门；不覆盖旧 guard state/lock。无自动重试、无 CPU/设备/权重替代。
4. 以 Windows Task Scheduler 的**无 trigger、按需手动启动一次**、当前用户交互登录任务启动 launcher，减少 Codex 桌面应用进程重启对子进程树的影响。此前[基础 probe](../runs/valais_scheduler_probe_20261001.json)和[无 trigger probe](../runs/valais_scheduler_no_trigger_probe_20261001.json)均未访问 GPU/数据、成功运行且父进程为 svchost.exe；两项 probe 注册均已删除。实际任务名必须唯一、无周期/自动触发、禁并发实例、无自动重试、隐藏窗口，训练完成/失败后只删除这一个任务注册。此宿主隔离不能避免 GPU 驱动本身的错误，温度守卫仍必须附着。任务启动后核对 launcher 的实际父进程不属于 ChatGPT/Codex，再让 GPU 训练继续。

若上述任一门失败，保留现场并报告；不改冻结源码、训练超参数或已完成 checkpoint。只有100轮训练、温度守卫 durable completion 与受守卫独立 clean-val 复核都成功，才形成新的完整 Valais clean 结果。后续 B1/局部机制仍需单列协议。
