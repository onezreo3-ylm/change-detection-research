# Valais FP32-val：第 38 轮后监护进程失联的恢复门

日期：2026-10-01。状态：**恢复方案，尚未发出 epoch38 再启动授权**。原 Task Scheduler 的 PowerShell 宿主与 launcher 在训练中途退出，LastTaskResult 为 0xC000013A（STATUS_CONTROL_C_EXIT）；温度守卫和训练进程随后一度仍活跃。谁发出了控制台中断尚未查明。为避免100轮完成后缺少 launcher 的 durable guard_completion，运行拥有者在完整第38轮 checkpoint 已落盘后，用 PID 绑定 helper 停止训练；第39轮部分训练丢弃，status=interrupted/manual_resume_required，guard 正常看到训练进程退出。原 task 注册仅在它已变 Ready、无 trigger、进程全部退出后删除。

[独立中断审计](../runs/valais_fp32val_supervisor_loss_20261001/audit.json) SHA256 d7f3296db88e935d84c6c1a11f9892cc7cb71a7ea807d45bcd3ec9f3e64a0eea，绑定停止前后状态、日志、epoch38 last/best、history、源码/数据审计和上次恢复授权共14项。last.pt epoch38 SHA256 2ea5b2db2ced33bef992e3078d5e6d7db19a40a3eda8c081d87ec196f86cce05；status=interrupted/epoch38 SHA256 6d1077bf41d95782a49bf98d67eeef317616a5f4cd9b829b65351a77631384db。Martigny test 未访问，旧 LEVIR held-out 不重开。

## 长时宿主试验

无 GPU/无数据访问的 Task Scheduler 宿主试验采用与研究任务相同的当前用户交互登录、无 trigger、无自动重试、隐藏窗口，并对该任务设置 DontStopIfGoingOnBatteries。第一版探针在第二次写心跳时因自身 File.Replace 传入空备份路径报错，LastTaskResult=1；这不是宿主耐久性证据，产物保留。修正版 v2 在[单独目录](../runs/valais_scheduler_liveness_probe_v2_20261001)每30秒写一次心跳，目标至少3600秒且最终 completion.json 表明 parent=svchost/data_accessed=false/gpu_accessed=false。**只有 v2 完成且时长≥3600秒，才允许下一次训练恢复**。试验失败须停下查明，不通过连续换任务名绕过。

## 唯一恢复授权与执行

[恢复 YAML](../configs/valais_epoch38_supervisor_resume_20261001.yaml)冻结研究 YAML/审计/last.pt/status 哈希、probe完成门、唯一任务名、授权文件路径及 test forbidden。[预检/授权脚本](../scripts/authorize_valais_epoch38_resume.py)必须只读复核完整 epoch38 checkpoint/有限权重/RNG/优化器/GradScaler/调度器、同一 study fingerprint、35项源码快照与实时 train/val 全量审计、无匹配进程、GPU UUID/低于85°C温度、probe 时长和无数据/GPU条件；通过后写一次性授权 JSON，不改 run 其它产物。

然后注册唯一、**无 trigger/无自动重试、DontStopIfGoingOnBatteries=True** 的 Task Scheduler 任务，由 svchost 作为宿主按需启动一次[恢复 wrapper](../scripts/launch_valais_fp32val_resume_epoch38_task_20261001.ps1)。wrapper 在启动 launcher 前核授权 token/status=interrupted/epoch38 和 last.pt 字节哈希、实际父进程身份；原冻结 trainer/launcher/pcdc 源码与研究配置不改，严格 --resume 从完整 epoch38 重新运行 epoch39。PID/创建时间/命令行/GPU UUID 绑定的新温度守卫使用独立 resume session，原历史守卫文件不覆盖。注册任务仅在运行自然结束/失败且已无其进程后删除，日志和授权记录保留。

若宿主又退出但 train+guard 仍活跃，拥有者先判断本次证据和[独立完成审计预注册](VALAIS_GUARD_COMPLETION_AUDIT_PROTOCOL_2026-10-01.md)是否足以在100轮后只读认证；绝不补造原 guard_completion，也不让守卫缺位的训练继续。任何 GPU 驱动错误、数值异常、源/数据漂移或温度守卫失效均停下复核，不自动再试。100轮最终 F1 仍需守卫完成与受守卫独立 clean-val 复核才能报告。
