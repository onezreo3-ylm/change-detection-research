# Valais FP32 验证 clean 基线：系统关机后新目录重启

日期：2026-10-01。状态：**独立新尝试，未把上次不完整运行续为同一实验**。原[数值协议](VALAIS_CLEAN_FP32_VALIDATION_PROTOCOL_2026-09-30.md)的模型、数据、seed42、100 轮、训练 AMP、验证 FP32、阈值、选模与温度政策保持一致；只新增上次操作系统关机的故障绑定与新的运行身份。原 runs/valais_clean_fp32val_seed42_20260930 保留，status 仍为 running/epoch0 的中断现场，且没有完整 last.pt；不得在该目录强行 --resume 或改写原日志。

[原启动故障审计](../runs/valais_fp32val_launch_failure_20261001/audit.json)记录：epoch1 约 step275 后，launcher/训练/守卫同时消失，没有 traceback、温度报警或 checkpoint。[OS 关机补记](../runs/valais_fp32val_launch_failure_20261001/os_shutdown_addendum.json)绑定该审计并核对 Windows System 事件：00:49:19 ID1074 由 winlogon.exe/NT AUTHORITY SYSTEM 发起 Shutdown Off、00:49:20 ID6006、00:49:25 ID109、00:49:26 ID13；最后训练日志约00:48:56，守卫最后00:49:06/43°C/decision ok，下一次开机14:01:16。可判断为外部系统关机中断，而非本次 FP32 验证方案的数值失败。若系统再次关机/停电，当前守卫不能替代硬件持续供电；只从新目录中的完整 epoch checkpoint 走严格受守卫续跑。

新[配置](../configs/valais_clean_fp32val_seed42_relaunch_20261001.yaml)绑定上述 addendum 的 SHA256 fb5fb26a7bf98ca92effb7cb3bffc8e1a8a8bd5920be0db97fb6b16a756af3ea，输出至 runs/valais_clean_fp32val_seed42_relaunch_20261001。运行仍从相同 seed42 随机初始化，**不加载上次不完整 epoch 的任何张量或优化状态**。新的配置与全部脚本须经过独立全量 unittest、训练 AMP/验证 FP32 的真实 CUDA smoke、数据/源码/旧故障哈希核验、GPU UUID/温度预检，随后仅启动一个 PID 绑定温度守卫的研究进程。Martigny test 继续关闭，旧 LEVIR held-out test 保持已使用状态。完成研究结论仍需100轮、durable guard_completion及独立受守卫 clean-val 复核。
