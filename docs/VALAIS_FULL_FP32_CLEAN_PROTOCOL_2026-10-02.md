# ValaisCD clean 基线：训练与验证均为 FP32

日期：2026-10-02。状态：独立数值协议，待全量工程门与一次性受守卫启动。它不续训或改写失败的 AMP 运行，不把中间 checkpoint 当作 100 轮结果。

## 数值修复依据

[第 46 轮一次性隔离回放](../runs/valais_train_nonfinite_replay_epoch46_20261002/replay.json)从原 epoch45 checkpoint 恢复了完整优化/RNG 状态；五个训练累计 loss 点与原日志六位数一致，step480 的 AMP 梯度跳步也一致。首个非有限 loss 精确发生在 **step1104**。输入和标签有限；最早被叶模块钩子观测到的异常为 `decoder.refine.3.0` 卷积 FP16 输出中的一个 `+Inf`，最终 AMP logits 有 65,536 个 NaN。相同模型/批次的一次 FP32 前向 logits 与 BCE/Dice 均有限（total loss 0.000313582）。[守卫完成记录](../runs/valais_train_nonfinite_replay_epoch46_20261002/completion.json)确认该回放正常退出、原运行未修改、test 未访问。钩子只定位最早**观测到**的异常输出，不能单凭它证明更上游的根因。

此前 epoch34 的验证 AMP 故障也在同批 FP32 前向下有限。因此新[配置](../configs/valais_clean_full_fp32_seed42_20261002.yaml)相对原 Valais clean 配置仅更改训练 `amp: false`，并设置独立实验名和输出目录；验证同样 FP32。模型、随机初始化、seed42、train/val 原始切分、11025/1715 对、标签、归一化、batch4/eval32、AdamW、余弦调度、100轮、损失、阈值0.5、clean-val F1 选模和 test=false 均保持。完整协议不同，不把它与旧 AMP 运行拼接。

## 工程与完成门

1. [授权器](../scripts/authorize_valais_full_fp32.py)逐字段确认除名称/输出/训练精度外与源配置一致，核来源审计及实时 train/val 全量文件字节，绑定失败审计、一次性回放、源码、单测日志和真实 CUDA batch4 FP32 smoke。缺少 GPU、哈希变化、其他 Valais 训练/守卫进程或 GPU 达 85°C 时拒绝。
2. 使用固定任务名 `Codex-Valais-FullFP32-Seed42-20261002` 注册无 trigger、无自动重试、隐藏、禁止并发、不断电池切换的 Task Scheduler 一次性任务。[任务 wrapper](../scripts/launch_valais_full_fp32_task_20261002.ps1)核对授权与宿主身份，通过现有 `pcdc train` 和 PID/创建时间/命令行绑定的温度守卫训练。连续确认 85°C、单次 86°C 或连续传感器错误停机。每轮保存 RNG、模型、优化器、调度器、GradScaler 及 best/last；没有任何自动恢复。原失败运行保留。
3. 只有 100 轮、trainer status=completed/test=false、守卫正常退出、history100、来源哈希不变且 wrapper 正常收尾，才写持久化 `guard_completion.json`。若宿主再次失联，即使训练随后完成，也只可按新预冻结的只读审计处理，绝不补造原标记。
4. 完成后另以受守卫的 FP32 clean-val 独立复核选定的 best.pt，要求 pooled F1@0.5 精确重现，再判断正类检出是否足以进入 Valais train-only 扰动 QA。若仍全背景，按已有普通类别平衡门另立对照。Martigny test 继续封闭；旧 LEVIR test 不重开。

该数值修复只解决可重复的 FP16 溢出风险，不构成新校正方法的证据。Sierre 单城开发验证不能证明跨地区泛化。
