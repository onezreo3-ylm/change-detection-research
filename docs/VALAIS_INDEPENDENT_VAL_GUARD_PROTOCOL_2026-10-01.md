# Valais FP32-val clean 基线的受守卫独立验证

日期：2026-10-01。状态：**仅实现与预审，尚未运行独立验证**。当前 Valais clean 训练仍在进行；本协议仅在完整100轮、训练温度守卫正常收尾且 durable guard_completion.json 存在后启用。Martigny test 继续封闭，不用独立验证选择新超参数或阈值。

固定 checkpoint 是已完成训练的 best.pt，仅由 Sierre clean-val pooled F1@0.5 在训练中选出。独立复核同一 Sierre val 的 1715 对、相同严格0/1标签、相同RGB归一化、模型/阈值与 FP32 验证路径；只重算损失和 pooled 混淆计数，要求 F1 与 checkpoint best_val_f1 精确一致。它是工程复现，不是独立地理泛化证据。

独立[复核器](../scripts/evaluate_valais_fp32val.py)首先核训练 status=completed/test=false、[持久化训练完成标记](../runs/valais_clean_fp32val_seed42_relaunch_20261001/guard_completion.json)、最终温控会话的状态、preflight seal、best/last 权重哈希、seal 所有输入和完整源码快照。没有完成标记或任一哈希不符即拒绝。输出固定为 run/val_independent.json；已有文件不得覆盖。

[入口 wrapper](../scripts/valais_independent_val_entry.py)在读取 PID 绑定守卫握手前不导入/执行模型评估。[launcher](../scripts/launch_valais_independent_val_guarded.py)先固定本次来源/配置/best/checkpoint/完成标记/新脚本哈希，验证 GPU UUID 与低于85°C 的温度，再启动 wrapper、绑定[专属温度守卫](../scripts/guard_valais_independent_val.py)，校验 PID/创建时间/命令行/首温与 guard state 文件持久化后放行。guard连续85°C、单次86°C或传感器连续错误即停。进程、FP32数值、F1复现、守卫状态与所有哈希一致才写 validation_guard/completion.json。失败保留现场、不自动重试，也不改训练 run 的已完成结论。

脚本开发期间只允许 CPU/合成定向测试，不并行占用当前受守卫训练 GPU。训练完成且 GPU 空闲后再运行全量 unittest、真实 CUDA smoke，并在正式复核前冻结新 launcher/guard/wrapper 的代码哈希。
