# Valais 全 FP32 clean 基线的独立受守卫验证

日期：2026-10-02。仅在独立[完成审计证书](../runs/valais_full_fp32_completion_audit_20261002/certificate.json)签发后执行一次，不修改研究运行或打开 Martigny test。原外层 `guard_completion.json` 缺失，不能借用旧验证器的完成门。

选定 checkpoint 固定为 `runs/valais_clean_full_fp32_seed42_20261002/best.pt`，由训练中的 Sierre clean-val pooled F1@0.5 首次最佳选择。验证只重算相同 Sierre val 的损失、混淆计数和 F1，要求 F1 与 checkpoint/完成审计精确一致。这是工程复现，不是独立地区泛化试验。

[验证入口](../scripts/validate_valais_full_fp32.py)先核审计证书、其中全部输入/源码哈希、checkpoint 协议、GPU UUID/温度、全量单测与真实 CUDA smoke。单独输出目录 `runs/valais_full_fp32_independent_val_20261002` 必须不存在。launcher 启动等待握手的 worker，使用 PID/创建时间/命令行绑定的[温度守卫](../scripts/guard_valais_independent_val.py)确认首温，才允许导入评估器并推理。守卫异常即停，输出不覆盖；子进程/守卫正常退出、F1 精确复现、证书及 seal 哈希未变后才写独立 `completion.json`。

本验证不选择新阈值、不修改已用 LEVIR test、不接触 Martigny test。成功后按既有 Valais 正类检出门判断是否进入 train-only 扰动 QA。
