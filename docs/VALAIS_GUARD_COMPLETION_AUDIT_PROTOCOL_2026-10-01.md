# Valais 训练完成但 launcher 失联时的独立守卫完成审计

日期：2026-10-01。状态：**仅预注册只读审计；当前运行未完成，因此审计器明确拒绝签发证书**。此机制应对[epoch38 监护链中断](../runs/valais_fp32val_supervisor_loss_20261001/audit.json)的可重复风险；不代替 PID 温度守卫，不允许对活跃训练补造完成标记。原 scripts/launch_valais_fp32val.py 若正常完成，仍以它的 durable guard_completion.json 为主，本审计器在原标记存在时拒绝。

[外层审计 YAML](../configs/valais_guard_completion_audit_20261001.yaml)绑定原研究配置 SHA256 febae7befad0d4ea52281eaf9502b7ea0cafbb7c1f74e72f431b8f0f6626328e、epoch38 监护链审计 SHA256 d7f3296db88e935d84c6c1a11f9892cc7cb71a7ea807d45bcd3ec9f3e64a0eea、100轮预算、test forbidden及独立证书目录。[只读审计器](../scripts/audit_valais_guard_completion.py)使用源配置和冻结 snapshot，不加载 Martigny test、不做模型推理。

仅当 trainer status=completed/100轮/train AMP/val FP32/test=false、last.pt format2/epoch100/history100、best.pt 与 history 首次最佳 F1/epoch 一致且模型张量有限时才继续；原 guard_completion 必须不存在，train/launcher/guard 进程必须全部退出。然后重新全量审计 train/val 文件与 checkpoint 数据摘要、校验原 snapshot 的所有源码哈希、档案配置与最后一次 resume session 的 preflight seal 全部绑定、PID/创建时间/温度守卫 attach marker、最终守卫 state=training_process_exited/无残留 lock 或 pause 记录、epoch100 train/val 日志无 traceback。状态和守卫时间顺序必须合理。

全部通过时只在**新目录**写 certificate.json，措辞固定为 audit_recovered_guarded_completion_only，记录绑定产物 SHA、最佳开发验证 F1、最终守卫会话与 test=false。它**不写入**原运行的 guard_completion.json，不声称原 launcher 正常完成，也不把这份只读认证直接当作新的 held-out 测试。若任一校验失败，明确拒绝签证且不自动重跑/覆盖。该证书若产生，独立 clean-val 复核还须另行采用受守卫、明确接受 audit-recovered 完成状态的新协议；现有要求原 durable marker 的验证器不得绕门运行。

当前 epoch38/manual interrupted 状态下，预检按设计在 trainer-completed 门报错。未来要执行本审计，还须在 GPU 空闲时完成全量 unittest、合成/真实 CUDA 工程 smoke，并在执行前封存审计脚本/配置/协议 SHA；其结果只在培训运行真正结束后有意义。
