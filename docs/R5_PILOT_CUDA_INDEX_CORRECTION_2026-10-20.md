# R5工程pilot的启动查询修正

2026-10-20。原有限筛查协议和1137项封存输入保持原样，不改旧脚本或失败产物。
第一次pilot在torch.cuda.mem_get_info(torch.device('cuda'))处失败：本机PyTorch 2.0.1
要求显式device index。失败记录证明pair_views=0、optimizer_steps=0，且原seed42初始
张量哈希正确；异常发生在forward之前，没有BN更新或训练计算。

修正只在新的独立脚本中将显存查询传入明确CUDA设备整数索引。其余4 source、5 views、
一次联合BN、FP32、初始化、loss、AdamW及唯一1次工程更新全部沿用已封存配置。
不改旧授权哈希，不覆盖旧report，不重复BN诊断。新脚本、配置、测试和本说明另作封存。
仅在旧失败恰为上述启动错误且实际预算为0时允许执行一次；不是数值/温度/OOM自动重试。
两次启动累计pilot上限仍为600秒，扣除原记录的0.969秒；只允许总计1个实际优化步。
当前用户对该有限工程任务的授权覆盖这项实现修复，不引入正式方法训练或研究评分。
若这次发生OOM、温度或数值失败则保留现场，不再修正配置重试。启动初始化/读取仍为
额外物理开销，必须披露，不能隐藏为一次无开销启动。

新命令入口scripts.valais_r5_engineering_pilot；旧scripts.valais_feasibility_screen保持
封存原样。最终新独立审计同时核对BN输出和修正后的pilot，另保留原启动失败收据。
