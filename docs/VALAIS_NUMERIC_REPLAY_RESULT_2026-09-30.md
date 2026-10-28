# Valais clean 第 34 轮：隔离工程回放结果

日期：2026-09-30。结论：**受守卫的一次性隔离回放在原故障窗口内精确重现非有限 loss，并将其定位为该回放状态下 AMP 前向产生 NaN logits；同一模型和批次的 FP32 前向及损失均有限。** 这是数值故障诊断，不是完成的 clean 基线或可发表性能结果；原 [Valais run 状态](../runs/valais_clean_development_seed42/status.json)仍为 failed，完整 checkpoint 停在 epoch 33，Martigny test 未访问。

## 证据链

- [协议](VALAIS_NUMERIC_REPLAY_PROTOCOL_2026-09-30.md)、[冻结配置](../configs/valais_numeric_replay_epoch34_20260930.yaml)与[预测试 seal](../runs/valais_numeric_replay_epoch34_20260930.preflight_seal.json)绑定 epoch 33 checkpoint、原 13 项故障审计文件、数据/源码、诊断/守卫/launcher/单测及 GPU UUID。seal SHA256：a0362bd13eff3c87fb2f13bd5da95c7f4676abb4594e065ea0490538011c4be3。
- 执行前 **228 项全量 unittest 通过**，独立单批真实 CUDA smoke 通过。进程绑定温度守卫正常结束，最后温度 57°C、decision=ok；无温度或传感器停机。完成记录在[completion.json](../runs/valais_numeric_replay_epoch34_20260930/completion.json)，SHA256：4fdcca3ac1b3594d3c0d946c6becb204340a6bccb9277ff1f078b568cc8598b8。
- 训练回放的 12 个冻结累计 loss 点全部与原日志六位值一致；唯一 AMP 跳步均在 step 1836、scale 131072→65536；成功优化 2756 步、最终平均训练 loss 0.098780。验证 step 1/5/10/15/20/25/30/35 的八个累计 loss 点也全部一致。轨迹匹配只在日志记录的精度内成立，不能证明所有未记录的中间张量逐位相同。
- [replay.json](../runs/valais_numeric_replay_epoch34_20260930/replay.json) SHA256：be41f37c234f9bafbde5162cac926205178ff830c0662f231bdc5fd3c1ad629d。首个非有限 loss 出现在 **val step 36/54**，比原日志的 36–40 窗口更精确；该 batch 有 32 个样本，sample ID 已在 JSON 中逐一记录。AMP logits 为 FP16，2,031,616 个元素有限、**65,536 个 NaN**、无 ±Inf；BCE、Dice 和 total 都是 NaN。同一模型、同一批次、无梯度 FP32 logits 的 2,097,152 个元素全部有限，total loss 0.06661018（BCE 0.00414433；Dice 0.06246585）。

65,536 恰好等于一个 256×256 tile 的像素数，但目前只保存了全 batch 计数，**不能断言 NaN 全在单个 tile，更不能指认具体影像**。也没有记录首次出错的网络层或 FP16 中间激活。此证据支持“在复现的 epoch34 状态与 val step36 输入下，AMP 前向产生非有限输出，而 FP32 前向没有”；它不证明对任意初始化、epoch 或成像条件均如此，也不将类别稀疏自动归因于数值故障。

## 处置与下一协议

1. 原 run、原 checkpoint、history 和失败审计保持不动；不从隔离回放模型继续原 100 轮，也不补写原 guard_completion。旧 LEVIR held-out test 仍是已使用数据，Valais Martigny test 继续封闭。
2. 下一个正式 clean 基线应**从随机初始化重新开始**，训练 AMP 可保持，但验证改为 FP32；这是与原基线不同的、必须单列的新数值协议。先用新的独立配置、输出、源码快照、完整数据审计与温度守卫做单批及短程工程检查，再开展完整预算。B1/B2 后续也须使用同一验证数值协议，不能将精度变更误写成校正方法收益。
3. 若需要定位首次非有限层或具体样本，另冻结一次只读层级 hook 诊断协议；不得事后反复回放这份一次性结果或根据 Martigny test 调整。

结果仅说明本次工程回放的故障机制。它不恢复原研究运行的有效性，也不支持直接启动 PSE/PCDC/CPC 或局部动作。
