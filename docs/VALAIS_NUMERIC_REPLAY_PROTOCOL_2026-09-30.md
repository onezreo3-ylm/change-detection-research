# Valais epoch 34 数值故障的一次性隔离工程回放

日期：2026-09-30。状态：**工程诊断协议；不是原研究运行的恢复或完成**。目的仅是判断从完整 epoch 33 checkpoint 恢复一次相同训练轨迹后，epoch 34 validation 的非有限 loss 能否在原窗口重现，并记录 AMP/FP32 的首个可见数值差异。原 run 的 status=failed 不改写，原 checkpoint/配置/源码不修改，Martigny test 和旧 LEVIR held-out test 均不访问。

## 输入和输出冻结

- 唯一输入 checkpoint：原 Valais clean run 的 epoch 33 last.pt，SHA256 75bf25c9f50925ff889d029b5798efb212320a77b18e15ff6e10d27e556b64a4。原配置 SHA256 7a4931665591a24e005b7ea3b42b1cb88855accc52343c8642ad4efc5ceea6ab；原数据审计 SHA256 670d7d29361125e291d637f9fb9be44df31b67912dc2af837d719f6c4ae2cd87；[故障审计](../runs/valais_clean_failure_20260930/numerical_failure_audit.json)绑定状态、日志与这些产物。
- [独立 YAML](../configs/valais_numeric_replay_epoch34_20260930.yaml)规定 epoch、训练 trace、AMP overflow 位置、validation trace、失败窗口和 GPU/温度门。[诊断器](../scripts/diagnose_valais_epoch34.py)、[launcher](../scripts/launch_valais_numeric_replay.py)、[工程守卫](../scripts/guard_valais_numeric_replay.py)使用独立输出 runs/valais_numeric_replay_epoch34_20260930；禁止该目录指向旧运行或数据根。
- 启动器先计算配置、脚本、相关单测、全部 pcdc 源码、原 checkpoint/审计字节哈希，写不可覆盖的 preflight seal。诊断进程启动时重验 seal，结束时 launcher 再验；另外前后逐项核对原故障审计绑定的 status、last/best、history、data audit、守卫、四类日志、配置、engine/loss 源码共 13 项，才可写 original_run_modified=false。启动器核对 NVIDIA GPU UUID GPU-8ada1895-7628-bb88-1dda-c57679117fa2；守卫通过准确 PID、进程创建时间及命令行绑定进程并监测 GPU index 0。守卫成功绑定并完成首次传感器读取后写一次性 attach marker，launcher 只读该 marker 建立握手，**不读取正在更新的 guard state**。85°C 连续确认或 86°C 单次触发即停止，传感器持续失败也停止。守卫写入独立状态，与诊断结果分别存证。

## 一次性执行与停止门

1. **只读预检：**检查原运行仍 failed、checkpoint 为完整 epoch33、源码/训练签名/数据审计与 checkpoint 一致、test 禁止且独立输出不存在。新进程仅构建 train/val，重算完整数据审计；不读取 test。
2. **精确单轮训练回放：**恢复 epoch33 模型、优化器、调度器、GradScaler 和 Python/NumPy/Torch/CUDA/DataLoader RNG，只运行 epoch34 训练。必须逐点匹配原日志在 step 1/275/550/825/1100/1375/1650/1925/2200/2475/2750/2757 的六位累计 loss；唯一 AMP 跳步须在 step1836、scale 131072→65536；成功更新 2756、最终平均 loss 0.098780。任何一点不符即标为轨迹不一致并停止，不能进入 validation。
3. **验证轨迹门：**仅在训练全部吻合后，按原 eval_batch_size=32、原 val 顺序与 AMP 运行 validation。step 1/5/10/15/20/25/30/35 的累计平均 loss 均须与冻结日志六位数一致；不符立即停止。原故障只有 step36–40 的窗口范围，没有精确 batch ID。
4. **一次失效诊断：**如果在窗口内第一次出现非有限 loss，保存该批 sample ID、AMP logits 的有限/NaN/±Inf 数量和 BCE/Dice/total 分量；在**同一冻结模型和同一批**做一次无梯度 FP32 验证前向，记录相同字段，然后停止。若失效出现在窗口外、到 step40 仍未重现或任何前置门不符，明确报告“未重现/轨迹不一致”，不扩大到整套验证或反复试错。

即便 AMP 非有限而 FP32 有限，也只支持“该复现状态下数值精度有关”的定位，不自动证明原故障的一般根因。若 FP32 同样非有限，要继续定位首次非有限层、模型状态和输入，但应另列新诊断协议。epoch34 回放的模型状态和分数不得写入原 best/last、history 或论文结果；也不能从其检查点继续正式 100 轮。后续稳健 clean/B1 训练须单独冻结数值策略和完整训练预算。

## 验收记录

执行前要求全量 unittest、独立单批真实 CUDA smoke、GPU 温度与进程空闲检查通过；测试日志、脚本哈希和 smoke 状态写成独立预检证据并由 seal 绑定。正式 epoch34 回放由进程绑定温度守卫全程监测。执行后只有当进程正常退出、守卫状态正常、前后哈希一致且 replay.json 完整时才写独立 completion.json；其 reproduced_within_bound_window 字段仅表示工程失效轨迹重现。任何异常保留失败产物，不自动重试或覆盖。
