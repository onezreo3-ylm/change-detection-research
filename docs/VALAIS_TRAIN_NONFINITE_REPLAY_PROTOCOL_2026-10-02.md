# Valais 第 46 轮训练数值故障：一次性隔离回放

日期：2026-10-02。仅为工程诊断，不补完或改写失败的研究运行，不产生论文分数。

原受守卫运行在完整 epoch45 后，于 epoch46/train 的 step1100 与 step1375 日志点之间出现非有限 loss。训练 AMP、验证 FP32，`last.pt` 的模型张量有限，守卫正常结束且末次温度 75°C。[故障审计](../runs/valais_fp32val_train_nonfinite_20261001/audit.json)绑定原 status、checkpoint、history、日志和守卫。Martigny test 与旧 LEVIR test 均不参与本诊断。

## 固定输入与停止门

- [YAML](../configs/valais_train_nonfinite_replay_epoch46_20261002.yaml)只接受原 epoch45 `last.pt`，SHA256 `e9f0bdc9955af247d6138c5efbffe28b1dbf5f0dcaff010b83540f32293d15e4`；研究配置 SHA256 `febae7befad0d4ea52281eaf9502b7ea0cafbb7c1f74e72f431b8f0f6626328e`；故障审计 SHA256 `58642ffca1155b5d389b03e20c7a4b0ae1426d6d84a42f17384aaabe754bd76a`。输出是独立的 `runs/valais_train_nonfinite_replay_epoch46_20261002`；已有输出时拒绝启动。
- [诊断器](../scripts/diagnose_valais_epoch46.py)只读核对源码快照、研究状态、checkpoint 与 train/val 全量数据审计，从完整 epoch45 恢复模型、优化器、调度器、GradScaler 和所有 RNG。只运行 epoch46 训练至首个非有限 loss，最迟 step1375 停止。验证和测试均不推理。
- 原日志冻结的累计 loss 点为 step 1/275/550/825/1100 的 `0.000045/0.091539/0.090347/0.087594/0.091178`，AMP 梯度跳步为 step480、scale 131072→65536。任一前缀不符即停止；只有在这些点匹配且首个故障位于 step1101–1375 时，才称原窗口重现。
- 在 step1101 后，只读前向钩子记录最早可见的非有限模块输出。故障批记录 sample ID、输入/标签有限性、AMP logits 与损失分量，并对同一模型和批次做一次无梯度 FP32 前向比较。模块钩子只能定位第一个**被观察到**的叶模块输出；功能算子内部或上游原因仍须谨慎解释。
- 诊断进程由 PID、创建时间、命令行和 GPU UUID 绑定的温度守卫监测；连续 85°C、单次 86°C 或连续传感器错误停机。执行前完成全量单测和真实 CUDA 单批 smoke。launcher 保存输入/源码 seal，结束时验证守卫与哈希才写 `completion.json`。任何异常保留现场，不自动重试原运行。

根据结果另立数值修复协议。若 AMP 前向非有限而 FP32 有限，优先评估训练前向 FP32 的独立协议；不能把更改精度后的权重或分数与原 100 轮基线混为同一运行。正式训练还须保持原数据划分、标签、模型、阈值、初始化和完整预算，并先通过其独立 smoke 与温度守卫门。
