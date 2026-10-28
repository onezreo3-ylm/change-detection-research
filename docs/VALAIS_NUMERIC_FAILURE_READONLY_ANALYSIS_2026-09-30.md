# Valais clean 基线数值失败：独立只读分析与恢复准入

日期：2026-09-30。状态：**原运行失败、原因未定位、不得自动续训**。本文件只读现有训练日志、状态、history、源代码及 epoch 33 checkpoint 元数据；没有执行新的模型前向、修改原运行文件、访问 Martigny test，或将部分结果提升为完成的研究结果。[原运行故障审计](../runs/valais_clean_failure_20260930/numerical_failure_audit.json)由另一协同任务完成并绑定 13 项文件哈希；本文件只评估后续独立实验如何取得证据。

## 已确认事实

- [原运行状态](../runs/valais_clean_development_seed42/status.json)为 failed，异常文字为 FloatingPointError: Non-finite loss; experiment stopped，test_evaluated=false。训练、温度守卫及 launcher 均已退出。守卫最后记录 65°C 且 decision=ok；该故障不是温度门触发。
- [续跑日志](../runs/valais_clean_development_seed42.resume_20260930_epoch6.train.log)显示 epoch 34 train 完整 2757/2757 batch，平均 total loss 0.098780；在 step 1836 有一次由 GradScaler 确认的 AMP overflow skip，scale 从 131072 降至 65536。验证已打印 step 35/54 的有限累计 loss 0.221639，在下一次计划打印 step 40/54 前由 [run_epoch 的 loss 有限性检查](../pcdc/engine.py)报错。因此失败只可定位到 **val step 36–40 的某一批**，不能断言确切 batch 或样本。
- 完整 [last.pt](../runs/valais_clean_development_seed42/last.pt)停在 epoch 33，SHA256 75bf25c9f50925ff889d029b5798efb212320a77b18e15ff6e10d27e556b64a4；checkpoint 中 GradScaler scale=65536、scheduler last_epoch=33。best.pt 也由 epoch 33 选出，clean-val F1 0.3772850022。这是未完成运行的**中间验证值**，不代表 100 轮 clean 基线最终成绩。第 34 轮训练后的权重没有存盘。
- [逐轮 history](../runs/valais_clean_development_seed42/history.json)在 epoch 1–33 均写入有限损失；epoch 33 val 完整通过。val 输入是固定的 1715 个 256×256 配对；先前[全量数据审计](../runs/valais_candidate_preflight_20260929/pcdc_development_full_audit.json)证明 0/1 标签、配对尺寸与原文件哈希。它排除明显非法标签和数据被静默换掉，**不能**证明任意模型状态的 AMP 前向一定有限。

## 当前解释边界

[ChangeDetectionLoss](../pcdc/losses.py)在模型前向**之后**将 logits 转成 float32，再计算 BCE 与 Dice。因此若 AMP 的 FP16 前向已经产生 Inf/NaN，后续转 float32 无法恢复；但本次没有保存失败批次的 logits、BCE/Dice 分量或 epoch 34 权重，不能认定 FP16 溢出就是根因。另一可能是某个新模型状态只在特定验证输入上触发异常。epoch 33 checkpoint 的有限张量与 epoch 34 训练 loss 有限均不足以证明失败时所有中间激活有限。

类别极稀疏是另一个已确认但**不等同于数值故障原因**的事实：train/val 正类像素分别为 0.1548%/0.2718%，自然 batch=4 时约 67.1% 的 train 批全负，见[独立标签报告](../runs/valais_class_balance_20260930/report.json)。它影响学习与评价解释，不足以单独解释第 34 轮验证的非有限 loss。

## 最小诊断顺序（新输出，禁止覆盖原运行）

1. **封存失败状态。** 由原运行负责人固定 status、续跑日志、温度守卫、source/config/data 审计、history、last/best checkpoint 的哈希及进程已退出事实。任何诊断结果都不能将原 status 改写为 completed。
2. **先只读定位批次范围。** 在原 val 顺序和 eval_batch_size=32 下，step 36–40 对应 0-based tile index 1120–1279。[元数据清单](../runs/valais_numeric_failure_analysis_20260930/step36_40_inventory.json)已绑定这些样本 ID、原审计输入哈希和标签正像素数；SHA256 为 64898d4393f739bc3b01a8379cdddb6026deb006fef965d9135cce4c42c44948e。各批含变化 tile 数依次为 0、3、2、1、1。没有逐批失败记录时，不能根据这些统计反推单个“罪魁”样本。
3. **独立工程诊断。** 新脚本、新配置、新输出目录，从封存 epoch 33 checkpoint 创建隔离副本，先验证该权重在相同 val 批次的 FP32/AMP 前向 logits、BCE/Dice 各分量是否有限。此检查**不等同**于原 epoch 34 重现，因为权重不同。若要精确重现，必须另冻结一次 epoch 34 单轮训练回放：恢复所有 RNG/优化器/GradScaler 状态，保证相同 source/data 哈希、样本序列及成功/跳步数，输出到独立目录，记录首个非有限张量和 batch ID。若轨迹不一致，标记复现失败而非补全原研究 run。
4. **恢复或重训决策另列协议。** 若确认为 AMP 验证前向问题，可设计从随机初始化开始、训练 AMP 保留但验证 FP32 的新数值协议，或单列 FP32 全程对照；参数、计算成本与 checkpoint 选择均需新配置和完整预算。直接在原 run 修改 eval 精度并续训会混用两种验证数值协议，不能冒充原 100 轮无变化完成。epoch 33 出发的隔离探索只能称工程恢复/诊断，不能自动成为正式 B0/B1 研究结果。

在得到稳定、完整的 clean 检测器之前，不启动 Valais B1 扰动、局部动作或 PSE/PCDC/CPC。Martigny test 保持关闭，旧 LEVIR held-out test 的花费状态不变。新增源代码后须通过全量 unittest 与真实 CUDA smoke；数值诊断的训练/验证推理需使用同等温度守卫和明确的停止规则。

## 本次分析的工程复核

故障后 GPU 空闲、实测 48°C 时，运行 python -m unittest discover -s tests -v：**223 项通过**；另在独立[单批真实 CUDA smoke](../runs/valais_numeric_diagnostic_smoke_20260930/real_cuda_smoke.json)中完成一次随机初始化模型的有限训练/验证前向和一步优化，test_accessed=false。该 smoke 没有加载失败 checkpoint，不能替代失效批次的数值定位，也不会改变原研究运行的 failed 状态。[原 LEVIR 收口校验](../scripts/verify_research_closeout.py)仍返回 verified。
