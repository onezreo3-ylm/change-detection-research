# B0-E / B1 / B2 匹配训练协议

日期：2026-09-13。用户在视觉复核通过后明确要求继续完成剩余实验；本阶段仍保持
开发期 test 关闭，先执行十步方案第四步。三个运行均从 seed=42 随机初始化，继承
stable-b4 的 100 epochs、batch=4、AdamW、cosine、AMP、阈值与 clean-val 选优。

每个原始 source batch 产生 clean 与 secondary 两个 pair view，按
`[clean, secondary]` 合并后只调用模型一次、只做一次 optimizer step。模型内部再把
A/B 沿 batch 维一次性合并，因此每个 step 只有一次共享 encoder BatchNorm 更新。
B0-E、B1、B2 的样本顺序、两视图数量、前向顺序与 BN 方式完全一致。

- B0-E：secondary 是 clean 的值相同副本；用于拆分额外曝光/BN batch 形状影响。
- B1：secondary 使用 v1 train stream；每个 sample/epoch/view=0 由固定哈希均匀抽取
  family、L1–L5、direction、A/B 与 global/local；clean 和 perturbed 都接受变化监督。
- B2：与 B1 使用完全相同的 secondary；额外加入
  `mean((sigmoid(z_clean)-sigmoid(z_perturbed))^2)`，权重固定为 1.0，双侧均传梯度，
  无 EMA、teacher、伪标签或真实扰动类别输入。

实际 optimizer steps、AMP 跳步、双视图前向数、每档曝光计数和逐 epoch 暴露流 SHA
均写入运行产物。三者只按 clean val F1 选 checkpoint；此阶段不读取 held-out test。
配置见 `configs/robust_b0e_seed42.yaml`、`configs/robust_b1_seed42.yaml` 与
`configs/robust_b2_seed42.yaml`。
