# ValaisCD 独立 clean 基线：训练 AMP、验证 FP32

日期：2026-09-30。状态：**待工程预检和新运行冻结**。这是与失败的原始 Valais clean run 分开的新数值协议，不会续接 epoch33 权重、修改原 run/status/checkpoint，或补写原 100 轮完成声明。原失败与[一次性隔离回放结果](VALAIS_NUMERIC_REPLAY_RESULT_2026-09-30.md)保留为故障证据。

## 变更动机与唯一研究变量

受守卫的隔离回放在第 34 轮精确匹配原训练和验证轨迹，val step36 的 AMP logits 出现 65,536 个 NaN，而相同模型/批次的 FP32 logits 与 BCE/Dice 均有限。因此下一基线只把**验证前向精度**改为 FP32，以避开已观察到的验证数值故障。训练 AMP、模型、随机初始化、源数据、像素标签、归一化、损失、阈值、优化器、学习率、batch、100 轮预算与 clean-val F1 选模规则均保持原配置。两套验证精度的 checkpoint 选择轨迹不同，其分数不得混成同一个 run；这项数值修复不是校正方法创新。

## 固定输入与隔离

- Sion train 11025 对、Sierre val 1715 对，均为已经通过字节与解码审计的 ValaisCD 256×256 RGB/0-1 标签。Brig dev 只作先前 QA，Martigny test 不下载、不列举内部文件、不推理。
- 源[基础配置](../configs/valais_clean_development_seed42.yaml) SHA256 7a4931665591a24e005b7ea3b42b1cb88855accc52343c8642ad4efc5ceea6ab；源数据审计 SHA256 670d7d29361125e291d637f9fb9be44df31b67912dc2af837d719f6c4ae2cd87。实际训练再次全量审计 train/val，并绑定每个输入文件哈希。
- 新[外层配置](../configs/valais_clean_fp32val_seed42_20260930.yaml)明确 train AMP=true、validation AMP=false、100 epochs、独立输出目录。新 runner 位于 scripts/，仅调用冻结的 pcdc 模型、损失、数据与单 epoch 算法；不修改 pcdc/*.py。新 run 保存外层配置、运行源码及哈希、环境、数据审计、逐轮 history 和 best/last checkpoint；严格断点续训只允许相同字节的配置、源码、数据、初始化和数值策略。
- NVIDIA RTX 3060 Laptop GPU UUID GPU-8ada1895-7628-bb88-1dda-c57679117fa2；受 PID/创建时间/命令行绑定温度守卫监测。连续确认 85°C 或单次 86°C，以及连续传感器错误均停训；从完整 epoch checkpoint 人工复核后才能续跑。不得改用 CPU、其他 GPU 或随机权重替代。

## 预检、训练与验收

1. 在不打开 test 的前提下完成独立配置/源码/输入哈希预检、全量 unittest 和真实 CUDA 单批 smoke；smoke 验证 train AMP 与 val FP32 的有限前向/反向、一次优化更新、输出形状及开始/结束的 GPU UUID/温度，明确不作为研究分数。完整 100 轮运行另由 PID 绑定温度守卫附着后才允许训练。
2. 从 seed42 随机初始化进行 100 轮，epoch 边界保存完整 RNG、模型、优化器、调度器和 GradScaler。训练 batch=4、eval batch=32；训练和验证分别明确记录 AMP=true/false。验证只选 clean F1 最佳 checkpoint，先汇总整城混淆计数再算指标，固定阈值0.5。每轮记录 TP/FP/FN/TN、Recall、FPR、AMP 跳步及耗时。
3. 若再次出现非有限数值、数据/哈希变化、守卫故障或温度暂停，立即保留现场并停止；不调阈值、权重、采样、强度或模型以掩盖失败。仅 100 轮、持久化守卫完成标记和[独立 FP32 clean-val 复核器](../scripts/evaluate_valais_fp32val.py)全部完成后才报告完整研究结果。独立复核器只接受固定 val 输出路径，逐项核验完成标记、温度守卫状态、preflight seal 和全部源码快照，并在其专属 PID 绑定守卫握手前拒绝推理；专用 launcher/守卫必须在训练完成前另行冻结，届时再运行该复核器。
4. 若新的 clean 基线仍不能稳定检出正类，再依[普通类别平衡对照门](VALAIS_CLASS_BALANCE_AND_NEXT_BASELINE_GATE_2026-09-30.md)另立采样/加权协议。若可用，下一步先在训练影像完成 Valais 扰动强度 QA，再训练匹配的普通 B1；B2、局部动作与 PSE/PCDC/CPC 不自动启动。

## 推断边界

Sierre 是单个验证城市，tile 间有空间相关性，且无可靠原幅面/坐标映射；不能用 tile 级 bootstrap 冒充独立不确定性。新精度协议由已使用的 Sierre 开发证据启发，Martigny test 仍须等全部方法和阈值冻结后一次性描述。旧 LEVIR-CD held-out test 已使用，不再参与任何新方法选择。
