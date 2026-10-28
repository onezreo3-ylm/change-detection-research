# 40轮C5/R5缩时预案：显式独立数值协议

2026-10-20。用户要求“尝试优化一下方案缩短一下时间”。原40轮FP32协议、配置、
397项测试和smoke/authorization均保留，正式训练尚未启动；不改写这些封存文件。
本预案只有完整工程门通过且用户明确选择数值精度后才封存启动，不把性能探索当作训练。

## 实测与保持项

第一轮纯工程优化为CPU worker预排布/固定内存、合并标量同步和设备int64指标。
两行参考/候选各16步，loss、模型、优化器、RNG、指标与输入流逐位相同，但C5/R5
仅提速1.0573%/1.4299%，未达到预先固定的每行3%采用门，不用于长训练。
数据等待仅约0.02–0.03%每步，主要耗时是GPU计算，不增加worker或追求数据缓存。

所有科学设置继续继承C5R5_40EP_PROTOCOL_2026-10-20.md：各40轮、T_max40，seed42
原初始化、原模型、batch4、五view与权重、联合BN、完整source/实例/候选像素流、
110280成功更新、441000 source draws、2205000 pair views、原损失/优化器/clip/
mean/std/阈值、首次最高clean-val选模、温度规则、45条件val及固定64 train残余均不改。
Martigny及旧LEVIR test关闭，不准入新模型、几何、外部数据或M5/K5/W5。

## BF16候选的明确变化

本机实际CUDA支持BF16。独立预检只把训练forward设为CUDA autocast dtype=bfloat16，
参数、BN状态和AdamW状态保持FP32；损失在显式禁用autocast的区间累加FP32。
不用GradScaler，无自动跳步；非有限logits/loss/梯度/参数立即失败并保留现场，不降标准。
clean val和后续全部固定评价仍显式FP32。不是旧FP16的自动替代或原FP32轨迹续训。
此前FP16在第46轮可重复出现Inf/NaN的现场保留；不因短BF16 pilot通过而承诺长期稳定。

BF16有限pilot每行最多600秒：32步（前8步warmup）、真实下一更新保存/重载两次
物理计算精确复现loss/模型/优化器/RNG、一源五view尾batch，以及初始FP32/BF16
重复/交换自检。各35次物理更新全部丢弃。它不与FP32逐位等价、不证明研究效果。
本批两行同数值协议，主比较仍R5/C5；历史FP32、100轮B0-E只是不匹配背景。

## 人类选择与预算

原40轮协议冻结了FP32；BF16属于数值协议变化，因此长训练封存要求直接人类选择，
写入human_precision_approval.json后才允许生成新authorization。预案、测试和工程
预检先完成；不自动把“优化一下”的探索授权写作已接受BF16长训练。
若人类选择保留FP32，则回到相同科学协议，只更新实际已用工程时间账，不采用未达门的CPU优化。

所有先前C5/R5工程耗时继续从该行86400秒扣除：原32步pilot、本批40轮smoke、
16步两方案性能对照、BF16 pilot，不能因新输出目录归零。总计210次额外物理更新，
4158 pair forward views（原有限筛查/P1暂停重算另列）；正式训练仍从原初始化开始。
预算预测仍为新实测稳态均值×110280×1.25+40×60秒+累计工程扣款，逐行必须<=24小时。
实际worker从输入验证开始计时，预算到期停止，不续期、不重试、不缩短轮次或追分。

新flavor为matched_five_view_baseline_40ep_runtime_v1；identity必须同时显式记录
optimization_mode、training_autocast_dtype、parameter_optimizer_loss_dtype、validation_dtype、
grad_scaler、源码和数值预检hash，不能假借旧flavor通过旧评价器。
只启动一个隐藏队列C5→R5；完成预算、guard/hash、实际流和选模审计后才接显式新适配评价。
两行val各6小时、残余独立6小时，保持原门。闭环后保存报告，无已准入下一实验则暂停托管。
