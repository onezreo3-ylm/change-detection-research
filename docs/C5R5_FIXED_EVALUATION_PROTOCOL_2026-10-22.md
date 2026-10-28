# C5/R5 40轮 BF16训练后的固定 FP32 评价

2026-10-22。接续用户已选择并执行的独立40轮C5/R5协议。
本文件只具体化原先批准的45条件val和固定64 train残余，不改变任何门、不追加训练。
原100轮、40轮FP32和40轮BF16训练源文件/配置/checkpoint/guard全部保持。

两行训练均已完成40轮，各110280成功更新、441000源抽取、2205000 pair views。
队列完成证明使用supervision_attach_20261021的实际等待/退出码0及完整提交hash；
原root owner/status保留启动时0xC000013A的历史，不能当作当前完成证明。
重新生成40轮计划与封存计划/两行实际流逐项一致，完整实际候选像素流一致。
best必须是首次最高clean-val F1的checkpoint；best和last均验证实际flavor、全部训练源码、
原初始化、identity、train/val审计、模型/优化器FP32有限值、无跳步及全部累计预算。

新显式适配器scripts/c5r5_fixed_evaluation.py仅接受
matched_five_view_baseline_40ep_runtime_v1以及bf16_training_fp32_validation。
原checkpoint.train.amp=True代表其已获选择的训练forward BF16，并不代表评价启用AMP。
用实际训练配置校验原training_signature；再显式验证与原FP32 benchmark仅相差训练
precision、40/100轮训练horizon和实验名称/输出路径，规范相同数据/权重路径表示。
所有预测相关模型、预处理、数据、损失、设备、seed、阈值等逐项相同；
checkpoint对象、flavor和旧评价器检查均不修改。新评价独立命名，不冒用原100轮结果。

顺序固定：C5原45条件val → R5原45条件val → 原P0固定64个Sion train样本残余。
每个val阶段及整个残余各21600秒硬上限，单一GPU worker，无自动重试/覆盖/续期。
原45条件manifest/实例/视觉接受/codec/强度保持，先复现best的clean全部计数和五指标，
再原实例逐条件评分。模型参数和BN状态冻结，FP32 CUDA、channels_last、阈值0.5；
guard保持val 85°C确认/86°C立即，残余>=85°C立即停止。
原val保存计数/分组/边际守恒及guard/hash审计复用独立纯读取审计函数。

残余新入口scripts/c5r5_fixed_residual.py静态复用原P1诊断与独立四格/交换/守恒数学，
显式改为C5/R5实际身份；不修改原P0 manifest/visual_review。
四族brightness/contrast/gamma/color_bias，等级1/3/5，64样本16正48空；
6272主预测、6276总预测、1536记录必须完整，额外4预测用于重复/交换自检。
封存前在原首个train样本做两模型FP32真实CUDA smoke，每模型clean/重复/交换和12条件
四格/边际，共102预测，无训练、无val/test研究评分，600秒工程上限。
完整>=421项测试含实际CUDA测试须通过；smoke后再封存全部输入/新源码/旧来源hash。

六项门逐项继承原40轮配置，不因clean结果已知而修改：
R5相对C5需扰动mf1至少+0.005、clean F1最多下降0.005、clean/扰动Recall最多下降0.01，
clean/扰动空tile FPR最多增加0.0002。
C5相对历史B0-E需clean F1/扰动mf1最多下降0.005、两种Recall最多下降0.01，
两种空tile FPR最多增加0.0002；B0-E是100轮FP32不匹配背景，不用于声称数值/训练因果。
尚未计算的扰动/空tile门不能补写通过。

单seed开发结果和富集train残余不能称地域显著性、机制因果或算法创新。
Martigny和旧LEVIR test关闭，不准入基础模型、几何、外部数据、M5/K5/W5或P2。
闭环后保存报告；没有已通过文献/机制/计算pilot/有限预算/强普通对照门的下一实验就暂停托管。

物理开销保留：本批训练工程预检210次额外优化器计算、4158 pair forward views；
本次评价smoke102预测，全部为额外计算而非研究有效训练预算。
原筛查/恢复smoke、旧C-S95轮、A-S97轮至少1100步重算及更早未知未提交步数另列保留。
