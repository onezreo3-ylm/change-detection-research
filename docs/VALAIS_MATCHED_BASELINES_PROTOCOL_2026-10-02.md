# Valais B1 / B0-E 匹配对照协议

日期：2026-10-02。用户在本对话审阅重新评估后要求“开始执行”。授权本批整理已完成
B0 扰动结果、固定训练样例诊断及 B1/B0-E 两个独立100轮训练与一次性 Sierre 验证。
不授权 B2、类别平衡、校正/门控训练或 Martigny/已使用 LEVIR test。

## 冻结变量与预算

复用 Valais v1 已接受的训练视觉证据、九类五级算子、方向/时相/局部区域参数和
train seed510942、val seed510943；不依据已观察的 validation 分数改变强度或挑选类型。
原 acceptance 仅授权原 B0 验证，保持原样。新训练 review sidecar 与本批授权 seal
单独记录训练许可。旧训练器的 `approved_for: validation_only` 字段指开发基准的
评价生命周期；本批训练范围在新 sidecar、本文和 authorization 中明确，不继承
原 acceptance 的“no training”说明。候选 YAML/清单原状态和哈希不修改。

两个运行共用候选准备时归档的 `base_config.yaml`，共享 ResNet-18/FPN、随机初始化
seed42、Sion11025/Sierre1715、严格0/1、256输入、原RGB归一化、原损失/AdamW/cosine、
100轮、batch4/eval32、FP32 train+val、阈值0.5和首次最高clean-val F1选模。
各自从头训练；不续训 B0。每轮2757更新，每项总275700成功更新、0跳步、2205000
pair views；末尾单例 source batch 不丢弃。保持均匀源采样，不过采样、不改权重。

- B1：clean+perturbed，均有变化监督；无一致性。
- B0-E：clean+相同clean副本，均有相同监督。
- 一次合并两pair view前向，模型内部再合并A/B；每step一次编码器BN更新。

复用原训练器和守卫，原模型/损失代码不修改。单列 stage wrapper 只附加实际
source ID 顺序 SHA256、样本/批次数、初始张量哈希、实测峰值GPU allocation和进程
wall time；不改取样、前向、梯度、损失或优化算法。训练器原 provenance 加本批
wrapper seal 一并保留。B1/B0-E实际顺序、初始化和曝光必须相同，否则拒绝匹配结论。

## 训练样例诊断

仅Sion train：以冻结train正像素中位数556分成小/大正类，各按独立SHA种子810210
取2张，再取4张空标签。选图不使用模型得分或val/test。固定epoch53 B0，FP32、
eval()、单tile前向，保存原256像素A/B、标签、TP/FP/FN和概率；单tile结果不是
正式split指标。选中内容必须与B0 checkpoint文件哈希一致，温度守卫完成后做
原尺寸视觉审查，不从8张样例推断总体标签正确率或删去困难验证样例。

## 评价与预声明门

先重放各自clean-val最佳checkpoint，精确复现其记录，再评分同一45条件、77175
固定实例。每条件pooled像素F1后等权求mF1。相对B0报告问题改善，相对B0-E归因
扰动监督；报告clean F1、clean与扰动Recall、背景FPR和空标签tile误报。

本批seed42的**描述性筛选门**同时要求：

1. B1相对B0-E和B0的mF1增益均至少0.005（0.5个百分点）。
2. 相对B0-E和B0分别检查：clean F1下降不超过0.005；clean Recall下降不超过0.01；
   pooled corruption Recall下降不超过0.01。
3. pooled corruption空标签背景FPR相对B0-E和B0的增量不超过0.0002。

以上是本次在新B1/B0-E得分前冻结的实用筛选值，不是旧预注册门或已证明的显著性。
Sierre只有一城且缺少可信空间源ID，不做tile bootstrap、不报告空间独立置信区间。
通过只支持另列三seed确认方案；机制实验仍需独立训练动作空间、可预测性和误伤
证据门。未通过不自动改阈值、强度、采样或损失，也不自动启动新模型。

## 执行与保护

全量unittest与真实CUDA双视图FP32 smoke必须成功，smoke同batch4、一次有限反向/
更新/BN检查，仅工程记录。准备后审阅训练图并seal配置、协议、数据审计、checkpoint、
视觉证据、测试与smoke日志、训练/评估/守卫源码和解释器。

一次性隐藏Windows任务，无trigger、无retry、IgnoreNew；依次B1训练→B1验证→B0-E
训练→B0-E验证，技术失败或温度守卫停止即停止剩余队列，保留现场，不自动恢复。
85°C连续确认或86°C单次停止，传感器/身份/持久化异常拒绝继续；这是批次间软件
策略。任何源码/输入seal变化拒绝新阶段。只有训练/评估与guard/hash一致才称完成。
外层只调用Python完成核验，避免旧PowerShell顶层JSON数组的嵌套计数误判。
