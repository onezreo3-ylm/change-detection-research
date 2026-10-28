# 固定验证清单评估器

更新：2026-09-13。这里只评估现有checkpoint，不训练、不选新checkpoint、不访问test。
当前真实v1强度审核仍为pending；本轮用户的“继续”没有被写成已完成视觉验收。

## 三种入口

在 `E:/new research/detection` 使用既有解释器：

```powershell
$researchPython = 'E:/Download/anaconda3/envs/geosa-basa/python.exe'

# 只读预检：只读配置、清单、源码和版本记录，不打开原图或运行模型。
& $researchPython -m pcdc.evaluate_corruptions --config configs/evaluate_corruptions_val_v1.yaml --preflight-only

# Clean兼容性：独立目录，只读取真实val，精确复核旧best的历史结果。
& $researchPython -m pcdc.evaluate_corruptions --config configs/evaluate_clean_val_compat_20260913.yaml --clean-only

# 真实扰动评分：当前必须拒绝，直到存在与本批清单绑定的有效强度审核。
& $researchPython -m pcdc.evaluate_corruptions --config configs/evaluate_corruptions_val_v1.yaml
```

普通评分与clean-only使用不同输出目录，避免clean检查占用未来45条件运行目录。
所有运行拒绝覆盖已有输出；预检不创建研究结果目录。失败或中断产物保留，
不能把部分条件算成完整mF1，也不在原目录上自动重跑。

## 审核记录与边界

[真实审核文件](<E:/new research/detection/configs/corruption_v1_acceptance.yaml>)保持
`status: pending`、`visual_review: not_completed`、`reviewer: null`。
真实评分需要明确审核人、非空说明和通过的强度复核，且审核记录必须绑定：

- candidate协议指纹；
- manifest文件SHA256；
- 实例JSONL文件SHA256；
- 固定checkpoint SHA256。

不要修改旧candidate的status来伪造接受：status也是其指纹的一部分。
接受记录是独立文件；任何强度或清单变动都需要重新记录与审核，不能根据val
掉分改参数后仍使用同一个“已冻结”基准。训练、test和新模型组件从未被该记录授权。

`engineering_only`与`not_applicable_synthetic`只允许显式`kind: smoke`。
它不是视觉审核通过，不能用于真实研究checkpoint；checkpoint kind也会再次核验。
没有审核时，完整评分在设备初始化和数据访问前拒绝。只读预检/clean-only不批准扰动。

## 冻结内容如何核验

[frozen_validation.py](<E:/new research/detection/pcdc/frozen_validation.py>)会检查：

1. 任务YAML、归档base YAML、原candidate快照、manifest、JSONL、checkpoint身份。
2. 准备期15个Python源文件的当前内容和归档快照仍一致。允许添加新评估器文件，
   但不把“源码字典增添键”误当成原算子发生变化；新评估器另外保存自己的源码指纹。
3. NumPy、Pillow、JPEG codec和libjpeg-turbo版本与生成清单时一致。
4. 从冻结inventory枚举合法val tile，逐条重新计算实例并与保存的JSONL严格比较；
   拒绝缺失、重复、额外、越界、方向/参数/seed/区域不符或数量不符的条目。
   重新计算只用于核对，绝不静默替换保存的实例。
5. 真正评分前，只构建val数据集，比较manifest和checkpoint里的val inventory及
   每个A/B/label SHA。所有192个原文件还进入运行前后内容检查。

输入哈希使用已验证的期望值，不在中途重新采集“当前内容”当作新基准。
若配置、源码、审核、checkpoint、实例或原图/标签在运行中发生变动，
该运行不能标为completed。此为内容完整性检查，不是针对恶意并发修改的文件锁。

## 输入与数值计算

`FrozenInstanceDataset`按明确sample ID寻找原图及tile坐标，在原uint8 RGB上
施加保存的扰动，然后按原顺序float32 `/255`、减mean、除std。标签、另一时相和
alpha=0区域保持不变。模型只接收归一化A/B两张图，绝不接收类型、严重度、alpha、
真值或clean对照。seed超过int64时也只保存在JSON字符串中，不交给默认collate转整数。

真实配置继续使用CUDA、AMP、channels-last、eval batch32、4 workers、严格确定性。
每个条件保持单独batch边界，不混合不同条件；全局阈值仍是0.5。启动CUDA失败即失败，
不改用CPU、随机权重或别的模型。

首先运行clean val，对所选best epoch历史记录的4类计数及五指标进行精确比较；
不一致就停止，不继续评分、不重新选择checkpoint或调整阈值。

## 指标与分层

[evaluate_corruptions.py](<E:/new research/detection/pcdc/evaluate_corruptions.py>)对每tile只计算
一次预测布尔图，并用确定性int64求和累加TN/FP/FN/TP，未使用CUDA bincount。

每条件报告像素池化的Precision、Recall、F1、IoU、OA，以及：

- 背景误报率 `FP/(FP+TN)`；
- 变化漏检率 `FN/(TP+FN)`；
- 相对clean F1的下降；
- 方向/时相/全局局部的联合层和各边际；
- 每个原始影像对的计数，方便后续按原图而非相关tile重采样；
- 注入支持域 `alpha>0` 和其外 `alpha==0` 的计数与分母。

全局扰动的域外没有像素，因此其`pixels=0, metrics=null`，不是一个实测零分。
这些区域指标描述预测误差，不能直接当作校正动作的因果损伤或FP/FN转移。

最终mF1是45个条件F1的等权平均。不得平均tile F1，不得把45条件计数混池后
算一个F1替代mF1。仍须保留brightness/shadow算子重叠及有限校准覆盖的限制。

## 产物与运行范围

每次运行保存配置、完整评估源码快照、输入哈希、环境、审核记录、clean计数。
只有审核通过的完整评分才创建逐条件JSON、45条件汇总、家族mF1和整体mF1。
本轮真实运行只允许clean-only；45条件端到端仅用于独立合成工程夹具。

合成CUDA配置为 [smoke配置](<E:/new research/detection/configs/smoke_validation_evaluator_20260913.yaml>)，
只训练1 epoch，4/2对train/val、64像素tile、0 workers；0 workers是缩小工程耗时
的显式设置，不改变真实评估的4 workers。另有实际双spawn-worker加载回归。
所有合成分数只是工程数值，不能进入研究结果表。

正式长时间45条件运行前，仍需确认强度并为新评估进程接入现有温度保护策略；
不能复用已经退出的训练PID。接口保留每batch及结束时的`guard_callback`，本批没有
启动长评分、自动续训或新的后台监控。实际验收与clean复现结果另见执行记录。
