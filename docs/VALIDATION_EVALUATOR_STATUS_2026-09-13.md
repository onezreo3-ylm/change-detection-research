# 固定验证评估器：本轮执行结果

最新续接：视觉链路在此前短暂成功后再次省略图像；训练集补充QA已经生成，
完整测试增至145项通过，真实强度接受仍为pending。
见 [最新状态](VISUAL_QA_CONTINUATION_2026-09-13.md)，下文保留原评估器验收结果。

日期：2026-09-13。响应“请继续下一步”，完成了可以独立于强度视觉判断推进的
评估器实现、工程验收及真实clean val兼容性复核。**真实扰动评分仍未启动**。

后续更新：视觉工具已在同日14时后实测恢复，原18张联系表已技术复核；发现
局部扰动对新增建筑覆盖不足，正在增加独立train-only QA与温控评分入口。
见 [视觉恢复续接记录](VISUAL_QA_RESUMED_2026-09-13.md)。下文“图像不可读”是
本报告写作时的历史状态；真实强度接受记录仍未批准。

## 完成项

- 新增 `pcdc/frozen_validation.py` 和 `pcdc/evaluate_corruptions.py`；原准备期15个
  Python源文件保持原SHA。既有数据、checkpoint和候选参数/清单未修改。
- 预检逐条重放核对46,080条冻结实例，允许新增评估文件但拒绝原算子或数据协议漂移。
- 单独支持clean-only；完整45条件评分必须经过独立、哈希绑定的强度审核记录。
  当前审核pending已实测提前拒绝，不创建真实扰动评分目录，不访问图像或启动模型。
- 模型只收到A/B；类型、参数、区域和真值不进入模型。指标按条件内像素累计，
  保存联合层、边际、注入区内外及原始影像对计数。空区域记为无观测而非零分。
- 全部val原图/标签绑定checkpoint/manifest中的既定SHA，运行前后复查；不把中途
  变化重新采集成基准。相关参数和运行命令见 [评估器说明](FROZEN_VALIDATION_EVALUATOR.md)。

## 验收证据

| 检查 | 实际结果 |
| --- | --- |
| 完整unittest | 112项通过、0失败、0跳过，59.293秒 |
| 新包装器多进程 | 两个真实spawn worker与单进程结果一致；uint64 seed安全保留为JSON |
| CUDA基础smoke | 1 epoch合成train/val、best/last及独立val复核通过 |
| CUDA扰动端到端 | 合成数据完成45条件、90个实例；45个条件JSON均已落盘 |
| 工程/研究隔离 | engineering_only不能批准research checkpoint；合成test目录不存在 |
| 源码复核 | 当前17个评估源码与本轮归档快照均一致，原15个冻结源码未改变 |

CUDA smoke明确使用`kind: smoke`、64像素合成图和0 workers以缩短工程耗时；
真实评估保持4 workers、batch32、256tile、AMP与channels-last。没有设备或模型
静默替代。合成分数无论高低都不是鲁棒性结果，也不进入真实研究结果表。

日志：
[112项测试](<E:/new research/detection/runs/validation_evaluator_unittest_20260913.log>)、
[CUDA基础smoke](<E:/new research/detection/runs/validation_evaluator_cuda_base_smoke_20260913.log>)、
[45条件合成CUDA](<E:/new research/detection/runs/validation_evaluator_cuda_45_smoke_20260913.log>)。
只有已知TypedStorage弃用警告；单元测试中的test均为合成夹具，不是重新读取真实test。

## 真实clean val兼容性

新评估器加载既有epoch69 best checkpoint；只评估原val的64对、1024tile，阈值0.5。
四种混淆计数和五指标与原checkpoint的最佳验证记录**精确一致**：

| Precision | Recall | F1 | IoU | OA |
| --- | --- | --- | --- | --- |
| 0.9135744391 | 0.8985018418 | 0.9059754546 | 0.8281125486 | 0.9921735227 |

TP=2530413，FP=239381，FN=285845，TN=64053225，总像素67108864。
背景误报率0.0037233053，变化漏检率0.1014981582；它们仅描述这个clean基线。
评估循环23.11秒，包含循环内加载/计数，不包括前置哈希与模型构建，也不是正式测速对照。

产物：
[clean指标](<E:/new research/detection/runs/levir_clean_val_evaluator_compat_20260913/clean_metrics.json>)、
[运行摘要](<E:/new research/detection/runs/levir_clean_val_evaluator_compat_20260913/summary.json>)、
[机器复核记录](<E:/new research/detection/runs/levir_clean_val_evaluator_compat_20260913/verification_20260913.json>)。
本轮追踪216个输入文件，其中192个为真实val的A/B/label；运行前后检查通过。
这只是旧基线复现，不是新模型、新seed或扰动鲁棒性结果。

## 仍需用户确认

图像工具仍未返回可读取画面，预览面板连接也不可用，不能宣称强度视觉复核完成。
[当前v1样例索引](<E:/new research/detection/runs/phase2_protocol_v1_20260913/REVIEW.md>)包含18张训练图像联系表。
原4tile中仅1张有非空变化标签；有限覆盖、brightness/shadow数学重叠等限制仍然保留。

需要用户明确确认已复核并接受当前v1强度用于验证集评估，随后记录审核人、说明
及四类绑定哈希；不由“继续”自动补出“视觉已通过”。不得根据后续val成绩修改强度。
在此之前，只允许预检/clean-only，真实45条件评分目录仍不存在。

没有新的正式训练、PSE/MoE/校正器/门控、真实test或后台监控；所有本轮进程已结束。
