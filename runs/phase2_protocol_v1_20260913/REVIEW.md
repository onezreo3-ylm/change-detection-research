# 训练集扰动强度：待视觉复核

这18张图是候选参数样例，不是模型效果图。每行依次展示clean A、clean B、
变化标签、注入区域alpha以及L1/L3/L5扰动。行左侧标注来源、方向、作用时相。
原256像素tile未被算子缩放；联系表为192像素展示缩略图。

尚未完成图像可读性的视觉验收。请重点检查小建筑是否仍可辨认、最高档是否
过度抹除信息、局部边缘是否出现不合理伪影。数值不变量通过不等于可观测性通过。

| 类型 | 全局 | 局部 |
| --- | --- | --- |
| 亮度 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/brightness_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/brightness_local.png>) |
| 对比度 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/contrast_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/contrast_local.png>) |
| Gamma | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/gamma_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/gamma_local.png>) |
| 色偏 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/color_bias_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/color_bias_local.png>) |
| 雾霾近似 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/haze_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/haze_local.png>) |
| 阴影近似 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/shadow_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/shadow_local.png>) |
| 噪声 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/noise_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/noise_local.png>) |
| 模糊 | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/blur_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/blur_local.png>) |
| JPEG | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/jpeg_global.png>) | [查看](<E:/new research/detection/runs/phase2_protocol_v1_20260913/previews/jpeg_local.png>) |

四个来源由训练文件名/固定种子选择，没有用标签挑样例。只有第2行的
`train_130.png@x0_y768`包含变化标签（2738/65536像素），其余三行无变化标签。
该覆盖范围不足以确认所有小变化目标的可观测性；如需补充样例，应事先固定
训练集选择规则并记录新版本，不按验证/测试分数挑选。

第2行含变化的原尺寸对照：
[A](<E:/new research/detection/runs/phase2_protocol_v1_20260913/calibration_tiles/sample1_A.png>)、
[B](<E:/new research/detection/runs/phase2_protocol_v1_20260913/calibration_tiles/sample1_B.png>)、
[标签](<E:/new research/detection/runs/phase2_protocol_v1_20260913/calibration_tiles/sample1_label.png>)、
[全局L5模糊](<E:/new research/detection/runs/phase2_protocol_v1_20260913/calibration_tiles/sample1_blur_global_l5.png>)。

操作定义和重复机制限制见 [协议说明](<E:/new research/detection/docs/CORRUPTION_PROTOCOL_V1.md>)。
不得根据后续验证/测试掉分反向调整这批强度；当前没有生成任何鲁棒性分数。
