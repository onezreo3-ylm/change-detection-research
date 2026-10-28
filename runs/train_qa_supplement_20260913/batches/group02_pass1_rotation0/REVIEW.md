# group02_pass1_rotation0

以下为训练集预先固定样本。变化像素计数为选定后统计，不参与选样或注入区域生成。
照片内容尚未被本次会话实际视觉验收；请注意最高档小目标/边缘是否仍可辨认。

| 行（从1计） | 来源 | 变化像素/65536 | 原尺寸对照 |
| --- | --- | --- | --- |
| 1 | train/train_146.png@x512_y256 | 9885 | [A](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample0_A.png>) / [B](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample0_B.png>) / [标签](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample0_label.png>) |
| 2 | train/train_27.png@x256_y512 | 1024 | [A](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample1_A.png>) / [B](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample1_B.png>) / [标签](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample1_label.png>) |
| 3 | train/train_393.png@x0_y512 | 0 | [A](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample2_A.png>) / [B](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample2_B.png>) / [标签](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample2_label.png>) |
| 4 | train/train_215.png@x512_y512 | 0 | [A](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample3_A.png>) / [B](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample3_B.png>) / [标签](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/calibration_tiles/sample3_label.png>) |

每张表依次展示clean A、clean B、语义标签、注入alpha、L1/L3/L5。左侧标记实际方向与目标时相。

| 类型 | 全局 | 局部 |
| --- | --- | --- |
| 亮度 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/brightness_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/brightness_local.png>) |
| 对比度 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/contrast_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/contrast_local.png>) |
| Gamma | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/gamma_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/gamma_local.png>) |
| 色偏 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/color_bias_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/color_bias_local.png>) |
| 雾霾近似 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/haze_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/haze_local.png>) |
| 阴影近似 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/shadow_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/shadow_local.png>) |
| 噪声 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/noise_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/noise_local.png>) |
| 模糊 | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/blur_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/blur_local.png>) |
| JPEG | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/jpeg_global.png>) | [查看](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/previews/jpeg_local.png>) |

[返回补充总索引](<E:/new research/detection/runs/train_qa_supplement_20260913/REVIEW.md>)。
第二轮循环移位既改变时相，也可能改变多方向类型与seed/mask；不能当作仅时相不同的配对对照。
