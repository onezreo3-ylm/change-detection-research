# 训练集补充 QA：待视觉复核

本目录新增12个训练源样本，不改变v1强度、原验证清单或checkpoint。
共2160个实例、108张联系表，每个样本/类型/空间模式覆盖A、B两个作用时相。
工程不变量已经通过；本次视觉链路仍省略图片，尚未填写视觉通过或正式接受。

| 组/轮次 | 行顺序与变化像素数 | 样例入口 |
| --- | --- | --- |
| group01_pass1_rotation0 | train_171.png@x0_y0：289<br>train_185.png@x0_y0：18685<br>train_397.png@x256_y768：0<br>train_237.png@x0_y256：0 | [打开全部9类](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group01_pass1_rotation0/REVIEW.md>) |
| group01_pass2_rotation1 | train_185.png@x0_y0：18685<br>train_397.png@x256_y768：0<br>train_237.png@x0_y256：0<br>train_171.png@x0_y0：289 | [打开全部9类](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group01_pass2_rotation1/REVIEW.md>) |
| group02_pass1_rotation0 | train_146.png@x512_y256：9885<br>train_27.png@x256_y512：1024<br>train_393.png@x0_y512：0<br>train_215.png@x512_y512：0 | [打开全部9类](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass1_rotation0/REVIEW.md>) |
| group02_pass2_rotation1 | train_27.png@x256_y512：1024<br>train_393.png@x0_y512：0<br>train_215.png@x512_y512：0<br>train_146.png@x512_y256：9885 | [打开全部9类](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group02_pass2_rotation1/REVIEW.md>) |
| group03_pass1_rotation0 | train_395.png@x768_y768：0<br>train_267.png@x0_y768：0<br>train_106.png@x256_y0：5157<br>train_343.png@x0_y768：0 | [打开全部9类](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group03_pass1_rotation0/REVIEW.md>) |
| group03_pass2_rotation1 | train_267.png@x0_y768：0<br>train_106.png@x256_y0：5157<br>train_343.png@x0_y768：0<br>train_395.png@x768_y768：0 | [打开全部9类](<E:/new research/detection/runs/train_qa_supplement_20260913/batches/group03_pass2_rotation1/REVIEW.md>) |

新增12个tile中5个有非空变化标注。5个阳性样本的90个local/L5/类型/时相组合中，63个与变化标注至少有1像素交集（作用于A的32/45，B的31/45）。这只是空间覆盖统计，不证明目标可见或变化区域被完整覆盖。

原始4个来源未复用，选择仍按固定文件名hash、seed=730194完成；不读取val/test图像，不按模型分数调强度。
[完整数值与溯源记录](<E:/new research/detection/runs/train_qa_supplement_20260913/train_qa_supplement.json>)。
