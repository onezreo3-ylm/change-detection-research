# 扰动协议 v1 视觉验收记录

日期：2026-09-13。结论：**通过，仅批准绑定哈希的 validation 基准评分**。
这不是训练授权、held-out test 授权，也不把合成扰动解释为真实成像物理。

## 实际复核范围

- 逐页读取并检查原候选的 18 张 train-only 联系表，以及补充 QA 的 108 张
  联系表；合计 126 张，覆盖 9 类、global/local、L1/L3/L5、A/B 两个作用时相。
- 补充 QA 固定 12 个训练源 tile，其中 5 个为阳性；变化像素数覆盖 289、1024、
  5157、9885 和 18685。选择规则在读标签之前固定，没有读取 val/test 图片或标签，
  没有依据模型分数反调强度。
- 另以 256×256 原分辨率复核原候选 2738 像素阳性样本，以及 289、1024 像素
  小目标样本的 clean A/B、标签和 L5 noise/blur/gamma/haze/shadow 输出；两轮覆盖
  不同作用时相和局部区域。

## 视觉判断

- 九类在 L1→L3→L5 上均呈一致的视觉强度递增；JPEG 的变化相对温和，但顺序可辨。
- 未发现平移、缩放、旋转或 A/B 几何错位；未作用时相保持不变。
- 局部 alpha 为羽化椭圆，边界连续，没有一像素硬接缝；局部区域本身是人为干预形状，
  不应解释为真实天气或照明几何。
- L5 noise 与 blur 最强，明显损失纹理细节；但复核的建筑、场景轮廓和边缘小目标仍
  可由对应时相辨认，没有整片饱和、完全遮挡或几何破坏。因此保留候选五档，不再依据
  后续验证分数调整严重度。

## 已知限制

- 289 像素样本是 tile 边缘目标，L5 noise/blur 下精细边缘显著变难；本次通过只说明
  目标未被构造性抹除，不保证每个训练或验证目标在最高档均同等易辨。
- haze、shadow 和 local JPEG 是受控近似；brightness 暗化与 shadow 在部分参数下
  数学重叠。九个名称不代表九个独立物理机制。
- 联系表和固定小样本不能证明全数据集的可观测性；后续须按预注册的 45 条件报告分项、
  最坏条件和 harmful-rate，不能只报告等权均值。

## 哈希绑定

```text
protocol:   98f5e62efb60fa0cefd17c0191522cb11b0083755f1215484206a8b5e8469296
manifest:   951e1498f2d538d55be91ea2629ddde7edf1af6651bf97184db18ce2b7bc0039
instances:  01c6864811e6f20832dda13a754b5ed524d7be8b7e0d43cd33c41c3e022c4f2c
checkpoint: 033a0aa14becabc97c60b368b6d7c2e49bc8b386daa61dcc77e761c81ae747f3
```

机器闸门记录为
[`configs/corruption_v1_acceptance.yaml`](<E:/new research/detection/configs/corruption_v1_acceptance.yaml>)；
原候选与补充产物保持不变，分别见
[`runs/phase2_protocol_v1_20260913/REVIEW.md`](<E:/new research/detection/runs/phase2_protocol_v1_20260913/REVIEW.md>) 和
[`runs/train_qa_supplement_20260913/REVIEW.md`](<E:/new research/detection/runs/train_qa_supplement_20260913/REVIEW.md>)。
