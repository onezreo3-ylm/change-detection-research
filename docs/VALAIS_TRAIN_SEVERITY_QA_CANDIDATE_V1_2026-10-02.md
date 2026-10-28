# Valais train-only 扰动强度候选 QA v1

日期：2026-10-02。状态：**candidate_pending_visual_review**。clean 基线已有完整100轮和受守卫独立 Sierre clean-val 复核；best epoch53，F1@0.5=0.4162248162，TP=124266、Recall=0.4067214。该单城开发结果说明检出正类，允许进入扰动准备；不证明跨地区鲁棒性。

[候选 YAML](../configs/valais_train_severity_qa_candidate_v1_20261002.yaml)沿用已经实现的九类非几何 RGB 算子、五级**候选**数值与椭圆局部 alpha，明确绑定原[LEVIR 候选 YAML](../configs/corruption_protocol_v1.yaml)的哈希。这是跨数据集的初始可视化试探，不把 LEVIR 数值自动视为 Valais 已接受强度。扰动发生于 256 像素 tile 的 uint8 RGB、归一化之前；只改一个时相，保留另一时相、标签与像素网格。brightness/shadow 数学重合、合成 haze/JPEG 的物理局限仍按原候选协议披露。

选择只用冻结的 **Sion train** 标签清单：按固定 SHA256 种子分别排序正类和空标签 tile，各取 4 张，再正/空交替。类别分层是为了检查变化目标在强扰动下的可辨认性；主扰动参数与注入 alpha 不读语义标签。每张覆盖 9 family × global/local × L1–L5，方向按预定行轮转，目标时相按行与 spatial 模式交替；共 720 个 QA 实例。联系表展示 clean A/B、标签、注入 alpha 与 L1/L3/L5；原 256 像素图像不被算子缩放。记录 sample ID、方向、目标时相、局部几何、参数、种子、输出像素哈希、区域外不变性和数值占比。标签仅用于事后视觉与覆盖判断。

[准备入口](../scripts/prepare_valais_train_qa.py)不运行模型、不读取 val/test 影像或标签，也不创建正式 val/test 扰动清单。必须核对 checkpoint、独立 clean-val 复核、原候选数值、train 标签报告和选中 train 文件哈希。视觉检查重点是小变化目标仍可见、不过度饱和/裁切、局部边界自然，以及 L5 是否需要缩减。未经视觉审查的实例只叫**候选 QA**，不能用于正式鲁棒性分数或 B1 训练。

视觉 QA 通过后另立 Valais 正式扰动参数版本、固定 validation 清单及普通 B1 matched 协议；阈值、split、模型、seed、初始化、FP32 数值与 clean-val 选模规则保持，任何改变单列记录。Martigny test 封闭，已使用的 LEVIR test 不重开。
