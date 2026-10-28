# Valais v1 扰动视觉验收与 validation 评分协议

日期：2026-10-02。承接[中断续接记录](EXPERIMENT_CONTINUATION_2026-10-02.md)。候选产物保持原状态和原文件；评分许可单列在[acceptance YAML](../configs/valais_corruption_v1_acceptance_20261002.yaml)，仅授权固定 Sierre validation 清单。

## 真实视觉检查及其限制

本对话实际重新查看了方向均衡补充批全部18张联系表，另看首批 global blur/noise/JPEG 三张联系表；对 train/7174.tif 查看原尺寸256像素的 clean A/B、标签和 L5 global/local blur/noise，共7张原图。不得把此记录写成在本对话看完36张联系表、全部原图或所有验证图像。

两批训练 QA 共8张正类、8张空标签且来源不重叠。核对了16个来源文件的 frozen train 内容哈希、1440条记录的标签/另一时相/注入区域外像素/输入不修改不变量，以及288张保存的 L5 原图像素哈希。外部取样明确使用训练标签作正/空分层；生成器继承的 `selection_uses_labels=false` 不应被解释为整个取样过程未使用标签。算子参数与注入 alpha 不读语义标签。

检查中 L1/L3/L5 强度有可见次序；选中场景与较明显建筑轮廓可对照，局部边缘羽化未见硬接缝。L5噪声/模糊显著损失细节，薄小或 tile 边缘目标不能保证逐个恢复；16张样例也不能代表所有Valais影像的可辨认性。v1接受为固定、有限的开发挑战，不能据此宣称所有扰动标签均可由图像完整恢复。brightness/shadow 数学重合、RGB haze 代理以及 local JPEG alpha blending 的非物理性必须披露。

证据记录为[visual_review_resumed_20261002.json](../runs/valais_corruption_protocol_preflight_20261002/visual_review_resumed_20261002.json)，SHA256 `2ab4752a2a28f852619273836c110b969a444d71116dcf30ba80be539038c90f`。其查看文件和来源报告均有哈希绑定。强度未依据 validation/test 分数选择或修改。

## 冻结验证输入与报告

- [候选配置](../configs/valais_corruption_protocol_v1_20261002.yaml)及[准备快照](../runs/valais_corruption_protocol_v1_20261002/candidate_resolved.yaml)使用相同九类、五级参数和局部mask；train/val/reserved-test seeds为510942/510943/510944。保留test seed不表示创建或执行test。
- [清单](../runs/valais_corruption_protocol_v1_20261002/val_manifest.json)含1715个Sierre验证tile，每个45条件；[实例文件](../runs/valais_corruption_protocol_v1_20261002/val_instances.jsonl)含77175个实例。协议哈希 `aa16a00382be46d2e6d9f1918d072f86fbee1ccef0a3eb70da462cdad52c4416`，清单文件哈希 `aa6ee6f1bf4d72ac61d1299c649398b34872e9a50431fbe649c899389b52737c`，实例哈希 `d1f9252d8252f3b1ce4ac658a5f8d6f92d81a79f1a0fb1dd35ccd6b54b1523d7`。
- [评估配置](../configs/evaluate_valais_baseline_corruption_v1_20261002.yaml)固定完成clean基线的epoch53 best.pt，SHA256 `de42460dfce94717a890ae359124c659ce57d7d396f44429ac370463e1991b7d`。共享ResNet-18/FPN、FP32、原严格0/1标签、RGB归一化、阈值0.5、eval batch32、固定split与clean-val选模不变。
- 先重算clean-val，要求同一checkpoint的pooled F1精确复现0.4162248161980205；再评估45条扰动条件。每条件先汇总全验证像素混淆矩阵，再等权平均45个F1；报告family及方向/目标时相/global/local分层、FP/FN与召回。不得以平均tile F1代替pooled F1，也不对无独立地理来源ID的tile进行独立空间bootstrap。

## 执行保护和下一步

使用现有[guarded evaluator](../scripts/evaluate_guarded.py)与[温度配置](../configs/evaluation_temperature_guard.yaml)，绑定GPU UUID `GPU-8ada1895-7628-bb88-1dda-c57679117fa2`。在批次间同步检查温度与守卫源码/配置；85°C连续确认或86°C单次触发时停止，不自动重启。仅在status、summary、temperature_guard_state及guard_completion一致且产物哈希复核通过后报告正式完成。这是批次间软件停止策略，不能称为硬温度上限。

预检必须通过、全量单测和真实FP32 CUDA smoke必须有成功记录，GPU不得已有研究计算进程。输出目录[baseline corruption val](../runs/valais_baseline_corruption_val_v1_seed42_20261002)必须原先不存在；运行前登记准确解释器、命令、进程身份及所有输入/实现哈希。任何失败保留输出，不在该目录补写完成或自动续算；恢复另立审计/协议。

本次只测已有clean模型的扰动退化。B1训练须另立匹配协议；校正/门控模型、Martigny test及已用LEVIR test仍未开放。后续小时heartbeat在本对话跟进，运行健康时静默，完成、失败或实质新证据时通知。
