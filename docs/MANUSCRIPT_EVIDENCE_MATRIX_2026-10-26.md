# 论文主张与证据矩阵

2026-10-26。工作稿；不是投稿就绪认证。主文围绕“诊断到的动作收益是否能转化为可部署鲁棒收益”，
将后续探索放在明确分开的扩展中。原9月内部稿及其哈希不修改。

| 可审阅主张 | 原始结果入口 | 适用范围 | 尚缺证据/不能写成什么 |
|---|---|---|---|
| 普通增强相对干净训练在所测合成条件有较高绝对分数 | `runs/levir_b0e_corruption_val_v1_seed42/summary.json`，`runs/levir_b1_corruption_val_v1_seed42/summary.json`；Phase4报告 | 单seed42、同架构、开发val | 缺B0-E43/44；不能由三seedB1/B2比较代替增强效应重复性 |
| 输出一致性没有通过已声明的增量门 | `runs/r1e_protocol_20260926/analysis.json`、`completion_audit.json` | 三个seed、原64个开发源、固定45条件 | 区间跨零是没有足够证据，不能写成两方法等价或一致性普遍无效 |
| 标签知情动作效用存在正负异质性 | Phase5–6报告及其原动作清单/效用产物 | 固定detector与adapter、train-gate | 标签知情诊断不是可部署策略，也不是所有可能动作的理论上界 |
| 所测整图策略未转化诊断空间 | `runs/levir_phase7_val_seed42/decision.json`、Phase7报告 | 单seed、冻结策略和常数控制 | 不推广到所有局部策略、所有选择器或其他架构 |
| 单个预选模型的原最终测试分数可由保存产物核对 | `runs/levir_b1_final_test_v1_seed42/`，V1R2恢复证书与审计 | 单checkpoint、已消费test、一次原推理 | 原V1守卫完成失败；不得写“原运行正常完成”，不得补测其他方法后称原最终比较 |
| Valais匹配普通增强未过其开发门 | `runs/valais_matched_study_v1_20261002/comparison.json`及10月4日报告 | 原未筛除数据、Sion→Sierre开发、FP32seed42 | 监督歧义未裁定；不是无偏跨城最终效果，不能与论文筛除版分数直接比 |
| 条件尾部重权未通过clean/总体/尾部联合门 | `runs/levir_tail_risk_evaluation_20261025/analysis.json` | 单seed、同扰动族、已观察开发val | 不能因最差5条件点估计略升就宣称鲁棒改善；不否定CVaR一般原理 |
| 固定模型同一像素随读取窗口改变预测 | `runs/levir_tiling_context_probe_20261025/inference/analysis.json`、独立保存核对 | 固定16张已见train源、共同支持 | 不是泛化提升，也未分离卷积上下文与GN作用；归一化分块问题已有直接近邻 |
| 配对大窗口训练没有通过原效果门 | `runs/levir_native_window_training_20261025/analysis.json`及10月26日报告 | seed42、等像元/损失区域/组顺序、原256开发评价 | F1+0.011238个百分点且FPR变差；不能称“上下文方法有效”或GN因果证据 |
| 普通增强效应的缺失控制正在准备 | `docs/B0E_REPLICATION_PROTOCOL_2026-10-26.md`、新run的实际status | 明确补充seed43/44，继承原共享预算 | 只有实际运行和评价完成后才能填结果；工程smoke/pilot不填论文效果表 |

主文应报告绝对cleanF1与corruptionmF1、Recall和FPR，而非只报告退化gap。
mF1是45个条件各自pooled像素F1的等权均值，不能换成tileF1均值或将45条件混为单张表。
像素、同源tile、同源45次腐蚀都不是独立地理重复。源对bootstrap条件于现有训练seed；
三个seed的均值/样本标准差另报，二者不合称已估计所有训练随机性。

扩展失败候选是探索历史，不能与早期冻结比较合并后声称所有实验均在统一预注册下完成。
完整实验清单保留失败、被预算拒绝及未准入项，表图不得仅挑选对当前叙事有利的条件。

下一写作产物应从机器计数生成统一结果表、来源/协议表和限制说明，保留待完成单元为空。
当前仍有模型覆盖、独立开发/最终保留证据及直接近邻差异不足等问题；是否形成可投稿论文须在
这些实际证据到位后再判断，不能由“代码通过测试”推导论文贡献成立。
