# 论文图与实验登记补充包

2026-10-27，Asia/Shanghai。基于已保存结果的工作包，论文仍未投稿就绪。
本文与原工作稿[manuscript.md](<E:/new research/detection/runs/manuscript_working_draft_20261026/manuscript.md>)配套。

当前采用r3。初版图1的标题/样本量标注重叠；r2修正间距，r3进一步修复登记表中带空格的
本地链接。原版、原验证记录均保留。r2/r3的PNG字节一致，因此r3沿用两张r2的实际视觉核对。

- [图1：九族动作效用符号](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/figures/figure1_action_signs.svg>)
- [图2：全部八个部署策略与诊断O](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/figures/figure2_policy_differences.svg>)
- [图注及解释边界](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/figure_captions.md>)
- [20项有文档的实验族登记](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/experiment_register.md>)
- [登记CSV](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/experiment_register.csv>)
- [未舍入图数据](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/plot_data.json>)
- [SECOND失败现场审查](<E:/new research/detection/runs/publication_evidence_audit_20261027_r3/second_transport_review.json>)

图1逐族分母均为7120，所有符号计数之和为64080，与原Phase6汇总及独立整数核对一致。
图2原始点估计、原保存区间、decision中的差值三者一致；没有新bootstrap、p值、模型输出
或缺失B0-E填值。SVG保留文本，PNG仅作渲染预览。20项登记为事后透明性说明，不是调用总账
或所有试验都事先注册的证明，历史验证调用总数仍unknown。

五项训练归档工程CPU测试通过。原完整训练前526项测试不代表全部新工具已完成验收；
全项目测试与模型smoke仍待当前CUDA训练队列结束，不并发抢占GPU。

复现图和登记时使用新的输出目录（拒绝覆盖）：

```powershell
& 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -X utf8 -m scripts.build_publication_saved_evidence --output-dir runs/publication_evidence_audit_20261027_new
```

当前训练B0-E43→44及既有45条件评价顺序不变。SECOND公开下载配额导致传输停止，
保留partial不可作为数据；使用条件、building编码、内容/来源审查及train-only QA尚未齐备。
其他旧候选WHU/HRSCD的问题见9月28日与10月24日来源审查，未在本轮重复下载或换入实验。

这些图表使已有证据可审阅，不解决单架构、动作单seed、反复使用开发集和缺少独立冻结后
验证等实质缺口。不能由工作包完成推导论文贡献或发表成立。
