# 当前论文工作包

入口manuscript.md，参考资料references.md，数字来源tables/verification.json。
这是本地中文工作稿，不是投稿终稿。原9月文稿、旧研究结果与原始数据均未覆盖。

五张表从七份历史结果及固定动作结果生成，CSV、Markdown和未舍入JSON均保留。
B0-E43/44两格明确pending，三seedB0-E均值与主效应保持空值；没有填入当前训练的中间分数。
build_tables.py完成真实保存结果的生成smoke，并用NumPy和既有R1汇总独立核对均值/SD，
还检查效用符号、转换率及八策略点差方向。没有新推理、新重采样、checkpoint加载或rawtest读取。

复现表格时在新的工作包副本执行：

```powershell
& 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -X utf8 runs/manuscript_working_draft_20261026/build_tables.py
```

此脚本拒绝覆写已有tables目录，保留现有输出。正式新结果到位后应另建版本，读取完成认证的
九格分析manifest与结果，而不是直接改写本次历史快照或回填当前表格。

当前已识别的重要解释边界：

- 动作实验actor只用356个train源，完整基线用445源，不能直接按跨组分数差归因。
- Phase7的主参考是按事前规则选择官方val上较优的A0/G0，本次为G0；G0身份本身并非预先固定。
- pooled腐蚀F1与45条件mF1不同；表格使用后者为主终点。
- 标签知情O以任务损失选择实际动作，不能称F1上界或部署选择器。
- B2区间跨零不能写成等价或普遍有害；未通过联合门只支持原开发决策。
- 最终B1test是审计恢复的历史单模型描述，原V1仍失败，当前不能补测方法追分。

稿件仍缺B0-E多种子完整结果、额外模型/独立数据证据与更充分近邻核查。
代码绿灯、生成了文稿或表格并不意味着论文贡献成立。
