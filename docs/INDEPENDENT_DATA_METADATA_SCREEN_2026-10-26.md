# 独立数据候选元数据结果

2026-10-26，Asia/Shanghai。范围见INDEPENDENT_DATA_METADATA_SCOPE_2026-10-26.md。
本轮没有获取、解压、查看原始数据，没有模型前向或新训练，也没有新数据准入。

## S2Looking

[作者仓库](https://github.com/S2Looking/Dataset)与[固定arXiv v3](https://arxiv.org/abs/2107.09244v3)
确认其为侧视建筑变化任务、全球乡村影像，并有分开的增长/拆除标注。
出版方检索可读的§2.3–2.4还说明残余配准误差及挑战移除地理坐标；直接出版方页429，
仓库子目录页面不可取、猜测README路径404，未继续绕过或循环请求。
[论文入口](https://www.mdpi.com/2072-4292/13/24/5094)。

当前不优先用它补本研究的“独立、非几何”缺口：全球来源尚不能核对与Texas的地理关系，
侧视/配准又引入了不同问题。这不是数据质量否定，也不声称不能用于其他研究。
在所检查作者页面中未核实数据专属许可；没有采用第三方汇总的CC0说法，亦未把论文许可当影像许可。

## SECOND

[作者项目页](https://captain-whu.github.io/SCD/)与[作者提供论文§IV](https://captain-whu.github.io/SCD/SCD_files/Semantic_Change_Detection.pdf)
说明4662对512像素影像、六类语义及Hangzhou/Chengdu/Shanghai等城市。
论文明确分别标注语义与变化，包含同一类别内的拆除/重建变化。因此，building子任务不能未经核查就
用两期语义类别不相等或XOR替代真实变化掩膜。原文§VI-A的2968/1694 train/test划分可作为待核对入口，
不在本轮重划分。固定书目版本另见[arXiv v2](https://arxiv.org/abs/2010.05687v2)。

**选择为下一项元数据/标签格式核查候选，尚未准入。** 这些已报告城市比全球范围更有利于来源审查，
但本轮没有逐文件地理或像元身份认证，也没有从论文中的城市举例推断全部样本独立。
需先从发布包说明核对building编码、non-change/独立change-mask语义、数据使用说明、包身份与完整split清单。
未得到明确schema前不实现标签转换，不把其他语义变化全部当建筑变化，不使用模型一致性代替真值。

作者页链接到Google Drive的second_dataset.zip。本次view文本只有登录导航，不能由此认定必须登录；
一次HTTP HEAD经公开入口返回200、Content-Type=text/html、Content-Length=2466，
应用未读取body，未得到ZIP体积或内容清单。因此不能宣称数据已下载、完整或已经可用。
原始头部记录在runs/independent_data_metadata_screen_20261026/second_public_release_head.json。
没有把这个HTML长度误报为数据ZIP大小。

## 后续具体门

仅推进SECOND发布说明/公开下载页的有界元数据读取及允许split的schema审查；先冻结范围，
不读取原始test影像/标签。若资源可用，才另写来源、标签、内容/重复审计、train-only QA及
有限预算协议。一个可能的后续问题是冻结旧B0-E/B1/B2 seed42做无调参的建筑子任务外部描述，
但当前没有接受其标签定义、没有冻结最终评价，也不开始45条件或新的长训练。

已消费LEVIR-test与Martigny不进入准入/调参，原普通控制队列继续。当前原48小时账本不扩展。
所有事实是文献/发布入口核查，不是已完成的独立泛化证据。
