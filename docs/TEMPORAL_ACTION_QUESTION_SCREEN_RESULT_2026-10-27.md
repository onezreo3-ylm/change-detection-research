# 双时相动作身份筛查结果：未准入新实现

2026-10-27。这是概念筛查，未改变或运行现有四区域探针，没有新模型、门或训练。

一手检索摘要已经显示，时间交换对称性不是新概念：[Change Is Everywhere / STAR,
ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Change_Is_Everywhere_Single-Temporal_Supervised_Object_Change_Detection_in_Remote_ICCV_2021_paper.pdf)
明确研究binary object change的temporal symmetry；[ChangeMask](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002835)
把temporal symmetry作为模型设计先验。因此若未来对A-only/B-only评分采用交换等变结构，
不能把时间对称性本身宣称为本项目的新贡献。

[AsymCD](https://www.sciencedirect.com/science/article/pii/S0925231226020904)的一手摘要
讨论空间范围与信息粒度不对称、配准和非对称门；它的几何设定不属于当前准入方向，
不能为了借用标题而加入配准。[CPEN](https://www.mdpi.com/2072-4292/18/15/2583)的一手
检索摘要讨论成对光度一致性与增强，而非仅各自增强。因此“同时考虑两日期的非语义
质量差异”也不足以单独形成新颖性主张。这里未声称这些摘要覆盖或排除了具体动作选择。

按范围用了两次搜索工具请求、六条查询、六个直接资源尝试。STAR PDF及landing、
ChangeMask、AsymCD返回403，CPEN返回429，FSRNet DOI转至IEEE机器人验证页；未重试
这些失败入口、绕过验证或获取外部代码/权重。可见的一手检索摘要只支持上述粗层次
排重，不能代替完整方法/实现对齐。完整近邻核查未完成，不宣称“没找到所以新”。

结论：A-only/B-only/both/none只是待审科学问题，当前不实现或训练。它不能替代当前
冻结探针失败后的原停止门；若将来提出，须是明示的新动作空间、独立版本、等成本
全部互斥候选解码后选择控制、原数据/模型/精度保持和实际资源准入。原二动作whole-tile
selector失败保持关闭，四区域先预测方案的稠密解码次数优势仍已撤回。
