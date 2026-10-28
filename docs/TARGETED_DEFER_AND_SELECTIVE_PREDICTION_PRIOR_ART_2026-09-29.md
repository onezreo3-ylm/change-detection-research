# 针对动作选择主张的补充近邻复核：learning-to-defer 与 reject option

复核日期：2026-09-29。用途：约束[内部论文初稿](MANUSCRIPT_DRAFT_NEGATIVE_RESULTS_2026-09-28.md)的主张，不修改实验、不新增模型或测试分数。本次只读核对了 arXiv 的版本记录与摘要；Mozannar–Sontag 的 HTML 正文另核对了摘要/引言关于分类器、拒绝器和 cost-sensitive reduction 的表述。**未**系统阅读全文、代码或出版终版，所以下表只给摘要/引言明确支持的事实，不作全面优先权或性能比较。

| 文献与所核对版本 | 原文明确的决策问题 | 与本项目的关系及不能据此宣称的内容 |
| --- | --- | --- |
| [Mozannar 与 Sontag, *Consistent Estimators for Learning to Defer to an Expert*, arXiv:2006.01862v3](https://arxiv.org/abs/2006.01862v3) | 学分类器与 rejector，在自己预测与交由下游专家之间选择；以 cost-sensitive learning 的一致 surrogate 分析。 | “依据下游分支效果学习是否转交”已有理论近邻。此处专家决策与本项目固定、自动执行的特征校正动作并不相同，但本项目不能把“有不操作选项/基于效果选择”称为首次提出。 |
| [Verma、Barrejón 与 Nalisnick, *Learning to Defer to Multiple Experts: Consistent Surrogate Losses, Confidence Calibration, and Conformal Ensembles*, arXiv:2210.16955v2](https://arxiv.org/abs/2210.16955v2) | 研究多专家 deferral 的一致 surrogate、专家正确概率的校准及按需查询专家子集。 | 多分支/多专家路由和校准不是空白。其专家可能需被查询；不能把多个固定特征动作当作无代价人类专家，也不能仅凭本项目的标签知情选择分数声称可部署路由收益。 |
| [Hemmer 等, *Learning to Defer with Limited Expert Predictions*, arXiv:2304.07306v1](https://arxiv.org/abs/2304.07306v1) | 在专家预测昂贵且有限时，用表征与 expertise predictor 估计专家能力，再训练 defer 决策。 | “预测替代分支在样本上的能力”已有直接近邻；本项目拥有固定动作的配对 train-only 效用记录，监督可得性不同，但尚未证明独立方法贡献。 |
| [Geifman 与 El-Yaniv, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, arXiv:1901.09192v4](https://arxiv.org/abs/1901.09192v4) | 端到端学习预测和拒绝，分析 risk–coverage 权衡。 | 拒绝选项是“不作当前预测”，不等同于自动执行另一特征校正路径；但“安全地选择不操作”本身不足以构成本项目新颖性。 |

现有[六篇近邻复核](PUBLIC_PRIOR_ART_2026-09-12.md)中的 DeferredSeg 更贴近“根据各分支实际对错学习保留/转交”的语义分割场景；其最终输出依赖被选专家的预测，不能当作本项目免费获得的自动特征修复能力。上述四篇说明，若将来重新提出动作效用学习，必须比较 cost-sensitive/defer/reject 现成路线、专家/动作的执行成本和信息可得性，而非仅重新命名 `uT`、`u0` 或“不校正”分支。

## 对现有论文初稿的约束

1. 可写：本项目对**具体** stage-2 adapter 的整图选择器做了预声明、匹配常数动作与伤害约束的负结果；Phase 5–6 的标签知情诊断空间并没有在 Phase 7 转化为部署时增益。
2. 不可写：首次提出动作收益监督、可选择不校正、cost-sensitive gate、多专家/分支路由，或已证明所有局部修复机制无效。G2 `(uT,u0)` 与 G3 `(uT,uT−u0)` 在目标信息上可逆等价，不能以 G3 名称宣称额外信息。
3. 不可拿上述论文的跨任务分数与 LEVIR-CD F1 作直接公平比较；数据、专家来源、标签、拒绝成本及风险定义不同。现有 held-out test 已用于最终描述，不得据文献结果再调本项目模型。
4. 投稿前需核实正式出版版本、完整方法/实验与更广泛文献；本备忘录是针对性风险筛查，不是系统综述或新颖性证明。

来源边界：四项 arXiv 标题、作者、版本与摘要来自 2026-09-29 的 arXiv API `export.arxiv.org/api/query` 只读检索；Mozannar–Sontag HTML 引言来自 `arxiv.org/html/2006.01862`。未下载或改动研究数据及权重。
