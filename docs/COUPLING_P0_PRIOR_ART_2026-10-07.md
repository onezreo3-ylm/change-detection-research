# P0 定向近邻核查：扰动耦合诊断的准入边界

日期：2026-10-07（Asia/Shanghai）。本记录先读 `RESEARCH_SPEC.md` 与
`EXPERIMENT_REPLAN_2026-10-07.md`，仅检查公开论文/代码，不访问数据、训练或推理。

## 结论

目前阅读范围未发现“固定同一个 source 的四张退化图，严格保持单时相经验边际，
只交换对角/交叉配对，比较真实 FP/FN”的同一实验。**这允许开展低成本 P0 诊断，
不构成排重完成、首创或方法创新证明。** BGFD 代码身份/版本、Hessian 论文 HTML
仍有未完成项；这些阻止宣称完整文献复现，不阻止不训练新算法的可证伪诊断。

S2C 已包含原/增强的四组图与跨时相对比，不能把“四张图”当创新；BGFD 已针对双时相
域差引入统计特征噪声与依赖约束；FAF-CD 已有明确单边伪变化压力测试；Hessian Penalty
已有抑制混合二阶响应的通用方法。P0 仅提出更严格的配对对照，不把混合响应当因果损伤。

## 检索与证据范围

通用 web 工具返回网关 404，随后用 PowerShell `Invoke-WebRequest` 只读访问 arXiv
HTML、作者链接的 GitHub 页面及 raw 源码。GitHub contents API 返回 403 rate limit，
改读公开页面。未运行任何外部代码、安装依赖、下载权重或影像。正文长输出曾截断，
随后分别重取完整方法与实现段；下列阅读范围以未截断的定向读取为准。

论文固定为所读 v1；GitHub 使用当日 main/master 分支，**未锁 commit，不能充当
可执行复现快照**。本次不是全面最新文献搜索，也未核验官方出版目录中的录用状态。

## 方法比较

| 工作 | 输入与扰动联合分布 | 目标 | 推理信息 | 训练/计算预算与本项目差别 |
|---|---|---|---|---|
| S2C | 无标签双时相及各自增强图；谱适配使用另一时相作参考，另有 RGB、降采样及空间偏移；不是固定边际的四格交换实验 | 双向 triplet 时间对比、跨区域 InfoNCE 空间对比、grid sparsity | 原图双时相；余弦变化图，可用 VFM masks/IoU 精化 | 论文20轮、512 crop、SGD；四组特征，不等于本项目五 pair views；预训练/LoRA及可选后处理不同 |
| BGFD | ResNet-50/CLIP；逐样本、逐时相、逐尺度均值/方差生成高斯特征噪声，权重随迭代增长 | 任务检测、噪声前后双时相 MI 不下降约束、特征细节/注意力处理 | 双时相及网络特征；所读正文不足以确认推理是否继续随机噪声及其 seed 规则 | 论文60k迭代、batch22、AdamW；256 patch/128 stride；MI只算最小尺度前128通道；非匹配本项目预算 |
| FAF-CD | 双时相；本文架构为 DINOv3 ConvNeXt+空间/Fourier/Haar三分支；补充评测分别只扰动A或B | 像素分类交叉熵，融合门控；不是四格正则 | 双时相、预训练特征；评测扰动元信息不作为目标网络输入的必要条件 | 论文150轮，LEVIR batch8/5090约13.47h；大预训练网络，不是轻基线公平替代 |
| Hessian Penalty | 通用G(z)，对输入加Rademacher方向正/负位移；不是双时相配对任务 | 方向二阶有限差分的方差，鼓励输入 Hessian 非对角元素减小 | 正则不要求推理时额外扰动输入；具体网络另定 | 官方PyTorch默认k=2、epsilon=0.1；需要2k+1次G调用（已有G(z)可复用），不能与四格i²称完全等价 |

上述算法概括分别依据：[S2C §III–IV-B](https://arxiv.org/html/2502.12604v1)、
[BGFD §III–IV-B](https://arxiv.org/html/2410.09539v1)、
[FAF-CD §3–4与§6](https://arxiv.org/html/2606.03114v1)、
[Hessian 官方实现](https://github.com/wpeebles/hessian_penalty/blob/master/hessian_penalty_pytorch.py)。

### S2C：代码确认及协议差异

已完整阅读作者论文链接仓库的
[readme](https://github.com/DingLei14/S2C/blob/main/readme.md)、
[入口](https://github.com/DingLei14/S2C/blob/main/train.py)、
[engine](https://github.com/DingLei14/S2C/blob/main/s2c/engine.py)、
[CLCD增强](https://github.com/DingLei14/S2C/blob/main/datasets/CLCD_aug.py)。
未读全体模型与loss源码，不声称完整适配或复现。

代码先共同 weak_aug，然后分别调用 `strong_aug(A,B)` 和 `strong_aug(B,A)`；谱分布适配
依赖 partner，不能称增强与双时相完全独立。训练读取 A/B/A_aug/B_aug，分别编码，计算
triplet、双向交叉 InfoNCE，并另调用 bi_forward 算稀疏项。没有看到本项目D/O四格
经验边际的同源交换评测；这只限定到已读文件，不是全仓库不存在证明。

重要：engine 无条件构建 train/val/test，并在训练结束自动 test；其 val 选模是逐样本
F1平均，test另用pooled计数。设备函数允许回落CPU。**不得原样运行或移植这些协议。**
后续若适配只能另立 train/val 隔离版本，使用本项目固定指标/device/初始化/预算。

### BGFD：全文方法读到，公开仓库归属未认证

正文 §III-A–D 完整覆盖 DFC、GNDD、FDF；算法是分别从各时相统计分布采样，论文
未明确共享随机数/相关噪声场的联合实现，因此不能把“统计来自双时相”说成“共享扰动耦合”。
MI约束保留的是双时相信息依赖，仍不是边际严格相同的像素扰动配对控制。

arXiv 摘要/HTML未提供代码链接。公开 GitHub 搜索找到
[Kevinorve/BGFD](https://github.com/Kevinorve/BGFD)，完整读到README和README_BGFD，
但两者将BGFD展开为“Building-aware Global Feature Distillation”，与论文题名不同；
README引用的S2Looking配置raw链接返回404。**该仓库只能记为待认证候选，未读到
可信GNDD核心实现，不能写成作者代码已复现。** 需要作者来源关联、固定版本及真实模块
逐项核验，特别是MI估计可微性、随机数联合规则和eval路径。

### FAF-CD：方法与发布配置可读，不能照搬测试路径

完整阅读论文 §3.1–3.5、§4.2、§6.1–6.3；单边测试明确涵盖亮度/对比度、color jitter、
haze、shadow，等级1–5、A/B分别作用，不包含both-image/noise/blur的二值CD汇总。
因此单边压力测试不是本项目可主张的新意；D/O匹配边际是另一个诊断问题。

完整读到作者链接仓库
[README](https://github.com/VimsLab/FAF-CD/blob/main/README.md)、
[builder](https://github.com/VimsLab/FAF-CD/blob/main/models/builder.py)、
[LEVIR配置](https://github.com/VimsLab/FAF-CD/blob/main/configs/faf_cd/config_levir_dinov3_convnext_large.py)。
未读完整encoder/fusion/perturbation源码，不声称核验扰动参数或复现指标。
发布配置CrossE类权重为[1,1]，不能凭正文“weighted CE”假定特殊正类加权。
配置含 `enable_val_test_dual_tracking=True`，README区分`best_test`/`best_val`权重及
legacy-compatible评测；仅凭命名不能判定具体选模实施，但足以禁止照搬到本项目。

### Hessian：通用二阶重合明确，但不是同一估计量

arXiv HTML返回404；ar5iv返回转换失败，因此**未读到完整论文正文**。已完整阅读作者
[README](https://github.com/wpeebles/hessian_penalty/blob/master/README.md)、
[PyTorch实现](https://github.com/wpeebles/hessian_penalty/blob/master/hessian_penalty_pytorch.py)、
[tips](https://github.com/wpeebles/hessian_penalty/blob/master/tips_and_tricks.md)。
代码对每个方向计算 `(G(z+x)-2G(z)+G(z-x))/epsilon²`，沿方向取无偏样本方差，
默认跨输出max归约；本计划的四格混合响应i与mean(i²)不与它逐项同一。

本项目在光滑小扰动极限下仍属于混合二阶响应约束范畴，不能声称发明新的二阶思想。
作者明确要求有限差分前向的随机性及移动统计固定；P0 eval、重复输出和时相交换自检
因而是必要工程条件。P2若要训练必须重新审查BN与多view计算图；正则下降本身不证明
任务或解耦改善。尚须读论文正文和其他交互正则近邻后才可完成M5方法准入。

## 对本批执行的约束

1. P0仅复用冻结checkpoint，在train样本做D/O匹配边际诊断，实际FP/FN为主，logit分解为辅。
2. 不纳入基础模型、空间几何扰动或外部代码训练入口；不访问val/test筛扰动与配对。
3. 不把四图、普通一致性、最坏组风险或有限差分重命名为新算法。
4. 若后续残余问题成立，方法阶段必须补齐公开代码版本与最近邻适配；P0通过不替代此门。
5. 本记录只新增文档，未改变代码，因此未运行训练/单元测试/smoke；本批代码实现的测试
   由执行主线单独记录。本记录没有科学实验结果。
