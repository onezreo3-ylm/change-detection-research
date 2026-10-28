# 均匀采样C5/R5普通对照：有限执行协议

2026-10-20。用户在上一轮“下一步建立同预算C5/R5普通对照，M5暂缓”的完成报告后明确“继续”。
本批将对照具体化、作多batch工程准入，预算通过后串行执行两行正式普通基线及固定开发评价。
这是普通增强边界，不是M5/P2新方法准入，不声称创新。旧训练和筛查产物保持原样。

## 冻结比较

- C5：clean重复五个pair views；R5：clean、uu、uv、vu、vv。监督权重1/2及各1/8。
- 两行从同一个已验证seed42随机初始张量开始，均100轮、每轮11025个原Sion train源各一次、
  2756个满4源batch与一个尾源batch；各275700成功更新、5512500 pair views。
- 新的local Python Random按配置中的epoch种子洗牌完整train名字，不改变全局RNG。
  两行采用同一完整source流、实例元数据流和候选扰动像素流；C5实际输入仍全为clean。
  这不是与历史B0-E完全相同的顺序；历史双view B0-E只是背景参照。
- C5/R5单次模型forward容纳该source batch的全部五view，随后双时相一次concat。
  不拆microbatch、不冻结/替换BN、不减batch、不改AMP/设备/模型/初始化/阈值。
- 沿用原FP32 CUDA、batch4、AdamW 0.001/1e-4、原betas、clip1、100轮cosine至1e-6、
  原BCE+per-source soft Dice、mean/std、256 tile、标注解码、Sion train/Sierre val。
  逐epoch干净val并仅以首次最高clean F1选模；不据扰动分数选checkpoint。

## R5配对分布（明确的新训练协议）

复用原P0的brightness/contrast/gamma/color_bias四族、等级1/3/5及未改参数表，全部global。
每source/epoch由固定hash独立选族/等级，u/v按原P0函数各自有放回抽原direction表。
u/v相同的退化四格照常保留，不重抽、不依赖标签/分数，不增幅或新增扰动。
每个时相的四个候选图固定为Au/Av/Bu/Bv；D和O的单时相经验边际严格相同。
C5同样生成候选四图并记录hash，但不用它们训练，以核对相同候选输入流。
两行的actual-view像素流当然不同；不能宣称C5拥有R5同等扰动信息。

该训练分布不同于旧B1/A-S九族×五级的单边global/local增强；不把差异藏在原配置中。
原45条件val与原P0的64 train probe/四族×等级1/3/5评价终点全部保持，不能按训练族删评价条件。
原P0七张train视觉审核的范围与不足继续披露，没有新的细目标可辨认性承诺。

## 工程准入与预算

训练前：完整unittest、train/val身份与字节审计、每epoch100份source/实例计划封存。
两行各从原初始化作32步真实train计算，前8步为warmup；不用F1选配置。
另用下一真实batch作保存/重载的精确下一更新复现，比较loss、模型/优化器/RNG；
再测试真实尾batch的一源五view更新。每行35次物理优化计算全部丢弃，不充当正式训练。
首train源作无梯度FP32重复/交换自检；全程保持原85°C确认/86°C立即停止和传感器策略。
pilot每行最多1800秒，不循环重试温度/数值/OOM失败，不降标准。

用户已明确选择每行24小时（两行合计48小时），配置中固定86400秒/行，不得上调。
预算为每行pilot与正式worker累计墙钟上限，不代表连续满载GPU时间，pilot耗时先扣除。
pilot的稳态均值×275700×1.25，再加每epoch60秒的验证/保存余量和已用pilot时间，
必须不超过该行上限；否则停在工程预检，不削短100轮。
评价另各6小时、残余独立6小时。预算用尽停止并保留现场，不延长或追分。
只允许一个研究GPU worker，C5完成核验后才启动R5。每个阶段只启动一次，失败不自动恢复。

## 接续、身份与评价门

新checkpoint必须显式标记matched_five_view_baseline_v1，保存实际source/实例/像素流、
原数据/源码/配置身份、初始化、优化器/调度器及完整RNG；不能伪装旧flavor或放宽旧evaluator。
仅完整epoch提交last/best。未提交步骤及pilot为额外物理开销；用户暂停优先，恢复另留记录。
守卫、退出状态、PID创建身份、实际日志/进度及checkpoint时间共同判断状态；活PID不等于进步。

正式结果使用新的显式五view适配器，依原45条件val评分，再复用固定train残余诊断。
R5相对C5：扰动mF1至少+0.005，clean F1最多−0.005，clean/扰动Recall最多−0.01，
clean/扰动空tile FPR各最多+0.0002。全部保护同时满足才通过普通增强门。
C5相对旧B0-E为背景保护筛查：对应F1/mF1最多−0.005、Recall最多−0.01、空FPR各最多+0.0002。
该参照非同view/同source顺序的机制比较；若C5干净能力仍不足，不能只摘取R5相对弱C5的收益。
所有原始FP/FN及D/O较差组、方向反转和好组下降均报告。门不是显著性或创新证明。
单seed/单城开发、缺乏空间source IDs的局限保持；不作tile bootstrap地域显著性。
Martigny和旧LEVIR test持续关闭，不新增基础模型、几何变换或外部数据，不启动K5/M5/W5。

训练先执行，评价适配在新flavor源码/数据/采样/选模身份逐项核验并封存后执行；
不能让普通单步pilot替代方法准入。全部完成且没有已准入下一实验时保存报告并暂停托管。
旧C-S95、A-S97至少1100步及其他未知精确丢弃步骤、本次所有工程开销继续单列披露。
