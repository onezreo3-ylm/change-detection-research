# 论文证据链接续运行记录

2026-10-26，Asia/Shanghai。当前总目标仍active，论文尚未达到投稿就绪判定。
已完成证据矩阵与直接近邻补查，冻结缺失B0-E43/44控制及原预算边界。

工程准备：原train/开发val审计与旧B1一致；原核心11文件SHA256不变；两seed共200轮旧B1
source+instance流全部精确重建，并封存同序clean流。初版525项全量测试通过，随后CPU smoke
发现原runner即使FP32也需要禁用scaler对象；首次失败保留，修复仅涉及工程辅助函数及显式
准备快照修正支持。原准备/源码快照保留于source/，修正源码另存source_cpu_r1/；
cpu_smoke_correction.json绑定两版身份。修正后526项全量测试、CPU合成smoke均通过。
不把工程合成更新或pilot列为论文效果。

一次真实train-only pilot完成98次工程尝试/784pair views；seed43/44初始化匹配旧B1，
seed43独立32步状态及恢复后下一更新精确一致，实际clean采样流匹配。完整owner墙钟
52.204秒，最高62°C，worker实际退出0，owner记录正常收尾且进程已退出，guard/watchdog正常收尾。

冻结公式预测每行39238.548秒（约10.90小时，含6000秒验证/I/O余量及25%倍率），
两行加3600秒评价预留为82077.097秒，低于当时共享余额82403.173秒。
这是准入用保守估算，不能作最终实测计算开销。原172800秒共享上限及86400秒/行上限保持，
没有预算重置；剩余不足时停止并保留现场，不缩短轮数或换精度。

正式队列于2026-10-26约22:12:46（北京时间）首次启动，ownerPID26288、
workerPID21748，创建ticks分别134374975662027145/134374975675485631。
22:13:35快照确认seed43训练批间检查点已推进37次、GPU61°C/91%、四个绑定进程均active，
guard/watchdog健康。检查点数不当作已提交optimizer计数；后者以原history/status及完成核对为准。
seed44只在43正常完成并核对后首次启动，不依分数选择是否运行。

当前唯一轻量监控入口：

```powershell
& 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -X utf8 -m scripts.b0e_replication_supervision monitor --mode train
```

源代码与配置已封存，训练期间不编辑，不并发GPU，不重跑正常测试/pilot/全量hash/模型加载。
连续状态无变化无需重复叙述。新结果、实际失败或需要人类处理时报告。
旧levir-cd定时自动化此前已删除，本轮没有恢复或创建调度；当前持续研究由active目标接续，
训练安全由隐藏CPUowner及独立秒级保护承担，不依赖聊天轮询。

两行训练完成后先检查owner/worker真实退出、guard/watchdog状态、共享账本、原100轮
history与逐轮流、初始化、选模和新checkpoint元数据/哈希。保留原matched_robust_baseline_v1
flavor和源证据；单独封存本批外部监督完成证明，不能伪装成旧in-process守卫。
然后在原45条件、阈值与实例不变的前提下，准备并冻结新身份验证适配，仅执行各一次开发评分。
重放clean计数，完成三seed B1−B0-E全部条件/分组、预定保护门和源配对区间。
结果失败也完整报告，不选择性追加训练。还有模型覆盖、独立证据和近邻差异问题，
该控制闭环不等于整个论文总目标完成。

核心目录runs/b0e_replication_20261026；优先看evidence_index.json、training_authorization.json、
train_session/、seed43/status.json与seed44/status.json。旧test和Martigny未访问。

## 当前目标续接：分析准备

同日22:23快照核对实际PID创建身份仍有效：seed43已提交2轮，正推进第3轮，GPU77°C、
guard/watchdog正常。这些是中间进度，不报告方法收益。
新增独立scripts/analyze_b0e_replication.py及YAML，在看到任何新45条件评分前封存算法、
门、源配对bootstrap与历史七格身份。对七份历史结果的64源/45条件精确核对全部像素及标签支持，
不计算缺失两格的三seed主结论。九项CPU检查与保存结果重放smoke通过，日志
runs/b0e_replication_analysis_cpu_tests_20261026.log；历史验证与规则封存位于
runs/b0e_replication_analysis_20261026。新分析代码的全项目测试**仍待当前CUDA队列结束后运行**。
对应固定评价范围现见B0E_FIXED_VALIDATION_PROTOCOL_2026-10-26.md，评价适配实现尚未完成。
本轮CPU工作与正在计费的完整owner墙钟重叠，不并发修改活动账本，不将其重复计费为额外GPU计算。

## 同日22:46接续：固定评价适配已准备

独立评价/监督代码、YAML与11项CPU检查完成，原架构CPU synthetic smoke通过。
工程证据与执行runbook见B0E_FIXED_VALIDATION_ENGINEERING_2026-10-26.md，
原始记录在runs/b0e_fixed_validation_engineering_20261026。
新评价尚未启动，全项目测试等CUDA idle后实际运行，旧526项不代表新代码验收。
22:46快照确认同创建身份四角色仍active：seed43已提交7轮、正进入第8轮，GPU76°C，
本次训练最高78°C；无新失败和test访问。当前累计共享计费92429.108秒，原上限保持。
这些是进度与工程证据，不能作训练效果或论文完成判断。

## 同日23:28接续：工作稿与独立来源筛查

runs/manuscript_working_draft_20261026包含8655字符的中文工作稿、五张CSV/Markdown表、未舍入JSON、
一手引用阅读范围及逐项输入/产物hash。七历史结果和固定动作数值已生成并独立算术核对，
不计算缺失两格的均值或新置信区间，不将当前训练中间分数填入表格。
文稿澄清了actor356源/完整基线445源、事前B*常数选择规则、mF1与pooledF1及O不是F1上界等边界。
没有覆盖原文稿、发布外部材料或新增test访问。

S2Looking/SECOND的一手元数据审查见INDEPENDENT_DATA_METADATA_SCREEN_2026-10-26.md。
下一候选仅为SECOND发布说明和标签格式核查，尚无新数据准入；没有下载/读取原始影像或标签。
一次HEAD返回200 HTML而非ZIP，不能当数据已可用。全部CPU工作包含在重叠活动owner墙钟内。
23:28快照四个创建身份仍有效，seed43完成16轮、正推进第17轮，GPU76°C，最高78°C，原账本
累计94922.561秒。原队列、546项新全量测试待GPU空闲及固定45条件评价顺序保持。

## 2026-10-27 00:11接续：独立数据停止与图表登记

实际创建身份绑定的四角色仍active：seed43完成25/100轮、正在推进后续批，GPU77°C，
当前最高78°C；seed44仍按既定顺序等待43完整完成。共享累计97500.327秒，原172800秒上限保持。

SECOND外层ZIP目录成功取得，只有训练RAR、macOS元数据及test ZIP三项。原训练member大范围
传输与一次固定分段修正均失败；修正第4段返回明确quota-exceeded HTML，前三段partial保留。
两次累计11请求、159.360秒，未完成CRC/长度/EOF验证，未打开嵌套test、解压或解码影像/标签。
不再重复同一归档下载。结果与公开schema缺项见SECOND_QUARANTINE_OUTCOME_2026-10-27.md；
仍未准入，不能把部分下载算作外部数据或实验。历史WHU/HRSCD缺项继续保留，未重复下载。

新runs/publication_evidence_audit_20261027_r3包含两张可复现SVG/PNG图、未舍入CSV/JSON、
20项事后实验族登记及图注。完整九族与八策略保留，O明确为标签知情诊断；区间仅重用旧结果，
无新推理、重采样或B0-E缺失值填补。图1初版标注重叠已在r2修正，r3修复登记文件链接，
旧版本及失败保留；r2两PNG均实际查看。新CPU归档5项测试通过。
本批新代码的全项目测试仍pending至GPU空闲，546是评价验收最低覆盖要求而非当前总测试数。
两图渲染及既存计数算术核对不能替代独立预测复现或论文贡献证据；总目标active。
