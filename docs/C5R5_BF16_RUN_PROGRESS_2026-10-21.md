# C5/R5 40轮BF16正式执行进展

## 2026-10-22 固定评价闭环，托管已暂停

两行训练与各45条件val/固定64 train残余全部完成、保存结果审计通过，Task退出码0。
C5/R5 clean F1分别0.3711384319946564/0.32769892124259875；
45条件mf1分别0.3303648451287665/0.30037528310673806。
R5相对C5的clean F1、clean Recall、mf1、扰动Recall四项门失败，两项空tile FPR门通过。
残余筛查两模型四族均通过，但不能覆盖效果门，也没有准入新方法。
本批结论保留为当前单seed40轮BF16开发负面证据；不推及其他未验证协议的可行性。
完整报告docs/C5R5_FEASIBILITY_RESULTS_2026-10-22.md；
最终审计runs/valais_c5r5_fixed_eval_v1_20261022/final_closeout_audit_20261022.json。
没有已完整准入的下一实验，levir-cd已工具更新为PAUSED并读取实际配置确认；不追加训练/评分。
下方启动/运行/ACTIVE描述均为历史，当前状态以本段与MANAGED顶部为准。

## 2026-10-22 两行训练完成与固定评价启动记录（历史）

C5于10月21日19:52、R5于10月22日00:00正常结束（北京时间）。两行各110280成功更新、
441000源抽取、2205000 pair views；guard/hash/实际退出码0、完整实际采样和候选像素流通过。
最高温度均83°C。CPU监督接管后未重启C5，无本批研究步骤回滚重算。
实际worker合计8小时14分18秒；加原工程预检扣款合计8小时24分03秒，各24小时上限满足。
此数字不包含今天新评价适配的全量测试/102预测smoke及其后正式评价开销。

| 模型 | 最佳轮 | clean F1 | clean Recall | worker秒 | 含原预检秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C5 | 38 | 0.3711384319946564 | 0.3491724244021065 | 14780.172 | 15078.032 |
| R5 | 21 | 0.32769892124259875 | 0.33058838546661384 | 14877.469 | 15165.407 |

R5−C5 F1=-0.04343951075205765、Recall=-0.01858403893549264，均失败于原clean保护门。
尚无45条件扰动mf1/Recall/空tile结果，不作通过判断。普通五view增广当前不支持clean有效性。
本批为单seed40轮BF16训练，历史100轮FP32 B0-E仅背景，不混作匹配因果对照。

训练完成审计：runs/valais_c5r5_40ep_runtime_v1_20261020/training_commit_audit_20261022.json；
完整新适配审计：runs/valais_c5r5_fixed_eval_v1_20261022/training_completion_audit.json。
原owner失败历史保留，当前完成证明使用supervision_attach_20261021/status与exit记录。
421项全量测试及5477项准备输入核验已通过，原train首样本FP32 CUDA smoke102预测通过，
耗时26.875秒、最高46°C；两模型重复/交换、12条件四格/边际和参数/BN冻结验证通过。
协议docs/C5R5_FIXED_EVALUATION_PROTOCOL_2026-10-22.md；配置configs/valais_c5r5_fixed_eval_v1_20261022.yaml。
5481项输入已封存，10月22日12:35一次启动C5原45val→R5原45val→固定64 train残余，
各阶段6小时，无重试/追加训练。任务Codex-Valais-C5R5-FixedEval-20261022，
实际owner36104、首worker21776创建身份/等待有效，当前C5 clean复核中。
新轻量命令：python -m scripts.c5r5_fixed_evaluation monitor --config configs/valais_c5r5_fixed_eval_v1_20261022.yaml。
90分钟heartbeat已更新当前入口并恢复ACTIVE、实际配置核验通过；健康静默，仅重要证据/失败/完成通知。
新进度docs/C5R5_FIXED_EVALUATION_PROGRESS_2026-10-22.md；下方运行中/工具断连均为历史。

## 最新状态：同一C5 worker持续，无控制台CPU监督已接管

原owner18000返回0xC000013A退出，其Task现Disabled，原worker21216持续训练且已到第3轮。
无控制台CPU owner30532已接管现有worker，未回滚/重算/重启任何GPU步骤；R5首次启动仍在队列中。
完整事故与接管证据/4项CPU测试/真实进展smoke见supervision_attach_20261021及C5R5_OWNER_ATTACH_2026-10-21.md。
当前监测改为：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.c5r5_supervision_attach monitor --config configs/valais_c5r5_40ep_runtime_v1_20261020.yaml --attachment runs/valais_c5r5_40ep_runtime_v1_20261020/supervision_attach_20261021

自动化工具Transport closed，实际levir-cd仍PAUSED；独立队列继续训练。恢复工具连接后才可恢复90分钟heartbeat，
必须采用以上命令和当前独立监督状态，原root owner/status保留为失效历史。预算/原数值/温度/评价门全部保持。

2026-10-21，用户直接选择“采用BF16方案并启动（推荐）”。
实际选择已记录为runs/valais_c5r5_40ep_runtime_v1_20261020/human_precision_approval.json，
包含原提问/答复、日期、配置和协议hash。该选择明确接受新数值协议，不再等待精度确认。

## 当前入口与预算

配置configs/valais_c5r5_40ep_runtime_v1_20261020.yaml；协议docs/C5R5_RUNTIME_PROTOCOL_2026-10-20.md；
运行目录runs/valais_c5r5_40ep_runtime_v1_20261020；入口scripts.valais_c5r5_runtime。
新flavor matched_five_view_baseline_40ep_runtime_v1，optimization_mode=bf16_training_fp32_validation。
每行40轮、T_max40、110280成功更新、441000 source draws、2205000 pair views。
模型/初始化/seed42/五view及权重/联合BN/源与实例/损失/优化器/阈值保持；只有训练forward
autocast BF16，参数/BN/优化器/损失/clean val与全部后续评价FP32，无GradScaler和跳步。

407项全量测试及真实CUDA多batch/尾batch/精确断点复现已通过；输入封存38383项。
不重复正常测试或pilot，启动前重新核验现有封存SHA。预算估算C5 5.13699小时、R5 5.00979小时，
合计10.14678小时，含25%余量、每轮60秒及先前工程耗时；不是长期时间或模型效果保证。
各86400秒上限包括该行全部已用工程成本：C5 297.860秒，R5 287.938秒；
剩余正式worker墙钟分别86102.140秒和86112.062秒，输入核验/训练/val/保存均计入。
累计210次额外物理优化计算/4158 views全部丢弃，正式训练从原初始张量开始。
更早P1暂停重算与有限筛查成本另列，不把有效训练预算匹配写成所有物理开销一致。

## 启动状态

当前状态：已完成正式封存并于2026-10-21 15:45（北京时间）一次启动；owner PID18000与
C5 worker PID21216创建身份及等待/退出状态有效，首次输入核验中，正在等待实际batch进展。
新任务Codex-Valais-C5R5-40ep-BF16-20261021已注册，0个自动trigger、0次重试、隐藏、
禁止并发；旧40轮FP32任务保持Disabled，原100轮预算失败现场保留。
启动后C5完成guard/hash/实际流/预算审计才启动R5，不并发研究GPU任务；失败保留现场、不自动恢复。
守卫保持原85°C确认/86°C立即和传感器失败策略。人类暂停或改向优先。

轻量监测：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.valais_c5r5_runtime monitor --config configs/valais_c5r5_40ep_runtime_v1_20261020.yaml

PID创建身份及等待/退出状态、实际step/log/checkpoint时间和GPU共同判断进展；活PID不等于进步。
健康时90分钟核查静默，不重复模型加载/全量hash/测试/smoke/图像或历史，不重复启动。

## 完成后的固定评价

两行结束核验40轮、110280更新、2205000 views、完整source/实例/候选像素流、首次clean选模、
guard/status/hash及实际退出；然后新建显式新flavor/数值身份适配并测试封存。
依次C5原45条件val→R5原45条件val→原64个Sion train源/四族/等级1、3、5残余诊断。
各行val6小时、残余独立6小时，不续期，不覆盖已完成评分；保持原manifest/visual_review/
参数强度/阈值/FP32，val原85°C确认/86°C立即、残余>=85°C立即停止。
clean计数与五指标须精确复现best，逐条件/四格/边际/错误交换/预算须保存独立审计。
R5/C5原全部主要门保持；历史100轮FP32 B0-E仅不匹配背景，不据此宣称同预算因果效果。
不填未计算的通过门，不将单seed/富集train诊断称地域显著性、机制或算法创新。
旧P1负面证据保持，不追分续训，不自动准入M5/K5/W5/P2；Martigny与旧LEVIR test关闭，
无基础模型/几何/外部数据。闭环保存报告，无已准入下一实验时暂停本托管。
