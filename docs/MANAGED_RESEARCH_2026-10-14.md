# 当前托管接续

## 2026-10-22 C5/R5固定评价已闭环，托管已暂停（当前优先状态）

两行40轮训练、各45条件val及原64个Sion train残余均完成，13:30:45队列正常结束（北京时间）。
14:11最终保存结果审计通过：两行clean计数/五指标精确复现、45条件来源/分层/边际守恒、
guard/hash/源码与全部退出码0、六指标冻结门重算一致；残余6272主预测/6276总预测/1536记录通过。
C5 clean F1=0.3711384319946564、45条件mf1=0.3303648451287665；
R5 clean F1=0.32769892124259875、mf1=0.30037528310673806。
R5相对C5分别下降0.04343951075205765/0.02998956202202846，clean和扰动Recall亦超过下降限值；
主要六项门前四项失败、两项空tile FPR保护通过。C5相对历史B0-E亦前四失败/两空tile通过，
但B0-E为100轮FP32不匹配背景，不据此归因BF16/轮数/五view。
两行残余筛查均四族通过：D−O F1为C5 0.03013008101339376、R5 0.038775873052178056，
R5 D/O F1均更低；全像素交互幅度变小同时前景幅度变大，不能宣称机制修复或创新。
没有已完整准入的下一实验，按原协议已用工具将levir-cd设为PAUSED，并读取实际配置核验。
不自动推进M5/K5/W5/P2，不追加训练或重跑完成评分；Martigny及旧LEVIR test保持关闭。
完整结论docs/C5R5_FEASIBILITY_RESULTS_2026-10-22.md；
审计runs/valais_c5r5_fixed_eval_v1_20261022/final_closeout_audit_20261022.json；
分数/原门runs/valais_c5r5_fixed_eval_v1_20261022/comparison.json。
原报告、checkpoint、协议与所有额外物理开销记录保留。本次收尾没有新推理/训练或重跑正常测试。
下方训练/评价运行中及ACTIVE/工具断连等段落均为历史，由本段覆盖当前状态。

## 2026-10-22 C5/R5训练完成，固定评价队列启动记录（历史）

两行40轮均已正常退出，各110280成功更新、441000源抽取、2205000 pair views；
最高温度均83°C。C5于10月21日19:52完成、R5于10月22日00:00完成（北京时间）。
实际worker训练合计8.2382小时，计入原已扣工程预检后8.4010小时；均低于各24小时上限。
C5最佳epoch38 clean F1=0.3711384319946564、Recall=0.3491724244021065；
R5最佳epoch21 clean F1=0.32769892124259875、Recall=0.33058838546661384。
R5相对C5 clean F1下降0.04343951075205765，Recall下降0.01858403893549264，
两个clean保护门已失败，不能以尚未计算的扰动结果覆盖。无追加训练或P2准入。
完整提交/guard/hash/实际退出及逐轮候选像素流匹配已通过；原root失败owner/status保留。
当前完成依据为supervision_attach_20261021/status.json及两行exit记录，而非旧失效owner。
训练证据runs/valais_c5r5_40ep_runtime_v1_20261020/training_commit_audit_20261022.json。

新固定评价协议docs/C5R5_FIXED_EVALUATION_PROTOCOL_2026-10-22.md；
配置configs/valais_c5r5_fixed_eval_v1_20261022.yaml，目录runs/valais_c5r5_fixed_eval_v1_20261022。
显式新适配器严格验证实际BF16训练flavor/FP32评价身份，不改checkpoint/旧评价检查。
421项全量测试（含真实CUDA）通过，5477项准备输入及完整采样/模型数据身份核验通过。
原首train样本两行FP32 CUDA重复/交换/12条件四格/边际/状态冻结smoke已通过：
102预测、26.875秒、最高46°C，交换误差均0，未发生训练/val/test研究评分。
5481项输入封存完成；10月22日12:35（北京时间）一次启动固定评价队列，
仅串行C5原45val→R5原45val→原64 Sion train残余，每阶段6小时。
评价FP32 CUDA/0.5，val85°C确认/86°C立即、残余>=85°C立即；失败保留不重试、不覆盖。
无heldout test/Martigny，无新训练/基础模型/几何/外部数据。
新隐藏无控制台任务Codex-Valais-C5R5-FixedEval-20261022，无自动trigger/重试、禁止并发。
owner PID36104/创建ticks134371173541607643，首C5 worker PID21776/创建ticks134371173675935992，
两者等待/退出状态均有效；12:37:59已精确复现C5 clean计数/五指标且实际完成3/45条件，GPU55°C。
后续条件进展以monitor为准，不能只凭存活进程判断健康。
轻量命令：E:/Download/anaconda3/envs/geosa-basa/python.exe -m scripts.c5r5_fixed_evaluation monitor --config configs/valais_c5r5_fixed_eval_v1_20261022.yaml。
90分钟heartbeat levir-cd已用工具更新为当前入口并ACTIVE，实际automation.toml已验证；健康静默。
不重跑正常测试/smoke/全量hash/模型加载，PID存在不等于实际进步。新进度见docs/C5R5_FIXED_EVALUATION_PROGRESS_2026-10-22.md。
下方10月21日“正在训练”“工具连接关闭”等为历史，不代表当前；本批闭环无已准入下一实验则暂停托管。

## 2026-10-21 无控制台CPU监督已接管，C5持续训练（当前优先状态）

原BF16队列owner18000在C5开始后以0xC000013A退出，原Task已禁用；原因发送方未明，
证据保存在runs/valais_c5r5_40ep_runtime_v1_20261020/supervision_attach_20261021。
原C5 worker PID21216/创建ticks134370423605411398持续运行，未重启/回滚/重算。
CPU监督已直接使用同环境pythonw.exe无控制台接管，owner PID30532，任务
Codex-Valais-C5R5-CPU-Attach-20261021，0触发器/0自动重试/禁止并发；一次CPU接管已使用。
原407项训练全量测试对应的训练源码/配置/authorization未改变；新增4项CPU监督测试和
实际worker身份/等待/真实step推进smoke通过，未并发重跑3个真实CUDA测试。
当前实际C5进展已进入第3轮，并有完整epoch2 last.pt提交；精确最新值以新monitor为准。
R5仍由新CPU监督在C5真实退出码0与原check_completion全部通过后首次串行启动，不再启动C5。
旧root owner/status保存失败历史，当前队列状态在supervision_attach_20261021/status.json，不能把旧owner存在当健康。
当前轻量命令：E:/Download/anaconda3/envs/geosa-basa/python.exe -m scripts.c5r5_supervision_attach monitor --config configs/valais_c5r5_40ep_runtime_v1_20261020.yaml --attachment runs/valais_c5r5_40ep_runtime_v1_20261020/supervision_attach_20261021。
必读docs/C5R5_OWNER_ATTACH_2026-10-21.md及docs/C5R5_BF16_RUN_PROGRESS_2026-10-21.md；
原训练dtype/40轮/预算/guard/完整流和后续45val/固定64train残余协议全部保持。
自动化工具连续返回Transport closed，现有levir-cd实际仍PAUSED，旧P1 prompt未更新。
不能宣称90分钟托管已恢复；连接恢复后必须先更新当前命令/CPU监督接管/数值身份/预算及安静通知合同，再ACTIVE。
当前prompt草稿保存为监督目录heartbeat_prompt.txt，不能手写修改automation.toml或创建cron替代heartbeat。
独立训练队列和温度守卫继续运行；仍只允许一个GPU worker和一次当前CPU监督，后续失败保留现场不循环接管。
人类暂停优先；停止时同时核对当前CPU owner与当前GPU worker身份，不能只处理失效旧owner。
下方“90分钟已恢复”等启动过程表述已被本段实际PAUSED状态更正，保留为历史。

## 2026-10-21 BF16 40轮C5/R5已获选择并启动（当前唯一入口）

用户直接答复“采用BF16方案并启动（推荐）”。human_precision_approval.json与新authorization已保存，
407项测试/旧真实CUDA pilot不重跑，seal重新核验38383项准备输入后通过。
当前配置configs/valais_c5r5_40ep_runtime_v1_20261020.yaml；协议docs/C5R5_RUNTIME_PROTOCOL_2026-10-20.md；
运行目录runs/valais_c5r5_40ep_runtime_v1_20261020；进展docs/C5R5_BF16_RUN_PROGRESS_2026-10-21.md。
本次一次启动额度已于2026-10-21 15:45（北京时间）使用，新任务Codex-Valais-C5R5-40ep-BF16-20261021。
旧100轮预算失败现场保留，原40轮全FP32任务Disabled；不得启动这些旧入口。
当前owner PID18000、C5 worker PID21216，创建身份/等待/退出状态有效，正在首次输入核验。
训练仅forward BF16；参数/BN/优化器/损失/所有val与评价FP32，40轮/T_max40、五view/权重/单次BN、
seed42原初始化及所有source/实例/候选像素流保持。各110280成功更新/2205000 views/441000源抽取。
各24小时硬上限扣C5 297.860秒/R5 287.938秒工程成本；预计总10.147小时，不保证长期吞吐或效果。
C5完成审计后R5串行；失败/预算/温度/数值异常保留现场，不自动重试或恢复。人类暂停优先。
90分钟托管levir-cd已按当前入口恢复，健康时静默，仅读紧凑monitor快照，不重复加载模型/测试/pilot/全量hash。
轻量命令：E:/Download/anaconda3/envs/geosa-basa/python.exe -m scripts.valais_c5r5_runtime monitor --config configs/valais_c5r5_40ep_runtime_v1_20261020.yaml。
不要只看PID；核对创建身份、等待/退出、实际step/日志/checkpoint时间。当前初始化尚待实际batch进展证明。
完成后严格核验两行40轮预算/完整实际流/首次clean选模/guard/hash/退出，再封存显式新flavor与数值身份
评价适配，C5原45val→R5原45val→原64 Sion train四族/等级1、3、5残余，各val6小时/残余6小时。
原全部门/强度/manifest/视觉审核/阈值保持；历史100轮FP32 B0-E仅不匹配背景，不能称因果机制或创新。
Martigny/旧LEVIR test关闭，无M5/K5/W5/P2自动准入、基础模型、几何或外部数据；闭环无已准入实验则暂停托管。
下方2026-10-20等待选择、旧FP32准备及P1记录为历史，不覆盖本次直接人类授权。

## 2026-10-20 缩时预检完成，等待数值精度选择（当前优先状态）

用户要求“尝试优化一下方案缩短一下时间”。正式训练从未启动：原100轮预算失败、
原40轮FP32封存保留；原40轮Scheduled Task已禁用，无trigger/重试。托管仍PAUSED。
纯工程优化实测仅+1.057%/+1.430%，虽逐位等价但未达3%门，不采用。
BF16真实32步/行、精确下一更新及尾batch均通过；参数/BN/优化器/损失/验证FP32。
本机真实BF16支持，C5/R5每步0.11458/0.11133秒，含25%余量/验证/累计工程扣款，
各预计5.137/5.010小时，合计10.147小时；各24小时上限仍保持。不是长期稳定或效果承诺。
407项完整测试通过；新独立40轮数值预案38383项输入封存完成。
报告docs/C5R5_SPEED_OPTIMIZATION_RESULTS_2026-10-20.md；协议docs/C5R5_RUNTIME_PROTOCOL_2026-10-20.md；
配置configs/valais_c5r5_40ep_runtime_v1_20261020.yaml；入口scripts.valais_c5r5_runtime；
目录runs/valais_c5r5_40ep_runtime_v1_20261020。该目录当前只有prepared输入，
没有human_precision_approval/authorization/owner，不能把原“优化一下”改写为已接受BF16长训练。
已向人类提供“采用BF16方案并启动”或“保持全FP32”选择；等待直接选择后才能封存启动。
若选择BF16，记录直接人类答复到human_precision_approval.json再seal，不再重复完整GPU pilot。
若选择全FP32，需将新独立配置改为reference并重新准备身份，保留当前BF16预案，不复用旧40行24h账。
所有原C5/R5工程累计210次物理更新/4158 views，扣款C5 297.860秒/R5 287.938秒，
旧P1暂停重算与有限筛查仍另列；正式研究更新0。不恢复已完成P1、不开test、不准入新方法。
后续C5→R5串行，各40轮/110280成功更新/2205000 views，再显式新flavor适配原45val及固定64train残余。

## 2026-10-20 独立40轮C5/R5（当前批次，优先于下方历史）

用户批准“改为40轮独立对照”，保留“两行合计48小时、每行24小时”。
原100轮工程通过但预测56.08/56.41小时，正式训练0更新，预算失败现场完整保留；
详见docs/C5R5_100EP_BUDGET_STOP_2026-10-20.md，不能调用旧100轮入口训练。
新配置configs/valais_c5r5_40ep_seed42_v1_20261020.yaml，协议docs/C5R5_40EP_PROTOCOL_2026-10-20.md，
新入口scripts.valais_c5r5_40ep，新目录runs/valais_c5r5_40ep_seed42_v1_20261020。
40轮与cosine T_max40同步，原seed42重新初始化，原模型/FP32/CUDA/batch4/单次BN/阈值和
五view权重/实例保持，110280有效更新、441000 source draws、2205000 pair views/行。
当前40轮397项测试及真实CUDA smoke/封存通过，正式训练尚未启动。
用户随后要求优化缩时，现执行独立16步/方案/行的限时性能对照（401项测试通过），
配置configs/valais_c5r5_speed_pilot_v1_20261020.yaml；结果未定，不重复启动旧队列。
原pilot耗时加本批smoke继续从各86400秒扣除，预算到期停、失败不重试。
两行随后串行C5→R5，原45条件val和固定64个Sion train残余另经显式40ep flavor适配后执行。
旧B0-E 100轮仅不匹配背景；R5/C5主要门和原背景保护门保持，非新方法或创新准入。
Martigny/旧LEVIR test关闭，无新模型、几何扰动或外部数据。进展详见新40轮协议与运行目录。

## 2026-10-20 C5/R5普通对照准备（当前人工批次）

用户明确“继续”，并选择“两行合计最多48小时、每行24小时”。
协议docs/C5R5_ORDINARY_CONTROL_PROTOCOL_2026-10-20.md；配置
configs/valais_c5r5_study_v1_20261020.yaml；目录runs/valais_c5r5_study_v1_20261020。
保留100轮、原模型/seed42/FP32/CUDA/阈值/损失等，两行同均匀source与候选实例/像素流，
C5五次clean、R5 clean+四格，监督权重1/2及各1/8，单次联合BN，不是新方法准入。
正在全量测试/完整数据审计/封存采样计划，随后各32步多batch CUDA吞吐、精确断点下一更新
复现和真实一源尾batch检查。全部pilot仅工程用途且先从该行24小时上限扣除。
按冻结25%余量和每epoch60秒验证/保存余量预测超过24小时，则两行正式训练不启动，
不缩短100轮、不加预算、不改batch/精度/BN。托管目前仍PAUSED；待实际准入结果更新此处。

## 2026-10-20 用户授权有限筛查已完成（当前状态）

用户在有限可行性筛查建议后明确“好，开始执行”。独立协议：
docs/FEASIBILITY_SCREEN_PROTOCOL_2026-10-20.md；配置：
configs/valais_feasibility_screen_v1_20261020.yaml；目录：
runs/valais_feasibility_screen_v1_20261020。
本批保存结果归因、三模型train-only BN统计干预、一个真实batch的R5工程更新；
不是恢复已完成100轮训练，不准入正式P2，不重开test或新val推理，不恢复90分钟托管。
最终377项全量测试通过；三模型真实CUDA smoke、1137项原封存输入及独立保存审计通过。
BN诊断完成576条probe记录、594个probe预测、3072校准pair views；参数/非BN缓冲不变、
原预测逐源精确复现、错误交换守恒及原模型恢复通过。最高62°C，211.328秒。
自然BN校准使C-S/A-S的固定train probe F1分别下降1.7909/0.8445个百分点，不支持直接修复。
保存val归因：C-S净增FN 22747，>4096前景像素层的14个tile贡献净增22760；不能说是14栋建筑。
R5五view真实工程更新通过：4 source、20 pair views、一次联合BN及AdamW更新，FP32、
seed42初始hash保持，峰值PyTorch allocated约2.36GiB。工程可运行，不是效果提升证据。
第一次pilot在显存API查询因缺少明确device index失败，0 forward/0更新；旧脚本/失败产物
均保留，新独立入口scripts.valais_r5_engineering_pilot另行封存修正，只消费原剩余一次实际更新。
两次pilot guard累计3.797秒；具体说明docs/R5_PILOT_CUDA_INDEX_CORRECTION_2026-10-20.md。
最终审计runs/valais_feasibility_screen_v1_20261020/final_audit.json由新入口生成；不要再执行旧pilot/audit。
本批结论和下一普通对照建议见docs/FEASIBILITY_SCREEN_RESULTS_2026-10-20.md；
完成收据为新目录closeout.json。所有本批研究进程已退出，90分钟托管保持PAUSED。
正式C5/R5效果协议、机制/近邻门仍未完成；未启动多轮训练或M5，不以单步pilot准入正式P2。

## 2026-10-20 P1本批完成（当前状态，优先于下方执行历史）

C-S/A-S各100轮、原45条件val各45/45及固定train残余诊断全部完成。
队列于2026-10-19 23:39结束；owner/worker均已退出，无当前研究GPU进程。
最终重新核对173个封存输入、各275700有效更新/2205000 pair views、两行实际source流，
clean计数/五指标精确复现、val保存计数/guard/hash和残余1536记录/6276预测审计均通过。

结果：原B0-E/C-S/A-S的clean F1为0.4205899068/0.3740468676/0.3623707017；
扰动mF1为0.3755254825/0.3363859171/0.3415488875。C-S普通采样改善门失败；
A-S虽相对C-S mF1+0.0051629703，但clean F1和clean空tile FPR保护失败，综合门仍失败。
残余门通过：两行各三个族有绝对D/O差>=0.005及非零有害交换；A-S总体差仅0.0034889823，
方向反转和抵消存在，差距缩小主要伴随D组变差，不能当作方法成功或因果机制证明。

完整报告：docs/P1_FEASIBILITY_RESULTS_2026-10-20.md。
完整证据：runs/valais_p1_posttraining_v1_20261019/comparison.json、final_completion_audit.json，
以及residual/final_saved_result_audit.json。保存结果审计未重新执行模型。
训练/评分原预算已经完成，不再次续训或重跑已完成评分。P2没有已准入的下一实验，
按原合同暂停每90分钟托管；未来人类新指令需从新的证据门接续，不能复活已完成预算。
原pcdc/模型/配置/旧成果/实例/QA保持；test未重开；额外物理重算披露继续有效。

## 2026-10-19 22:51固定评价接续（当前入口，训练已结束）

C-S/A-S均完成100轮，最后A-S于21:51完成；两个训练进程及owner均已退出。
各275700有效成功更新、2205000 pair views和guard/status哈希通过；按原train源名与
标签分区重新生成100轮采样计划，与封存计划及两行实际source流完全一致。
C-S选中36轮clean F1=0.3740468676；A-S选中17轮clean F1=0.3623707017。
原C-S clean改善门失败，A-S相对C-S的clean F1下降0.0116761659，亦不满足<=0.005保护门。
其余扰动/空tile门待评分，不把未算项写成通过或由单seed结果宣称创新。

当前配置：configs/valais_p1_posttraining_v1_20261019.yaml。
当前状态：runs/valais_p1_posttraining_v1_20261019/status.json。
协议：docs/P1_POSTTRAINING_PROTOCOL_2026-10-19.md。
当前轻量核查命令：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.valais_p1_posttraining monitor --config configs/valais_p1_posttraining_v1_20261019.yaml

新独立适配器明确验证stratified flavor、训练/采样/源码/初始化/数据/曝光与首次clean选模，
原pcdc、trainer、旧evaluator及checkpoint未修改。364项全量测试通过；两模型在同一原P0
首train样本的FP32 CUDA重复/交换、12条件四格及边际检查通过，模型参数/缓冲逐位不变。
适配预检确认两行base配置与冻结benchmark完全相等；评价输入173个文件已封存。

GPU独立队列串行C-S原45条件val→A-S原45条件val→原64个train样本残余诊断。
Val clean计数/指标须精确复现各best，完整条件/源/strata计数审计和guard/hash后才接受汇总。
残余直接引用未变P0 manifest与visual_review，不创建新视觉验收；两模型6272主预测、
6276总预测、1536条记录，逐源/四格/错误交换独立重算，含最终guard预算审计。
各val行6小时上限、残余诊断独立6小时上限；原val 85°C确认/86°C立即和P0 >=85°C停止规则保持。
启动额度已用，不自动重试或覆写已评分行。正常时只读紧凑进展，无变化静默，不重复测试/
smoke/全量哈希/模型加载/图片。任何新暂停指令优先，须停该新队列owner及当前worker。

结束后核验两行saved_result_audit及residual/final_saved_result_audit、comparison.json，
按原门报告普通采样/增强的负面结果与残余现象的局限；完成本批且无已准入下一实验时
保存总结并暂停90分钟托管。P2不自动启动，仍需原文献/机制/计算pilot/有限预算与强对照门。
test保持关闭。训练物理额外计算披露继续有效，不把有效预算匹配写成总物理开销匹配。

## 2026-10-19 21:06再次恢复（当前入口，优先于下方暂停历史）

用户新指令：“恢复并继续”，解除19:13暂停。已核验上次A-S封存、当前96轮断点及
暂停收据中6个进程退出；96轮history/实际采样/预算/选模/模型优化器有限值/RNG均通过。
复用未改动的scripts.resume_valais_p1_as及原trainer，独立封存当前A-S恢复链与guard。
352项完整unittest通过；两次96轮真实保存状态CUDA下一更新的模型/优化器/loss/RNG
完全相同，且首batch loss按6位打印精度复现原97轮日志的0.015690。

当前配置：configs/valais_p1_resume_as_epoch96_20261019.yaml。
当前状态：runs/valais_p1_resume_as_epoch96_20261019/status.json。
协议：docs/P1_RESUME_AS_EPOCH96_2026-10-19.md。
新轻量核查命令：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.resume_valais_p1_as monitor --config configs/valais_p1_resume_as_epoch96_20261019.yaml

仅续训97–100轮，剩余11028成功更新/88200 pair views，按上一轮耗时约44分钟，
不含启动数据复核、I/O或后续评价；本次一次启动额度已使用，不自动重试。
新guard会话为A-S temperature_guard_resumes/session_0002，原配置与85/86°C规则不变。
前97轮未提交部分日志确认至少1100步，精确废弃步数未知；与旧C-S95轮重算、恢复smoke
一并单列额外物理计算，不能宣称总物理曝光/耗时完全匹配。

恢复每90分钟托管核查，正常时仅读紧凑快照并保持静默，不重复测试/smoke/模型加载/
全量哈希/图片；有效owner或worker存活不启动其他GPU任务。两行结束后核对各100轮、
275700更新/2205000 pair views、实际采样及guard/hash，再显式适配stratified flavor的
原45条件val，继续固定64个Sion train样本残余耦合诊断。不可改checkpoint flavor或放宽
身份/数据/源码/选模/阈值检查。C-S clean改善门失败的现有证据保持，A-S结果未最终完成。
P2依旧须原冻结证据门，test保持关闭；人类新的暂停或改向优先。

## 2026-10-19 19:13用户暂停（当前最高优先）

用户明确指令：“暂停”。已暂停当前chat的90分钟托管核查，并按PID/创建身份/可执行文件/
完整命令行核验后，停止恢复owner、A-S训练worker和4个数据worker；6个进程均确认退出。
C-S保持100轮完成；A-S完整checkpoint保存到96/100轮，第97轮未提交部分不计入有效预算。
best/last、96轮history/sampling/预算、优化器/RNG及旧日志/guard均保留；独立复制核验哈希。
已只读确认last.pt内epoch96、完整96轮history、best epoch17及完整RNG键。
本次为用户主动暂停，不补记training/guard完成，不自动重启、评分或探索新实验。
等待人类新的恢复指令；下方恢复入口和计划均为暂停前历史。
收据：runs/valais_p1_user_pause_20261019T111311Z/pause_receipt.json。

## 2026-10-19用户恢复（当前入口，优先于下方暂停历史）

用户明确要求“恢复实验，并继续进行可行性验证，并开始执行训练任务”。解除10月16日暂停，
本次仅接续A-S第74–100轮；C-S已完成100轮并通过预算、采样及guard/hash核验。
两级旧封存链未变；原owner/worker均已退出。A-S完整73轮状态已复制并逐文件验证哈希。
352项全量unittest通过，两次真实A-S断点CUDA下一更新的模型/优化器/loss/RNG完全一致。

当前配置：configs/valais_p1_resume_as_epoch73_20261019.yaml。
当前状态：runs/valais_p1_resume_as_epoch73_20261019/status.json。
新协议：docs/P1_RESUME_AS_EPOCH73_2026-10-19.md。
使用原trainer --resume，模型/FP32/CUDA/初始化/采样/扰动规则/超参数/阈值均不变。
本次一次启动额度已使用；不自动重试失败，原暂停状态和旧guard不补记为完成。

轻量核查命令：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.resume_valais_p1_as monitor --config configs/valais_p1_resume_as_epoch73_20261019.yaml

正常运行只读紧凑快照，核对owner/worker创建身份及退出状态、日志进度、checkpoint时间和GPU。
保持每90分钟核查、健康时静默；有效worker存活不并发GPU任务，不重复测试/smoke/全量哈希。
完成后按原P1协议核验各100轮、275700有效更新、2205000 pair views和两行实际source流，
再显式适配stratified checkpoint的原45条件val，继续固定64个Sion train样本残余耦合诊断。
新flavor身份/源码/采样/数据/clean选模必须完整核对，不能修改checkpoint字段或放宽旧检查。

可行性已得负面证据：C-S最佳clean-val F1=0.3740468676、Recall=0.3506550890；
原B0-E分别0.4205899068、0.4251057994，差值为-0.0465430392、-0.0744507104，
未达原定+0.01/+0.02门。A-S在73轮时best F1=0.3623707017，仍为中间值。
详情：runs/valais_p1_resume_as_epoch73_20261019/preliminary_feasibility.json。
不能由可恢复性/smoke或P0现象宣称方法有效；不追加训练追分，不跳过强普通控制。
P2仍须原文献/机制/预算门，test继续关闭。旧C-S95轮重算与本次未提交步骤如有重算的
额外物理开销需单列，不能宣称两行总物理曝光/耗时完全相同。

## 2026-10-16用户暂停（最高优先）

用户指令：“暂停”。已暂停90分钟托管核查，并在A-S第73轮完整checkpoint提交后，
核对创建身份，停止恢复队列owner与训练worker；不自动续训、评估或探索新实验。
C-S已完成100轮；A-S完整保存至73轮，best/last、优化器、RNG、历史与原成果保留。
本次为用户主动暂停，不将其补记为训练完成或guard完成；后续须等待人类恢复指令。
暂停时间：北京时间2026-10-16 07:37。
记录：runs/valais_p1_user_pause_20261015T233719Z/pause_receipt.json。
下方恢复入口和托管计划均为暂停前历史，不代表当前继续执行授权。

定时任务文字编码已于2026-10-16修复：此前Python stdout为GBK，而PowerShell按UTF-8
解码；暂停更新时把损坏的name/prompt回写，已逐字段复现并恢复原文。今后更新任务用
结构化工具直接传Unicode；经shell读取JSON必须使用ensure_ascii=True，再比较完整
原文和读回值，拒绝含U+FFFD的name/prompt。不得为此更改系统全局编码或重启训练。
修复审计：runs/automation_encoding_repair_20261016/encoding_audit.json。

## 2026-10-15恢复入口（优先于下方历史入口）

用户新指令：“恢复并继续”。原C-S完整保存至94轮，在95轮中途终止，A-S未开始。
旧监控的PID创建身份检查会把仍可查询的已退出进程误判为存活，不能单独使用。
新恢复wrapper不修改已封存的原trainer/配置；原owner/status/log/guard/best/last已独立保存。
恢复准入：346项全量测试通过，两次真实保存状态CUDA下一更新的模型/优化器/loss完全相同。

当前配置：configs/valais_p1_resume_epoch94_20261015.yaml。
当前状态：runs/valais_p1_resume_epoch94_20261015/status.json。
新协议：docs/P1_RESUME_EPOCH94_2026-10-15.md。
当前轻量核查命令：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.resume_valais_p1_study monitor --config configs/valais_p1_resume_epoch94_20261015.yaml

先完成C-S95–100，再从原初始化完成A-S100轮；本次一次恢复额度已使用，不自动循环重启。
监控核验进程等待/退出状态、创建身份、日志时间和checkpoint进展；无需加载模型或打印
完整guard observations。原95轮未提交部分需重算，额外物理计算量必须在报告中披露，
有效提交预算/采样流仍须匹配。原45条件val与train残余耦合诊断待两行完整结束后接续。

用户授权：2026-10-14正式训练、证据驱动创新探索、每90分钟核查、节省token。
主协议：docs/P1_MANAGED_PROTOCOL_2026-10-14.md。
主配置：configs/valais_p1_study_v1_20261014.yaml。
工作区：E:/new research/detection；Python：E:/Download/anaconda3/envs/geosa-basa/python.exe。

状态入口：runs/valais_p1_study_v1_20261014/status.json。
两行目录：runs/valais_cs_fp32_seed42_v1_20261014、runs/valais_as_fp32_seed42_v1_20261014。
序列：C-S100轮→A-S100轮→新绑定的原45条件val→原train子集残余耦合诊断。
首次最高clean-val F1选模；FP32/CUDA/原模型/阈值与所有原协议保持，唯一新研究项为分层source采样。

轻量核查命令：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.valais_p1_study monitor --config configs/valais_p1_study_v1_20261014.yaml

正常运行只读这个紧凑快照，不加载checkpoint、不重跑检查或大文献检索。读取stderr只在异常。
不得在有效owner/worker存活时启动第二个GPU进程。温度/数值停止保留现场，先诊断；
外部中断按主协议另立恢复记录，一次epoch边界恢复，不重跑已完成行。

两行训练完成时先检查各guard/hash、100轮、275700成功更新、2205000 pair views、
sampling_actual逐轮一致，然后准备新flavor的冻结val适配。兼容验证须同时核验采样身份、
源码和真实源序列；不改保存的checkpoint字段，不放宽原data/threshold/clean选模检查。
现有pcdc.evaluate_candidate与scripts.evaluate_candidate_guarded可复用评分数学；
scripts.valais_matched_study.evaluation_job可参考其配置结构，新的行要独立命名。
随后复用固定P0 train QA及同一图像配对检查残余问题，原清单/checkpoint绑定保持原样。

创新探索按主协议与10月7日replan继续；优先补齐CPEN全文/代码差异及普通控制，
因果或方法优势未成立前不把混合响应i当损伤图、不把Hessian/DRO换名为新算法。
P2须在计算pilot、近邻与机制门通过后冻结新协议和有限预算再训练。Valais Martigny
test、旧LEVIR test、几何扰动、基础模型和未经审查的新数据集不进入托管。

正常状态无需重复报告。仅新结果、失败或需用户处理时最多三行中文，避免重复历史说明。
计划阶段完成且无已准入的下一实验时记录下一证据门，暂停此托管；不得无限自我续训。
