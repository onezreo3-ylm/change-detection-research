# C5/R5固定评价执行进展

## 当前：完成、审计通过、托管暂停

2026-10-22 13:30:45（北京时间）队列完成，C5/R5各45条件及残余均成功退出0，Task结果0。
14:11最终保存结果审计通过，没有重新推理/加载checkpoint/训练、重跑测试或全量输入哈希。
两行clean计数/五指标精确复现，45条件及原64 train四格/边际/交换/预算/guard/hash完整。
残余6272主预测/6276总预测/1536记录，最高53°C；C5/R5 val最高69/70°C。

| 指标 | C5 | R5 | R5−C5 |
| --- | ---: | ---: | ---: |
| clean F1 | 0.371138431995 | 0.327698921243 | -0.043439510752 |
| 45条件mf1 | 0.330364845129 | 0.300375283107 | -0.029989562022 |

R5主要六门中前四项失败、两空tile FPR门通过；残余两模型均四族通过，不构成效果或新方法准入。
报告docs/C5R5_FEASIBILITY_RESULTS_2026-10-22.md；结果comparison.json；
最终收尾审计final_closeout_audit_20261022.json和residual/final_saved_result_audit.json。
无已完整准入的下一实验，已按协议用工具暂停levir-cd，实际automation.toml状态PAUSED已核验。
保留全部结果、原协议/清单/阈值/checkpoint与额外物理计算记录，不自动推进新训练/P2。
下方为本次启动和运行阶段历史，不能据其ACTIVE或3/45状态重启任务。

## 启动阶段记录（历史）

2026-10-22 12:35（北京时间）首次启动隐藏无控制台队列。
配置configs/valais_c5r5_fixed_eval_v1_20261022.yaml；
协议docs/C5R5_FIXED_EVALUATION_PROTOCOL_2026-10-22.md；
目录runs/valais_c5r5_fixed_eval_v1_20261022。

训练两行已完成40轮，完整提交/预算/guard/hash/退出码0及全部实际流审计通过。
C5最佳epoch38 clean F1=0.3711384319946564，R5最佳epoch21 F1=0.32769892124259875；
R5相对C5 F1下降0.04343951075205765，clean F1和Recall保护门均已失败。
其他扰动/空tile门等待完整评分，不补写通过。

421项完整测试80.944秒通过（含实际CUDA），日志runs/c5r5_fixed_eval_unittest_20261022.log。
新flavor/BF16训练/FP32评价身份、完整历史/首次clean选模、40轮再生成计划、
全部train/val数据审计、原45条件及P0固定样本输入核验通过，5477项准备输入。
真实FP32 CUDA smoke102预测、26.875秒，最高46°C，原首train样本全部四格/边际通过，
两模型重复逐位一致、交换误差0、模型参数与BN状态冻结；无训练、无val/test评分。
新authorization绑定5481项输入；原训练checkpoint/源文件和旧检查均未修改。

任务Codex-Valais-C5R5-FixedEval-20261022；直接同环境pythonw.exe CPU owner，
无自动trigger、无重试、禁止并发，总任务20小时有限上限、每stage6小时硬上限。
owner36104/创建ticks134371173541607643，首worker21776/创建ticks134371173675935992。
12:37:59监测确认C5 clean计数与五指标精确复现，已实际完成45条件中的3项；
owner/worker创建身份和等待有效，最新日志5.1秒前推进，GPU55°C。
紧凑证据launch_monitor_snapshot.json。必须有实际条件/样本进展才持续认定健康。
序列C5原45val→R5原45val→固定64 train残余；原阈值/强度/FP32 CUDA/温度门保持。
失败保留现场，不重试/覆盖评分/追加训练。尚无完成的45条件或残余结论。

轻量监测：

    & 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m scripts.c5r5_fixed_evaluation monitor --config configs/valais_c5r5_fixed_eval_v1_20261022.yaml

现有levir-cd已通过工具更新为当前入口并ACTIVE，每90分钟，正常健康静默。
结束核对两行saved_result_audit和residual/final_saved_result_audit.json、comparison.json。
闭环保存可行性报告，无已准入下一实验就暂停托管，不能直接转P2。
训练8小时24分钟包含旧工程预检扣款，本日全量测试/smoke与正式评价计算另列。
210额外优化器计算/4158 pair forwards、旧P1重算/恢复smoke及未知废弃步数均保留披露。
