# P1 A-S 第73轮后的人工恢复

2026-10-19（Asia/Shanghai），用户明确要求“恢复实验，并继续进行可行性验证，并开始执行训练任务”。
该指令解除10月16日的用户暂停。保留原协议、原运行目录、原模型和全部历史成果。
本次只恢复A-S的74–100轮，不重跑已完成的C-S，不创建新算法训练，不开放test。

原入口为configs/valais_p1_study_v1_20261014.yaml；上次恢复队列为
configs/valais_p1_resume_epoch94_20261015.yaml。两者已封存，不修改其脚本或绑定文件。
本次使用独立configs/valais_p1_resume_as_epoch73_20261019.yaml和
scripts.resume_valais_p1_as。新输出目录保存暂停收据、旧owner/status/log/guard、
A-S best/last/history/sampling及C-S完成证据，逐文件核对复制前后SHA256。

恢复前验证旧owner/worker退出、两级封存链、C-S完成guard/hash与100轮有效预算，
A-S checkpoint内73轮历史、实际采样、最早最高clean-val F1选模、完整RNG、
优化器/模型有限值及原训练/采样/数据/初始张量身份。生产trainer再次全量审计train/val。
原stderr中pickle data was truncated出现在用户终止进程时，保留原文；结合主动暂停收据
解释为中断现场，不能据此证明checkpoint损坏或掩盖其错误。恢复必须通过真实保存状态smoke。

准入为完整unittest以及两次从同一A-S真实保存状态执行下一source batch的FP32 CUDA更新。
两次模型/优化器/loss与更新后的RNG须精确相同，每次4 source、8 pair views、
1成功更新和1次BN更新；只读train，不把smoke当作研究效果。

正式训练仅调用原scripts.train_stratified_robust --resume，完整恢复优化器、scheduler、
Python/NumPy/Torch/CUDA/loader RNG。保持初始张量、seed42、batch4、FP32、采样与
扰动实例规则、学习率、loss、固定阈值0.5和clean-val选模。新guard会话独立保存；
原85°C确认/86°C立即停止规则不变。一次启动额度，不自动重试数值或温度失败。

有效预算仍各100轮、275700成功更新、2205000 pair views。A-S前73轮为201261次更新、
1609650 pair views，剩余74439次更新、595350 pair views。未提交的暂停后步骤数量未知，
如有则全部丢弃并从74轮开头重算；额外物理计算与smoke不计入有效预算，不能宣称实际
总曝光、耗时完全匹配。C-S旧95轮重算的披露继续有效。

完成时核对两行100轮采样流、guard/hash与有效预算，再进入原45条件val和固定64个Sion
train样本的残余耦合诊断。新训练flavor的评价兼容必须显式核验身份，不修改checkpoint或
放宽旧数据/阈值检查；按P1_MANAGED_PROTOCOL_2026-10-14.md的冻结门报告可行性。
P0和clean-val的中间值不能代替最终鲁棒性门，不预先宣布创新成立。

本次延续此前每90分钟的托管核查，只在完成、失败或有实质新证据时通知。新暂停指令优先。
