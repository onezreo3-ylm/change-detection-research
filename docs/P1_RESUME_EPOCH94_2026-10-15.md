# P1第94轮checkpoint后的恢复

2026-10-15：用户明确要求“恢复并继续”。原C-S在epoch95中途停止，完整last.pt记录
epoch94；A-S尚未启动。原owner7128、worker12604的创建身份仍可查询，但Windows退出
查询均为exit0，活动进程列表中不存在，日志在07:39 UTC后停止。原监控仅检查创建时间，
错误地报告存活。新监控同时检查同一handle的创建时间、零超时等待与退出状态，并检查日志
进度；不修改已封存的原脚本。参考：
[GetExitCodeProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess)、
[WaitForSingleObject](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject)。

仅能确认进程已终止，不能由exit0或空stderr证明具体外部原因。原温度记录所有decision
为ok，最后81°C，已记录峰值84°C，未见温度/数值错误记录。恢复前再次确认旧owner/worker
及其子进程均已结束，保护全部现场；不把旧running状态改成已完成，不补造guard完成文件。

恢复目录：runs/valais_p1_resume_epoch94_20261015。
原owner/status、日志、guard及best/last等保存到其saved_prefix，核对复制前后哈希。
核验94轮history、实际采样与原sampling_plan、模型/优化器有限值、原训练/采样/数据身份、
best选模以及完整Python/NumPy/Torch/CUDA/loader RNG。生产trainer重新核对全train/val
数据审计。测试集继续关闭。

恢复准入：全量unittest与两次从同一真实保存状态恢复的CUDA单batch更新，模型/优化器/
loss必须逐位复现；每次仍为4 source、8 pair views、1次成功更新、1次BN更新。
smoke是工程检查，不报告研究效果。原trainer、YAML、初始化、精度、模型、loss、学习率、
采样种子与阈值均保持。新恢复wrapper和新进程监控独立封存，启动次数上限1，不自动重试。

正式序列：从epoch94的last.pt恢复，完整重算epoch95，完成C-S到100；随后原A-S从冻结
seed42初始化训练100轮。原best与94轮prefix已独立保存，恢复后仍按首次最高clean-val F1
选模。新温度guard使用原目录下独立temperature_guard_resumes/session_0001。

有效提交预算仍各为275700成功更新、2205000 pair views，两行实际source流应完全相同。
epoch95原未提交部分至少有日志确认的1375步，精确已执行但丢弃的更新数未知。恢复需要
重算这一部分，因此物理计算/读取/更新尝试总量额外增加；不能无说明声称两行实际总耗时、
总物理曝光完全匹配。原有效优化轨迹按checkpoint/RNG重建，废弃计算另列，不计入研究指标。

完成后验证100轮预算、实际采样与guard/hash，继续原45条件val及固定train残余耦合诊断。
后续证据门与P1_MANAGED_PROTOCOL_2026-10-14.md保持；不重开旧test或自动新增结构。
