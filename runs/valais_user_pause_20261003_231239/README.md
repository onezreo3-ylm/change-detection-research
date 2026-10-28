# Valais 用户暂时停止记录

已按用户“请确认安全保存情况下暂时停止”停止队列与B0-E worker，并禁用一次性Windows任务。
进程身份经PID、创建时间、可执行文件及命令行核对；停止后实验进程列表为空。

## 保留进度

- B1：100轮训练及45条件验证完成；训练与验证guard完成哈希核对成功，原结果完整保留。
- B0-E：保留完整第31轮的last.pt，以及epoch29的best.pt。
- 两项B0-E checkpoint已复制到本目录checkpoints，CPU weights_only加载成功，模型张量有限。
  checkpoint包含模型、optimizer、scheduler、scaler、随机状态及DataLoader状态、完整轮次history。
- 原checkpoint与备份SHA256一致；59个封存输入文件不变。
- 未完成的第32轮舍弃。其部分更新只在已终止进程的内存里，不写入第31轮checkpoint。
- 当前study及B0-E status明确记为paused_user_requested；修改前原状态、history、guard与日志已另存。
- 没有启动新推理或读取test，没有生成完整B1/B0-E比较结论。

## 恢复边界

不会自动恢复。用户后续要求继续时，先核对pause_record.json、checkpoint及封存输入。
训练数值状态可从第31轮checkpoint恢复，下一完整轮次为32。
原queue是一次性入口，存在execution_started记录；原train-stage拒绝已存在的输出目录，不能直接
重新启用任务或重复原启动命令。需要单列恢复入口和新的guard session，并保留本次中断证据。
B0-E wrapper的实际source顺序SHA256只保存在原进程内存，暂停时没有source_order.json；
恢复前应重建并核验完成前缀，再接续记录。完成上述核验前不能声称完整匹配顺序已验证。
旧guard快照保持原样，其monitoring状态是外部停止前的历史记录，不是完成guard。
计划任务LastTaskResult=1是人工终止结果，不能解释成研究方法失败。

详细保存与核验证据：pause_record.json、prestop_checkpoint_verification.json、stop_actions.json、
task_and_processes_after.json；原结果位置及关键文件哈希均在pause_record.json。
