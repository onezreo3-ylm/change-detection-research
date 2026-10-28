# 512训练完成与256匹配控制接续

2026-10-26 01:03:45，Asia/Shanghai。原native512组已完成100轮，worker/owner正常exit0，独立guard/watchdog正常收尾，最高GPU温度81°C。

178000次尝试=177939次成功更新+61次原FP16 AMP跳步；实际356000个512 pair views，折合1424000个256窗口等量。100轮源group顺序与事前hash逐轮一致，原64源/1024个256开发val窗口计数及F1算术核对通过。

按原clean-val F1选中的第58轮F1为88.8062432561%。目前只有512组完成，该数字不能建立相对配对256组的提升结论，也不把历史随机成批B0-E替代新的成组控制。

新的best.pt、last.pt已CPU核对flavor、arm、原初始化、来源、完整history和train/val隔离，并封存SHA256；没有模型前向、原图访问、held-out或新的评分。该CPU核对保守计费4秒，原共享账本累计60953.699964秒，不重置原48小时上限。

既定grouped256阶段可用上限86158.883秒，仍覆盖冻结预测40810.403秒。下一步是在原FP16、原GPU、相同group顺序和损失单位下串行运行grouped256的100轮；不再启动pilot、改阈值、换seed或归一化。

证据：`runs/levir_native_window_training_20261025/native512_completion_verification.json`。此前启动及首轮记录保持原快照；此文件承接新阶段。

## 256匹配控制已接续

2026-10-26 01:07:18（Asia/Shanghai）：grouped256正式worker已启动并实际完成966次尝试，当前第1/100轮。worker PID 28148、创建身份134374214749487065，新的owner/guard/watchdog身份均核验有效；当前GPU温度75°C、温控最高75°C，无数值报警。没有并发原512模型、重跑pilot或测试。首次完整epoch尚未提交，group顺序由冻结训练器在每轮末精确核验。原FP16、原GPU及原配对协议保持。

接续证据：`runs/levir_native_window_training_20261025/grouped256_start_verification.json`。每小时托管已改为检查当前grouped256，不再检查正常退出的native512旧PID；待该组100轮完成后按原门比较两组保存的开发val计数。
