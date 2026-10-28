# 普通B1尾部控制：最终固定验证

**更新：评价已正常完成，候选按原门结束。** 7项门仅召回保护门通过，其余6项未通过。
clean差-0.002035496、mean差-0.001350198、tail5差+0.000134989、最差单条件差-0.000026632，
pooled FPR差+0.000113266；tail5增量95%区间[-0.001697609,+0.001846092]跨0。
owner/worker退出0，guard/watchdog正常结束，47104 pair预测与20000源对bootstrap已审计。
报告：docs/LEVIR_TAIL_RISK_RESULTS_2026-10-25.md；完成证据verification_completed.json。
当前没有训练/评价worker，不要再执行launch或重跑评分。后续仅新问题/数据准入screen。

训练已正常完成并核验100轮全部原B1源/实例流、曝光、学习率与数据完整审计，
178000尝试=177934成功+66 AMP跳步，1424000有效run pair views，wall29015.172秒，最高观测77°C。
独立的训练完成证据在 `training_completion_verification.json`，best实际SHA256为
`79f2f81b932461ed53e5543f086cd2ef66a27ae5be64ab97dd6bfeb9cabcbad4`，epoch65。

471项全量测试和原架构CPU FP32 synthetic smoke通过，评价授权与源码/输入哈希已封存。
真实原FP16 clean重放与风险checkpoint的所有计数/指标精确相同，F1=0.9040373724549671；
原mean B1=0.9060728685624768，差-0.002035496107509749，已失败原clean>=-0.0005门。
因此不能追加训练、调q/λ、改强度/阈值、或自动43/44复现。原定45条件仍完整诊断。

当前评价运行使用下列监控命令，不加载模型、不启动训练：

```
E:/Download/anaconda3/envs/geosa-basa/python.exe -X utf8 -m scripts.levir_tail_risk_eval_owner monitor --runtime configs/levir_tail_risk_eval_runtime_20261025.yaml
```

再读本目录status.json与analysis.json（存在时）。JSON用 `scripts.windows_shared_json.read_json`。
启动观测owner28512、worker28220、guard30988、watchdog20768；具体PID/创建身份/命令行以
`training_session/attachment.json`及实时句柄为准，不按数字认活。training_session是复用安全容器名，
实际任务明确fixed_validation_only，optimizer_updates=0，没有训练或checkpoint重命名。

评价复用原冻结45条件/64原源对/每条件1024tiles，先clean1024，再45*1024，共47104个pair预测。
原0.5阈值、FP16、batch32、共享ResNet18/abs-FPN与floor255保持；不打开旧LEVIR test/Martigny。
每条件JSON保留全局/原源/原注入stratum计数，不能删条件或按结果改变门。
最后GPU模型释放后worker自动执行20000次共享原源对bootstrap，所有条件/模型共同抽样，
先池化TN/FP/FN/TP，再每模型每次重排最差5，生成analysis.json与bootstrap_tail_deltas.npy。

独立2秒温控85°C确认/86°C立即停、owner/guard/watchdog互监、30秒心跳/900秒实际进度保护。
最多7200秒，并扣同一候选24小时剩余与原48小时账本，不因新目录重置。首次废弃201步/47.906秒
仍保留并扣费，不并入有效模型100轮。评价异常保留并诊断，不自动重复运行。

完成后核对owner/worker退出0、guard/watchdog均worker_exited且无错误、45条件与64源支持完整、
全部固定门与源对bootstrap，写中文结果与停止/转向判断并封存。单seed/开发源对不能证明总体
训练随机性，也不能替代独立标签真值；两条AI代填/291歧义保持，旧区域3.30%未过5%门不重开。
原用户表、标签、旧评分、训练授权/快照及封存文件不覆盖，不启动旧C5/R5/M5/K5/W5/P2。
