# 固定普通尾部控制：训练结束后的验证协议

执行原 `LEVIR_TAIL_RISK_CONTROL_PROTOCOL_2026-10-24.md`，不新增成功条件或参数搜索。
本轮原mean B1和尾部控制同为seed42共享ResNet18/绝对差FPN、原FP16、100轮和clean val选模。
新训练于2026-10-25北京时间08:49正常退出。记录178000次尝试、177934次成功、66次AMP跳步，
1424000有效run pair views，最佳epoch65、记录clean F1=0.9040373724549671；尚需实际重放核验。
原mean参考clean F1=0.9060728685624768，当前记录差为-0.0020354961075097，低于原clean保护门。
后续45条件仍作为原定开发诊断完成；不通过任何保护门就不再推进该候选训练。

## 身份与数据

新增独立评价脚本，不改旧matched评价器，不重标checkpoint flavor，不monkeypatch全局。
实际risk flavor为 `b1_secondary_empirical_tail_v1`；核对原授权、配置/源码签名、初始化、
100轮历史和每轮原B1源/实例hash、物理曝光、学习率与尝试数，best只由clean val选择。
重新核验原train/val数据完整审计。旧test和Martigny始终关闭。
新数据目录仅为评价产物 `runs/levir_tail_risk_evaluation_20261025`，原数据/评分/用户文件不覆盖。

使用原已接受的冻结benchmark manifest及全部45条件/1024tiles/64原源对，0.5阈值、FP16、
eval batch32、原归一化和floor255标签。复用原run_condition/FrozenInstanceDataset统计与实例实现；
先精确复现risk自身best记录的所有clean计数及指标，再评分全部45条件，不按结果删条件或改强度。

## 门与bootstrap

原门不改：clean和mean F1差均>=-0.0005、pooled recall差>=-0.001、背景FPR差<=0.0001，
最差5条件均值差及最差单条件差均>=0.002，源对bootstrap尾部增量双侧95%区间下界>0。
每个模型从全部45个条件各自排序取最差5，不能固定有利的共同条件集合。

20000次原源对有放回共同抽样，seed20261024，numpy default_rng/PCG64，
每次integers生成64个源索引，转成multiplicity后共同池化所有条件及两个模型的TN/FP/FN/TP。
先池化计数，再重算每条件F1，每个模型每次重新排序取最差5，最后取候选减原mean。
分块512仅控制内存，不改变随机流，CPU单元检查与朴素逐次共同抽样/重排实现精确一致。
95%区间使用numpy linear percentile的2.5/97.5百分位。结果只描述开发源对不确定性，
不能当作多seed随机性或独立标签真值认证。

## 有限执行与监督

先运行新增代码所需的完整unittest、原架构CPU FP32 synthetic smoke（明确为工程检查），
封存身份后才开始真实原FP16验证。验证worker不创建optimizer，优化更新数为0。
隐藏CPU owner、独立温度守卫与独立watchdog保持PID/创建身份/命令行绑定、2秒级采样，
85°C确认/86°C立即停、30秒心跳/900秒实际batch进展保护，沿用已验证的Windows互斥JSON。
验证最多7200秒，还同时受原候选24小时剩余及公共48小时剩余额度限制；包括bootstrap/I/O的
整个worker墙钟继续记入原公共账本，不因评价目录或签名重置。异常保留，不自动重跑。
旧守卫使用的session容器沿用training_session文件夹名，里面实际任务明确为fixed_validation_only。
单seed若全部通过，也仅允许讨论43/44复现，不自动启动其它seed。
