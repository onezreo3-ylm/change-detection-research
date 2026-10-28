# 窗口诊断独立监督与连续执行

2026-10-25，Asia/Shanghai。用户要求“加快实验进度”，本次补齐独立监督后在同一轮串行完成已授权pilot和主诊断，
不再将可连续完成的步骤等待至下一小时。原小时托管频率不变，数据、精度、阈值、资源门和停止条件不变。

原核心、原冻结研究YAML及既有prepared/source_snapshot保持。新增独立模块为
`scripts/tiling_probe_supervision.py`、`scripts/tiling_probe_cuda.py`，运行配置为
`configs/levir_tiling_runtime_20261025.yaml`，专用测试为`tests/test_tiling_probe_supervision.py`。

监督进程不导入torch。CPU owner及两个独立guard/watchdog通过PID、创建时间、可执行路径和完整命令行绑定worker，
worker确认三个监督进程都存活、心跳新鲜、原GPU UUID及无CUDA映射后才导入CUDA执行器。
原geosa-basa、RTX3060 Laptop、FP16 autocast、channels_last及原checkpoint全部保持。

温控每2秒读取，85°C连续3次确认（热状态间隔1秒），86°C立即停；单次传感器超时1秒，连续3次失败停止。
guard与watchdog分别检查对方、CPU owner、30秒心跳、实际批次进度及独立墙钟。
CPU owner另行强制运行上限，不能用小时App调度或batch间回调代替独立保护。
共享gpu_study.lock与实际Python worker检查防止并发，两个mode均只能登记一次，失败保留，不自动重试。

先完成本次新增代码的全量unittest和原架构CPU synthetic smoke；检查通过后封存运行代码、配置、测试证据、
既有prepared身份和授权。新合成输出独立保存，不覆盖之前成功或失败的smoke。工程结果不作研究结果。

固定pilot只取原train16排序首源train_16.png。两个原模型各读取512窗口起点序列的前4块，以原batch4进行
一次正常及一次完全相同输入重复，共2模型×4×2=16个pair预测。检查有限FP16 logits、重复精确、模型状态不变及显存，
不算标签分数、不选择超参数、没有额外warmup。pilot的owner/worker/guard/watchdog全部正常退出后才可启动main。

main直接使用已封存四臂核心：3168个主patch-pair、4032个256窗口像元等量，另有16个重复/时相交换自检pair。
原batch32网格须逐像素二值和逐源混淆计数精确重放；失败则停止，不改阈值或容差继续。
共同中心支持、逐源计数池化、概率先平均后阈值及原双模型资源门均按原研究协议。

pilot<=900秒、main<=1800秒，新probe所有GPU worker累计<=3600秒；按CPU owner完整运行墙钟保守扣费，
包含启动、CPU、I/O和守卫结束开销。每次分配还取原48小时共享账本的剩余额度，不因新文件或mode清零。
本次工程测试开销继续保留；没有更长训练、原始开发val/test/Martigny访问或其他新模型的授权。

只有实际创建身份、日志中的批次推进及正常结束证据成立时才报告运行/完成。
所有臂和失败项一并报告，训练集窗口诊断不称独立泛化改善或新方法成果。
