# 当前托管接续

最近核验：首轮1780次尝试全部成功，0 AMP跳步，14240 pair views与原B1源/实例流/曝光计数精确相同；
首轮clean val F1=0.7788441239034746，best.pt和last.pt已写入。正在第2轮（核验时step435/1780），
四个创建身份绑定进程均活跃，GPU75°C，当前守卫最高观测76°C。首轮指标不是效果结论。
见 `verification_20261025.json` 和公共managed status；后续以实时状态为准。

2026-10-25北京时间。此目录是同一个普通B1 q=0.5尾部控制的已诊断基础设施修复运行，
不是新科学候选，也没有新预算。原首个训练prefix现场在相邻 `levir_tail_risk_control_20261024`，
201次更新/1608 pair views作废，不带入本次模型；47.906秒仍从公共账本和同一24h上限扣除。

恢复前：459项全量测试、CPU synthetic smoke、4进程2000写/11970读JSON压力检查、
99次真实CUDA尝试均通过。原mean、候选32步和重复32步的完整tensor/BN/optimizer/scaler/RNG
边界与修复前完全相同；q1值与梯度、恢复下一更新、原源/实例流及实际尾批均通过。
峰值allocated902886912/reserved1130364928字节（含重放）；pilot最高观测61°C。
工程GPU开销全部并入 `../managed_continuation_20261024/compute_ledger.json`。
含25%余量估算40925秒，本次最多86352.094秒，不延长、不换名清零。

已实际从原seed42初始化启动，worker/owner/guard/watchdog的精确PID、创建时间、命令行见
`training_session/attachment.json`。启动观测worker11252，owner3304，guard22856，watchdog32092；
这些数字仅用于定位，必须实时核对创建身份，不能仅按PID认活。

使用原geosa-basa Python运行以下命令（无模型加载、无训练启动）：

```
E:/Download/anaconda3/envs/geosa-basa/python.exe -X utf8 -m scripts.levir_tail_risk_supervision monitor --runtime configs/levir_tail_risk_runtime_20261025_iofix.yaml --mode train
```

训练保持100epoch、178000尝试/1424000有效run pair views、FP16、原增强、原模型与clean val F1选模。
每个epoch完成时自动比对原B1源/扰动实例hash及曝光计数，不一致立即停。
有效run的成功更新与AMP跳步分开记录；不能把废弃prefix加进有效100轮。
新checkpoint flavor为 `b1_secondary_empirical_tail_v1`，不得改名成旧matched flavor绕过评价器。

只作每小时紧凑监控；独立秒级温控/owner互监/墙钟watchdog继续运行。
不在有效GPU训练时跑全量测试、smoke、全量哈希或加载模型。状态读取用
`scripts.windows_shared_json.read_json`，避免CRT读者与原子replace发生Windows竞争。
85°C确认、86°C立即停；30秒心跳、900秒实际batch进度、连续AMP跳步>8或scale<1均保护停止。

一次已验证的基础设施恢复已经使用。若此run再异常，保存并诊断，不能再自动新建目录重启。
100轮正常结束后，再检查全部原B1流/计数/守卫/退出和真实flavor，然后新增显式risk评价适配，
先复现clean best，再按原45条件val及固定tail5/clean/mean/recall/FPR/source-bootstrap门分析。
所有门见原2026-10-24协议；单seed通过只允许讨论43/44，不自动启动更多seed。

旧LEVIR test及Martigny关闭；原区域共同miss3.30%未达5%门的结论保持，不能重跑/改门/转区域loss。
两行AI代填及291歧义不是独立真值，不要求用户再次填表，也不伪造独立标注。
