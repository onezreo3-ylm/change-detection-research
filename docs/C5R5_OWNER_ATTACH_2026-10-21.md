# 现有C5的CPU监督接管记录

2026-10-21。原一次启动任务Codex-Valais-C5R5-40ep-BF16-20261021在C5开始后返回
3221225786（0xC000013A，控制台终止退出）。原owner PID18000已消失，queue stdout/stderr
均空，不能据现有证据确定终止发送方。原记录、TaskScheduler返回码和owner/queue文件保留。
同时Codex app automation工具报告Transport closed；只记录两者发生，不声称已证明因果。

C5 worker PID21216、创建ticks134370423605411398仍存活并实际推进step，数值stderr为空，
其独立温度守卫正常。此次不重启C5、不回滚checkpoint、不重新计算已提交或未提交更新；
GPU研究进程保持同一个，原初始化/数值/40轮日程/数据/源码/预算/guard/hash全部保持。
这是CPU监督接管，R5仍只有原计划中尚未消费的一次启动，不是训练失败的自动重试。

新入口scripts.c5r5_supervision_attach，独立监督目录绑定原授权、配置、原owner/queue快照、
活C5的PID创建身份、新CPU控制源码和4项CPU专用测试。只在原owner失效且原C5身份/等待状态
有效、原queue明确C5运行且R5尚未启动时允许接管；已有接管owner则拒绝第二次接管。
新任务直接使用同一geosa-basa环境的pythonw.exe作为无控制台CPU宿主，GPU worker仍由
原python.exe运行且CREATE_NO_WINDOW；不再使用原PowerShell控制台wrapper作为队列宿主。
新任务0触发器、0自动重试、禁止并发、隐藏；此接管本身只允许一次。

CPU监督持有原C5 Windows进程handle，核对创建时间并等待真实退出，退出码0及原完整
check_completion预算/guard/hash/实际流审计通过后，才按原命令首次启动R5。R5完成后同样
核查40轮/110280更新/2205000 views/441000源抽取/两行实际候选像素流。所有输出留在
独立监督目录，原失效owner/queue不被伪装或覆盖。原worker预算继续扣除工程和已用时间。

新代码不加载模型、不增加GPU更新、不更改任何已封存训练源码/配置/原authorization。
原407项完整测试仍对应未变的训练实现；新增4项CPU监督测试通过，并以持有活worker身份/
等待状态、两次真实step进展核对作为CPU接管smoke。全套测试含3项真实CUDA用例，按正在
执行的单GPU协议不与训练并发重跑；后续评价适配的完整测试阶段再一并覆盖新增CPU用例。

独立监督的monitor是当前入口，必须同时展示原owner失效记录、现owner/worker创建身份/
等待退出状态与原实际日志/checkpoint进度。活PID不代表进步；无实际进展或再次监督失败
保留现场，不循环接管或重启GPU。人类暂停优先，需要同时处理当前监督与当前worker。
自动化工具连接恢复前，不宣称原90分钟heartbeat已恢复；独立队列仍按原授权顺序执行训练。
