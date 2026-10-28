# 实验中断分析与本对话续接（2026-10-02）

承接旧对话 `Execute staged PCDC experiments`（01a0f7d8-7bf2-7102-a554-0ac4374cc256），当前对话为 `续接中断的实验进度`（01a0fc9d-87f0-7542-962a-dceb836bac6a）。用户要求分析中断并在此续接；此前用户已授权按最新判断自主实验、分步推进和自行处理突发情况。本次先完成中断时尚未完成的 Valais validation 扰动协议准备与验收，保留历史研究和失败现场。

## 中断层与证据

- 旧对话在 2026-10-02 多个 turn 中记录同一错误：`504 Gateway Timeout: upstream timed out in stream_open attempt=1/1 after 1m0s, url: http://localhost:65467/v1/responses`。随后数次重连 turn 为 `interrupted`，没有新实验命令。该证据定位到模型响应连接层；无法仅凭该错误进一步区分上游拥塞、网络或代理内部阻塞。
- 当前 65467 监听进程是 `cockpit-cliproxy.exe`，路径 `E:/Download/Cockpit Tools/cockpit-cliproxy.exe`。当前对话可调用工具；这只证明当前链路可用，不能证明先前的超时永久消除。本次不更改账号、代理或模型配置。[官方配置参考](https://developers.openai.com/codex/config-reference/)说明 `model_providers.<id>.base_url` 是模型服务地址；具体故障判定来自本机进程与旧对话错误。
- 这次连接中断发生时已经没有待续训的 clean 基线：全 FP32 seed42 训练器和 PID 温度守卫完成 100 轮。外层 Windows PowerShell 5.1 将 JSON 顶层 history 数组解析为一个嵌套数组，导致完成门误判、原 `guard_completion.json` 缺失。该独立问题已由[只读完成审计](VALAIS_FULL_FP32_COMPLETION_AUDIT_2026-10-02.md)处理；原运行与失败标记不回填。
- [独立受守卫 clean-val 复核](VALAIS_FULL_FP32_INDEPENDENT_VAL_2026-10-02.md)已完成，选定 epoch53，Sierre pooled F1@0.5 = 0.4162248161980205，checkpoint SHA256 `de42460dfce94717a890ae359124c659ce57d7d396f44429ac370463e1991b7d`。此为开发城复现，不是跨城泛化结论。

## 准确续接点

原[训练候选 QA](VALAIS_TRAIN_SEVERITY_QA_CANDIDATE_V1_2026-10-02.md)与[方向均衡补充 QA](VALAIS_TRAIN_SEVERITY_QA_SUPPLEMENT_V1_2026-10-02.md)合计 16 个不重叠 Sion train 来源、1,440 个实例和 36 张联系表；原始报告仍标记待视觉验收。旧对话已经实际查看样例并指出 L5 噪声/模糊及小目标的局限，但在写正式验收前超时。

`configs/valais_corruption_protocol_v1_20261002.yaml` 已写入，SHA256 `921c4b0a3958c0b89b4736c4772bd75b23d73d22ebe0536b18b36fa6985aa19a`。原 `runs/valais_corruption_protocol_preflight_20261002` 有 257 项单测通过和真实全 FP32 CUDA smoke；尚无 `runs/valais_corruption_protocol_v1_20261002` 验证清单。正式扰动评分及 B1 长训均未启动。

## 当前批次与边界

1. 重新核对训练 QA 图像、报告及输入哈希，在单独记录中绑定视觉判断；不把数值不变量代替视觉验收。
2. 用现有准备入口产生独立的 validation 候选清单：1715 Sierre tiles × 九类 × 五级 = 77,175 个实例，共 45 条件。global/local、方向与目标时相采用预定流分配；强度共享相同方向、目标时相、mask 与噪声场。preparation 只产生元数据和训练预览，不执行模型。
3. 写独立 acceptance 与评估 YAML，绑定协议、清单、实例、checkpoint 和环境；确认源版本、阈值0.5、FP32、split与选模规则不变。完成必要单测和真实 CUDA smoke后，才能进入受温度保护的 validation 评分。
4. 结果仅供 Valais 开发比较。B1 仍须另立匹配初始化、曝光/预算、FP32与 clean-val 选模协议；本批不自动启动 B1 长训、校正/门控模型或 Martigny test。

Martigny test 继续封闭；已用于最终报告的 LEVIR test 不重开。所有新输出使用新目录，拒绝覆盖；失败保留现场，不补造完成标记。现有小时 heartbeat 后续迁移到本对话，并更新为本记录中的准确检查点；健康或无新证据时静默。

## 本次执行结果

进行中：正在复核训练 QA 并准备冻结 validation 清单。完成后补充真实产物与状态；本节不是评分结果。
