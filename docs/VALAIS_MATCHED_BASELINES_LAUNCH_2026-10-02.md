# Valais B1 / B0-E：本批启动记录

北京时间2026-10-02 **23:07:49**，一次性Windows任务
`Codex-Valais-Matched-B1-B0E-V1-20261002` 已启动队列，首次状态为 `b1_training`。
这是启动记录，不是100轮训练或鲁棒性比较的完成声明。

## 本次已完成

- 已有B0 45条件完成门、哈希及源计数复核，详细结果见
  [结果记录](VALAIS_BASELINE_CORRUPTION_RESULTS_2026-10-02.md)。
- 固定8张Sion train原256像素诊断与实际视觉审查，保留小目标/边界/语义标注限制。
  新PNG总量2341834字节；后续优先读diagnostic_review.json，不重复加载8张图。
- 267项全量单测通过；B1/B0-E真实CUDA FP32双视图smoke通过，初始化一致、各一次
  BN和优化更新。这是工程检查。
- 全量live train/val审计与B0相同；协议、解释器、源码、checkpoint、视觉、测试、
  数据审计和smoke证据已seal。authorization SHA256：
  `5006e7e680e80d7dfae2297973e2fcddd22e1999e8af08ff27704c421cc07062`。

## 队列与检查点

研究配置：[study YAML](../configs/valais_matched_study_v1_20261002.yaml)。
协议：[匹配对照协议](VALAIS_MATCHED_BASELINES_PROTOCOL_2026-10-02.md)。
研究目录：`runs/valais_matched_study_v1_20261002`。

队列依次执行：B1独立100轮→同一Sierre clean/45条件验证→B0-E独立100轮→相同
验证→实际source顺序/初始化/曝光匹配核验→预声明描述性保护门与RESULT.md。
技术失败、守卫停止、输入/源码seal变化即停止余下队列，不自动重启或续训。
原模型、损失、算子和训练器保持；不能在这批运行中编辑被seal的源码/配置。
新机制、采样/损失实验及任何test仍需独立方案，不随本批启动。

实时先读研究目录 `status.json`；训练过程读
`logs/b1_training.stdout.log` 和对应stderr，随后B0-E同名日志；epoch checkpoint
只在完整epoch后出现。启动时queue PID6700、B1 worker PID13100，分别创建于
23:07:46/23:07:49；之后必须以创建时间和命令行核对，不能只认PID。
进程身份、任务XML和注册信息均保存在execution_started/process/registration文件。

任务无trigger、无retry、IgnoreNew、隐藏，最长72小时；该上限是任务运行保护，
不是可靠耗时预测。程序自行推进剩余已授权阶段；不要重复Start或另开GPU训练。
已用LEVIR test不再开发，Martigny test保持封闭。

完成须有研究目录completion.json、comparison.json和RESULT.md，且训练与两项评估
均有匹配的guard完成产物。单seed/Sierre单城筛选通过也只支持另列独立seed确认，
不提供空间独立置信区间，不自动授权校正或局部动作模型。
