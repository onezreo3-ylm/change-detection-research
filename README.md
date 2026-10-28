# LEVIR-CD research — baseline, action diagnostics, and final closeout

## GitHub / Codex 续接入口（2026-10-28）

这是当前研究工作区的公开交接快照。**先读 [Codex 交接说明](docs/CODEX_HANDOFF.md)**，
再读 [研究规范](docs/RESEARCH_SPEC.md)；根目录 [AGENTS.md](AGENTS.md) 保存持续约束。
[新机器安装与验证](docs/PORTABLE_SETUP.md) 提供独立检出、依赖和合成 smoke 命令。
[证据目录](evidence/README.md) 说明原始文本记录、无损压缩文件和未上传资源的哈希清单。

最新完成状态为 2026-10-27：三种子 B0-E/B1/B2 开发集控制已闭环；
局部组合与时相放置两条固定候选均未通过效用门并已关闭；筛查准备通过 641 项测试。
**尚无已验证的新方法，论文未达到投稿就绪；没有获准启动的新训练。**
原 48 小时计算上限保留，历史已消费 LEVIR test 和 Martigny 不重新开放。
下文 9 月及 10 月初的“当前状态”均为按日期保留的历史记录。

把仓库链接和下面这段话交给下一位 Codex 即可开始交接：

```text
请接续 https://github.com/onezreo3-ylm/change-detection-research 。
先获取仓库，阅读 AGENTS.md、docs/CODEX_HANDOFF.md、docs/RESEARCH_SPEC.md
和 docs/PORTABLE_SETUP.md，核对 evidence/snapshot_manifest.json。
先汇报当前真实研究状态、已关闭候选、剩余预算和缺失资源，再按交接说明推进。
不要自动启动训练、重复一次性脚本、重开已关闭路线或运行 held-out test。
```

仓库文件能传递研究上下文；原聊天、登录凭据、活进程和本地模型文件不随 GitHub 链接迁移。
真实实验需要另行取得匹配的数据和权重，并通过原协议的身份与资源检查。


## 独立数据后续研究（2026-10-02）

已完成 ValaisCD 的 dev/train/val 候选预审、train/val 全量 0/1 标签与重复样本审计、
安全数据转换及真实 CUDA 单批工程检查。独立全 FP32 seed42 clean 基线已完成100轮，
并通过只读完成审计和受守卫独立 clean-val 复核：epoch53、Sierre F1@0.5为
0.4162248161980205。只使用 Sion train 与 Sierre val；官方 Martigny test 保持关闭。
原外层 PowerShell 完成门的失败仍保留，未回填原守卫完成标记。
[ValaisCD 数据准入](docs/VALAIS_NEW_STUDY_INTAKE_2026-09-29.md)记录来源与空间限制；
[本对话续接](docs/EXPERIMENT_CONTINUATION_2026-10-02.md)记录模型连接504中断与准确恢复点；
[续接执行结果](docs/EXPERIMENT_CONTINUATION_RESULT_2026-10-02.md)记录本对话已启动的受守卫validation评分；
[Valais v1扰动验收与验证协议](docs/VALAIS_CORRUPTION_V1_REVIEW_AND_EVALUATION_2026-10-02.md)
绑定train-only视觉判断、45条件验证清单和既有checkpoint。该单城开发结果仍不是
跨城最终测试结果，B1长训和校正/门控模型尚未启动。

本项目的检测模型为共享 ResNet-18 + FPN absolute-difference decoder。
后续实验已实现并检验固定动作与整图选择器，但预声明门没有支持学习式整图
选择器的增益；PSE / PCDC / CPC 或 MoE 不能据此被称为已验证创新。
研究约束见 [RESEARCH_SPEC](docs/RESEARCH_SPEC.md)，近邻与风险见
[创新性复核](docs/NOVELTY_REVIEW_2026-09-11.md)。

## 当前状态（2026-09-27）

Phase 7 已按预声明门停止整图选择器路线；R1/R1E 三种子复现保留普通增广
B1。单个预先选定的 B1 seed-42 checkpoint 在最终 held-out test 上的既有
一次性分数由独立 V1R2 只读审计恢复认证，但原始 V1 运行仍失败、缺少温度守卫
完成标记，绝不能改称正常完成。该测试集已用于最终报告，不得再用于选择或
调整模型。完整结果与下一证据门见
[研究收口与后续边界](docs/RESEARCH_CLOSEOUT_AND_NEXT_EVIDENCE_GATE_2026-09-27.md)，
内部写作稿见[负结果论文初稿](docs/MANUSCRIPT_DRAFT_NEGATIVE_RESULTS_2026-09-28.md)。
最新进展重评、独立数据候选预审与下一研究准入见
[2026-09-28 下一研究评估](docs/NEXT_STUDY_REASSESSMENT_2026-09-28.md)。
只读完整性检查可运行
`python -m scripts.verify_research_closeout check`。下文各日期状态保留为历史记录。

## 历史执行状态（2026-09-15）

训练集视觉验收已通过，随后依冻结协议完成了 B0、B1、B2 与必要的 B0-E
seed-42 正式训练/验证。45 条件验证结果为：B0 `0.8794148`、B1
`0.9023764`、B2 `0.9007421`、B0-E `0.8767160` corruption mF1；各正式
评估均由真实 CUDA 与温度守护完成并保持 `test_evaluated=false`。B1/B2 共享
初始化、采样扰动流、曝光量和 178,000 次尝试更新，B0-E 表明鲁棒增益不是
双视图曝光造成。

按 64 个原始验证影像对进行的 20,000 次成对 bootstrap 显示，B2 的
clean−corruption mF1 95% 区间为 `[0.0024115, 0.0038991]`，净增 FP/FN
区间也均严格大于零。因此 Phase 4 门只通过到“另行冻结和复核”的 Phase 5–6
动作收益诊断；校正器、门控、Phase 7–10 和最终 test 仍未自动开启。完整结果、
输入哈希和边界见
[Phase 4 冻结基线结果](docs/STAGE4_FROZEN_BASELINE_RESULTS_2026-09-15.md)。

## 准备批次归档（2026-09-13）

最新续接：视觉链路曾短暂可读，但本次复测再次省略图片，不能称为稳定恢复。
已完成12个新增训练源样本、2160个实例、108张补充对照表，145项测试及两类
CUDA smoke通过；真实扰动评分仍未启动。
[当前复测/补充QA记录](docs/VISUAL_QA_CONTINUATION_2026-09-13.md)包含证据和样例入口。

按用户复核后的 [十步新方案](docs/EXPERIMENT_PLAN_2026-09-12.md)，主线调整为
实际校正动作的收益与损伤监督；[公开近邻对照](docs/PUBLIC_PRIOR_ART_2026-09-12.md)
记录贡献边界，不再把 PSE/MoE/一致性组合预设为三个创新点。
本批只做开发保护和扰动基准准备，不启动新长训练或 held-out test。

`configs/levir_development_seed42.yaml` 保持 stable-b4 的科学和运行设置，显式新增
`evaluation.run_test_after_training: false`。该模式只审计/训练 train 与 val，
不需要 test 目录，不写 test 指标；独立 test 和 test dataset 构建会提前拒绝。
旧 YAML 默认行为及 checkpoint 推理签名保留，不能用新代码续训已完成的旧 run。
开发 checkpoint 未包含 test 审计时，即使改用 test-enabled 配置也不会静默补验；
最终测试需另行记录明确的审计协议。

九类×五档的候选配置为 `configs/corruption_protocol_v1.yaml`，操作定义、
时相/方向/空间分层、已知重合与验收边界见 [扰动协议](docs/CORRUPTION_PROTOCOL_V1.md)。
准备命令只检查 train/val 的既有数据指纹、生成固定 val 实例与训练图像样例，
不运行模型，不根据 val/test 分数选择强度，且拒绝覆盖已有输出：

```powershell
& 'E:/Download/anaconda3/envs/geosa-basa/python.exe' -m pcdc.prepare_corruptions --config configs/corruption_protocol_v1.yaml
```

本批的配置和样例仍标为 `candidate_pending_review`。下一道闸门是训练图像强度
复核及正式版本接受；之后才用既有 best checkpoint 做验证集退化评估。
实际已生成46,080个固定val实例、360次训练图像变换及18张样例；74项完整测试
和开发期CUDA smoke通过。图像内容未能返回给检查者，视觉验收尚未完成，
不能由数值不变量宣称强度合理。[本批执行记录](docs/PHASE2_PREPARATION_2026-09-13.md)
保存复核证据和样例入口。
后续已新增 [固定验证评估器](docs/FROZEN_VALIDATION_EVALUATOR.md)：支持只读预检、
原checkpoint的clean val兼容检查，以及受独立强度审核记录保护的45条件评分。
当前真实审核仍为pending；不会把无法读取的样例当成已通过视觉验收。
本次继续已通过112项全量测试与45条件合成CUDA smoke，真实clean val F1精确复现
为0.9059754546；仍未运行真实扰动评分。
[评估器本轮结果](docs/VALIDATION_EVALUATOR_STATUS_2026-09-13.md)保存全部证据。
原始 Phase 1 完成记录与所有历史 run 均保留，下面是其归档说明。

## 当前状态（2026-09-12）

Phase 1 的 100-epoch 稳健基线已完成并通过独立复核，最佳 checkpoint 为 epoch 69，
按 val F1 选出（0.905975）。固定阈值 0.5 的独立结果如下，指标均为 [0,1]：

| Split | Precision | Recall | F1 | IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 0.913574 | 0.898502 | 0.905975 | 0.828113 | 0.992174 |
| test | 0.913960 | 0.888505 | 0.901053 | 0.819923 | 0.990059 |

运行目录：`runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/`。其中
`val_independent_20260912.json`、`test_independent_20260912.json` 与
`verification_20260912.json` 保存独立指标及复核记录。独立 test 与训练末 test 的
四类混淆计数及五项指标完全一致；最佳 val 的混淆计数和指标也完全复现。
累计完整 epoch 的 train+val 用时 15229.817 秒（4 小时 13 分 49.8 秒），不含过夜
暂停、审计、checkpoint 写盘和独立评估。177939 次实际优化更新，61 次已记录的
AMP 跳步（61/178000，0.0343%）。最终 34 项测试与新的 CUDA smoke 全部通过。
仅是 seed42 的 clean baseline，不构成多种子、鲁棒性或 PSE/PCDC/CPC 的结论；
当时在 Phase 2 前停止；后续用户已复核新方案，当前仅开放上面的准备批次。
原训练和本地温控已结束，小时监控也已移除。

用户已要求按现有 LEVIR-CD 继续实验。原始数据和严格配置保留不动；真实基线使用
单独的 `configs/levir_local_author_floor255.yaml`，在读取时显式复现作者公开
loader 的 `uint8 label // 255` 语义。官方入口与数据局限见
[数据准备说明](docs/DATA_PREPARATION.md)。

训练、验证、测试、独立评估、best/last checkpoint、严格断点续训和 CUDA
smoke 已接通，34 个单元/集成测试通过（含 8 个温控辅助测试）。本地 `E:/thesis/datasets/LEVIR-CD` 含预期的 445/64/128 个原始
影像对，可切为 7120/1024/2048 块。但 **32 张 mask 有 709 个值为 156/254
的像素，严格标签审计仍不通过**。未改动原始数据，也未使用未记录的阈值化。
具体统计与已采用的独立解码协议见
[实验接入记录](docs/EXPERIMENT_STATUS_2026-09-11.md)。

作者兼容协议的全量审计已通过。原 batch-4/FP32 运行应加速要求在完整 epoch 25
后停止并保留，未完成 epoch 26 已丢弃；其中 best epoch 18 的 val F1 0.893736
只是中间验证值，不是最终研究结果。一次 fast-b8 尝试在 epoch 2 因旧 AMP 溢出
处理逻辑硬失败，产物和错误均保留且不续训。按用户“稳妥且高效”要求，最终的
stable-b4/AMP 100-epoch、seed42 运行从随机初始化独立开始，现已完成。运行状态与最终
指标以 `runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/status.json` 为准。
该运行于 9 月 11 日因用户报告电脑温度过高暂停在完整 epoch 41；9 月 12 日用户
明确要求继续后，已在源码/配置/全量数据核验通过后用原配置严格恢复至 epoch 42。
恢复前 GPU 为 44°C，未更改训练参数、源码或功耗设置。val F1 0.905834 是恢复
起点的历史中间验证值，最终结果见上表。恢复日志为同名前缀的
`.resume_20260912.stdout.log` / `.stderr.log`，旧日志保留。用户随后将温度上限放宽至
86°C 或合理值，最终采用 85°C 软件暂停线（驱动报告目标温度 87°C）。应用户减少
聊天上下文的要求，运行期间聊天每小时仅汇报一句当前 epoch；本地独立温控每 10 秒
采样，单次至少 86°C 或连续 3 次间隔 5 秒至少 85°C 时暂停，连续 3 次读温失败
也保护性暂停，不自动重启。异常先本地停训，下次小时检查时在聊天汇报。
已废止最初的 76°C 规则。不改驱动/固件或功耗；软件采样不是硬件限温，不能保证
采样间隙不越线，也不代表 CPU/显存/整机温度。

温控辅助程序为 `scripts/temperature_guard.py`，配置在 `configs/temperature_guard.yaml`，
状态在 run 目录 `temperature_guard_state.json`。它绑定训练进程的 PID、创建时间、
完整命令行与可执行文件；使用持有的 Windows 进程句柄暂停，不跟随复用的 PID。
训练退出后温控已自行结束。该运行预算已完成，不要再次续训。其他未来运行需要
明确使用新配置并重新绑定进程身份；勿手动启动重复守护。
辅助代码不在 `pcdc/` 训练源码中，现有 checkpoint 的严格续训签名没有被改动。

## 环境

依赖范围见 `requirements.txt`。本机已验证的解释器是
`E:/Download/anaconda3/envs/geosa-basa/python.exe`，对应 Python 3.11.14、
PyTorch 2.0.1 / torchvision 0.15.2、CUDA 11.7、NumPy 1.26.4、Pillow 12.0.0、
PyYAML 6.0.3。实际计算设备为 RTX 3060 Laptop 6GB。

终端 PATH 未配置 Python 时，可以在仓库根目录的 PowerShell 使用：

```powershell
$py = 'E:/Download/anaconda3/envs/geosa-basa/python.exe'
& $py -m unittest discover -s tests -v
```

这里只复用解释器，未安装、升级或修改这个已有环境。其 `pip check` 仍报告
既有 opencv-python/NumPy 与 openxlab/packaging 冲突；本项目不导入这两个包，
基线依赖与测试已实际验证。不要为本项目随意升级其他研究环境的 NumPy。
可在有明确 CUDA 安装方案后另建隔离环境，并保存新的环境记录。

## 数据与不变协议

每个 split 必须具有 `A/`、`B/`、`label/`，三者文件名一一对应。使用原始
LEVIR-CD，不是 LEVIR-CD+；不自动重新划分或 resize。默认 1024→256
不重叠切片，两时相采用相同 mean/std，随机初始化、无增广、阈值 0.5。
所有训练超参数在 YAML。`levir_local.yaml` 只改数据根路径，其余设置与
`levir_baseline.yaml` 完全相同，单元测试对此做比较。

所有 YAML 内相对路径以 **YAML 所在目录** 为基准；命令行路径以当前目录为基准。

```powershell
& $py -m pcdc inspect-labels --config configs/levir_local.yaml --output runs/labels_new.json
& $py -m pcdc audit --config configs/levir_local.yaml --full --output runs/audit_new.json
```

第二条在严格协议下会明确失败；作者兼容协议使用单独 YAML，不改变该诊断结果。
目录/数量/尺寸检查不等于已验证官方划分身份；全量审计也不能证明地理独立性。
全量审计逐文件存 SHA256，并按解码 RGB 内容排查跨 split 的相同双时相对
（包括交换 A/B）。

## 训练与评估

本机真实 train/val 测速表明：旧方案 13.24 train tiles/s；启用 4 workers、AMP、
channels-last 后，batch 4/8/16/32 分别为 51.40/57.66/52.35/45.63 tiles/s。
batch 8 的短测只快约 11.7%，却会把每轮优化器步数减半并改变 BatchNorm 统计，
所以最终选择 batch 4，不以占满显存为目标。更新代码后的 batch-4 复测为 51.61
tiles/s；真实长跑首轮为 2.71 分钟，按首轮简单外推 100 epochs 约 4.5 小时，
比旧运行每轮约 8.4 分钟快约 3.1 倍。测速不读取 test，且不是研究指标；完整记录
见 [实验接入记录](docs/EXPERIMENT_STATUS_2026-09-11.md)。

选定加速协议的真实研究命令：

```powershell
& $py -m pcdc train --config configs/levir_local_author_floor255_fast_stable_b4.yaml
```

该配置保持训练 batch 4，相对旧运行只显式改变 eval batch 4→32、workers 0→4、
FP32→AMP 和 contiguous→channels-last，其余科学协议保持不变。AMP 会改变数值
轨迹，因此它仍从随机初始化开始，不能接续或覆盖旧 run。GradScaler 确认的偶发
overflow 会安全跳过单步并降低 scale，执行/跳步数都写入 history；未确认的异常和
FP32 非有限梯度仍会失败，不会静默掩盖。

训练每个 epoch 只用 train 更新、val F1 选模；100 epochs 全部完成后，才加载
best 对 test 做一次预测评估。test 的文件完整性可在训练前审计，但 test 指标
不参与选模。不要把不完整训练、单个种子或跨论文分数差直接当作有效性证据。

每个 run 保存：

- `config.yaml`：解析为绝对路径的可重用配置；`environment.json`：版本、设备、种子。
- `source/` / `source_sha256.json`：本次 Python 源码快照及指纹。
- `data_audit.json`：数据清单、内容校验。
- `history.json`：逐 epoch 学习率、train/val BCE/Dice/total、五项指标、实际 optimizer
  step 与 AMP overflow skip 计数。
- `best.pt` / `last.pt`：模型、优化器、调度器、随机状态、数据及协议指纹。
- `status.json`：运行/完成/失败状态；`test_metrics.json`：最后的 held-out 结果。

独立验证与测试：

```powershell
& $py -m pcdc evaluate --config runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/config.yaml --checkpoint runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/best.pt --split val --output runs/levir_fast_stable_b4_val_independent.json
& $py -m pcdc evaluate --config runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/config.yaml --checkpoint runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/best.pt --split test --output runs/levir_fast_stable_b4_test_independent.json
```

Precision/Recall/F1/IoU/OA 全部为 [0,1]，基于整 split 累计混淆计数，不平均
tile F1。损失是按样本数加权的 batch 均值，Dice 先逐样本计算再平均。
评估拒绝变化的推理协议或数据内容，checkpoint 用 `weights_only=True` 加载。

断点续训（从最近完成的 epoch 恢复，不是从中断 batch 恢复）：

```powershell
& $py -m pcdc train --config configs/levir_local_author_floor255_fast_stable_b4.yaml --resume
```

禁止并发写同一个 run。普通训练拒绝已有输出目录；续训要求代码、数据、
训练设置、随机种子和运行设置全部匹配。改变 epoch 预算或中途换初始化需新
实验，不能冒充同一次续训。当前目录没有 Git 历史，故源码快照尤其重要。

## Smoke（工程验证，不是论文结果）

```powershell
& $py -m pcdc.smoke --config configs/smoke.yaml
```

此命令仅接受 `kind: smoke`，生成明确标记的 2/1/1 对 64×64 合成输入，运行
真实模型的一次训练/验证/测试和独立 checkpoint 评估。不会下载数据或权重。
默认 CUDA 不可用时直接报错，不切 CPU。重复执行请复制 YAML，明确改为新的
data.root 和 output_dir；不会覆盖已有文件。本轮已完成的 smoke 结果在
`runs/smoke_cuda_verified/run/` 与 `runs/smoke_author_floor255/run/`，首次发现
CUDA 计数兼容问题的失败记录保留在 `runs/smoke/run/`。它们均不是研究实验。
加速路径的最终 smoke 位于 `runs/smoke_fast_pipeline_final/run/`，并额外通过了
AMP + channels-last + 多 worker 的严格 epoch 边界续训一致性测试。
与最终稳健配置匹配的 batch-4 smoke 位于 `runs/smoke_fast_stable_b4/run/`；另有
强制 AMP overflow 回归证明跳步、降 scale 和权重不变的行为可审计。

## 代码入口与后续接口

`pcdc/config.py` 验证配置；`data.py` 配对、裁块、审计；`model.py` 基线结构；
`losses.py`/`metrics.py` 损失与计数；`runtime.py` 设备/种子/产物；`engine.py`
训练/续训/评估；`__main__.py` CLI；`inspect_labels.py` 只读标签诊断；
`smoke.py` 显式合成工程检查；`tests/test_baseline.py` 回归测试。

独立扰动核心为 `pcdc/perturbations.py`；实例/协议为 `corruption_protocol.py`；
样例和清单准备为 `prepare_corruptions.py`。它们位于 RGB 裁块后、归一化前。
尚未接入真实训练，Phase 1 的 `pair_transform` 默认仍为 None。
`SiameseChangeDetector.correct_features` 仍不做校正；新模型组件须等待后续研究闸门。
