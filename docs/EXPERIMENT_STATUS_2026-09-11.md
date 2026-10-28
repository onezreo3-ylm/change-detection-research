# Phase 1 接入与实验状态

记录日期：2026-09-11；最终复核更新：2026-09-12。本文件是实际检查记录，不把交接附件中的指令或创新声明
当作已完成事实。当前研究实施仍止于 Phase 1。

用户最新决定：按已有的 LEVIR-CD 继续实验。保留旧数据和严格配置不动，新增
`configs/levir_local_author_floor255.yaml`，显式采用作者公开 BIT_CD loader 的
`uint8 label // 255` 语义后继续 Phase 1 真实基线。

随后用户指出原方案耗时过长且显存利用不足，要求加快后续训练。加速改动被视为
一个新的、显式命名的训练协议，不把旧 checkpoint 冒充为同一次实验的续训。

## 接入结果

- 初始目录已有 config/data/model/losses/metrics/runtime，但没有训练入口、tests、
  README，也没有 `.git`。未重写已有模型，未改 backbone、宽度、初始化或损失。
- 已补训练/验证/测试、独立评估、日志、最佳/末次 checkpoint、源码快照与
  epoch 边界续训。严格续训的参数逐位一致性由中断/连续运行对照测试覆盖。
- 找到既有 `geosa-basa` 解释器；CUDA 11.7 / PyTorch 2.0.1 可使用 RTX 3060
  Laptop 6GB。没有安装/升级环境，没有回退 CPU、随机权重或其他模型。
  原配置本来就是 `encoder_weights: null`；ImageNet 是另一个需要记录的协议。
- `pip check` 存在原环境的两项非基线依赖冲突（opencv-python/NumPy，
  openxlab/packaging）。基线不使用这两个包；不宣称整个共享环境无冲突。

## 标签审计阻塞

接入候选：`E:/thesis/datasets/LEVIR-CD`，未修改任何原图或标签。
文件名/尺寸检查可通过，预计 tile 数 7120/1024/2048。
全量严格审计在 `train/label/train_18.png` 遇到 [0,254,255] 后停止。
随后只读扫描了全部 637 张标签：

| Split | 原始 pairs | 非 0/255 mask 数 | 值 156 像素 | 值 254 像素 | 总异常像素 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 445 | 24 | 144 | 486 | 630 |
| val | 64 | 3 | 9 | 1 | 10 |
| test | 128 | 5 | 34 | 35 | 69 |
| 合计 | 637 | 32 | 187 | 522 | 709 |

机器可读明细：`runs/label_encoding_20260911.json`，包含每张异常 mask 的 SHA256。
尚未完成全数据内容/重复对审计，不能由文件数推出官方划分已验证。
不能据 156/254 就认定文件损坏或标签一定应判为前景。

### 已采纳且单独命名的解码规则

数据集作者的 [BIT_CD loader](https://github.com/justchenhao/BIT_CD/blob/master/datasets/CD_dataset.py)
在 `label_transform == 'norm'` 时使用整数除法 `label // 255`；
[其 data_config.py](https://github.com/justchenhao/BIT_CD/blob/master/data_config.py)
为 LEVIR 配置了 `label_transform = 'norm'`。据此，uint8 的 156/254 都会变为
背景 0，255 变为变化 1。用户要求按现有 LEVIR-CD 继续后，本项目将该规则实现为
显式的 `author_uint8_floor255`；它只在读取时解码，不改原文件，并进入协议指纹。
这仍然**不是已证明这些像素的真实语义应为背景，也不能证明本地数据与原始下载
内容逐字节一致**。不能把本实验与 `>0` 或 `>=128` 的结果混为同一协议。

## 工程验证与实际运行

新增显式解码、加速运行时及兼容性回归后运行
`python -m unittest discover -s tests -v`：26 项通过，无跳过或失败。新增 CUDA
回归覆盖 AMP、channels-last、2 个 data workers 以及 epoch 边界恢复；中断恢复
与连续训练的最终权重和测试指标逐项一致。另一个 CUDA 回归强制制造 AMP 梯度
溢出，验证 GradScaler 只跳过该步、降低 scale、保持模型权重不变并记录跳步数；
FP32 非有限梯度或 GradScaler 未识别的异常仍会硬失败。

CUDA smoke 在严格确定性条件下完成前向、反向、val 选模、best/last 保存、test
和独立重载评估，主流程与独立评估的混淆计数完全一致。其配置为 `kind: smoke`、
64×64 合成图、1 epoch，不能用于 clean baseline 分数或论文效果结论。
本轮最初的 CUDA smoke 发现 torch 2.0.1 的 `_bincount_cuda` 不支持严格
确定性，随后将其替换为四类 int64 求和，指标定义和阈值保持不变；已增加 CUDA
严格确定性回归测试。没有用 `warn_only` 或关闭确定性绕过。新增的
`configs/smoke_author_floor255.yaml` 也完成 CUDA 端到端及独立重载检查；它仍然
只是合成工程检查，不是研究结果。

真实全量审计已通过，记录在
`runs/levir_author_floor255_full_audit_20260911.json`：445/64/128 对、
7120/1024/2048 tiles，保存所有 A/B/label SHA256，未发现跨 split 的完全相同
双时相对。该结论不证明本地副本与官方归档逐字节一致或地理独立。

原始 100-epoch、batch 4、FP32 运行于 2026-09-11 16:14（Asia/Shanghai）启动，
输出目录为 `runs/levir_clean_r18_author_floor255_seed42/`。应用户的加速要求，该运行
在完整 epoch 25 后停止，未完成的 epoch 26 被丢弃；`last.pt` 保留 epoch 25，
`best.pt` 保留 epoch 18（val F1 0.8937358261）。该值只是未完成训练的中间验证结果，
没有访问 test 做模型选择或最终结论。`status.json` 已标为
`stopped_protocol_superseded`，旧产物没有被覆盖。

### 加速基准与选定协议

吞吐基准只读取真实 train/val tile，不读取 test，也不保存研究 checkpoint；其数值
是工程测速，不是模型效果。每个候选使用相同 seed、数据、模型、损失、学习率和
阈值，测速 JSON 保存在 `runs/speed_benchmark_*.json`。

| 训练配置 | train tiles/s | val tiles/s | train CUDA 峰值 allocated |
| --- | ---: | ---: | ---: |
| 旧 batch 4 / 0 workers / FP32 / contiguous | 13.24 | 12.31 | 526.8 MiB |
| **稳健快速 batch 4 / 4 workers / AMP / channels-last** | **51.61** | **84.46** | 532.6 MiB |
| 快速 batch 8 / 4 workers / AMP / channels-last | 57.66 | 72.11 | 774.0 MiB |
| 快速 batch 16 / 4 workers / AMP / channels-last | 52.35 | 72.88 | 1274.7 MiB |
| 快速 batch 32 / 4 workers / AMP / channels-last | 45.63 | 71.85 | 2379.2 MiB |

短基准中 batch 8 的训练吞吐最高，但只比 batch 4 快约 11.7%；batch 16/32 虽占用
更多显存却更慢。用户随后明确要求不盲目追求速度，以稳妥且高效为先。因此最终
选择 batch 4：保留原方案每 epoch 的计划优化器步数和 BatchNorm batch 统计，
只使用 4 workers、AMP、channels-last 和更大的纯评估 batch 加速执行。

第一次 fast-b8 尝试保存在 `runs/levir_clean_r18_author_floor255_fast_b8_seed42/`。
它从随机初始化完成 epoch 1 后，在 epoch 2 遇到一次 FP16 梯度溢出；当时的代码在
GradScaler 执行标准动态退避前把它硬判为失败。该 run 的 `status.json` 如实为
`failed`，只保留完整 epoch 1 checkpoint，不续训、不覆盖，也不作为研究结果。
随后修复为标准且可审计的 AMP 语义：仅当 GradScaler 确认溢出并降低 scale 时允许
跳过单步，`optimizer_steps` 与 `amp_skipped_steps` 均写入 history；没有关闭检查。

最终配置为 `configs/levir_local_author_floor255_fast_stable_b4.yaml`，输出目录为
`runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/`。它保持 100 epochs、
seed 42、训练 batch 4、原始划分、256 非重叠切片、标签解码、随机初始化、模型、
损失、AdamW、学习率、调度器和阈值；显式改变 eval batch 4→32、workers 0→4、
FP32→AMP、contiguous→channels-last。新运行于 2026-09-11 20:14（Asia/Shanghai）
从随机初始化启动，绝不载入前两个 run 的 checkpoint。匹配该配置的 CUDA smoke
已完整通过，产物位于 `runs/smoke_fast_stable_b4/run/`；它只是合成工程检查。

稳健运行的完整 epoch 1 已落盘：train 147.000 s、val 15.625 s、合计 162.625 s，
train total loss 0.884726、val total loss 0.319768、val F1 0.778961。1780 个计划训练
step 中有 1 个经 GradScaler 确认的 overflow skip（0.056%，scale 65536→32768），
其余 1779 个 optimizer step 正常执行，epoch 2 已继续且 stderr 为空。按首轮简单
外推，100 epochs 约 4.5 小时；相比旧运行一轮约 8.4 分钟，实际约快 3.1 倍。
这些仍只是运行健康检查和中间验证值，不是最终模型效果。

epoch 2 随后也完整落盘：train 147.797 s、val 15.578 s、val F1 0.810377；有 2 次
overflow skip，scale 已降至 8192。前两轮合计 3560 个计划 step 中跳过 3 个
（0.084%），epoch 3 在该 scale 下继续正常运行，stderr 仍为空。监控会持续检查
跳步是否变为连续或异常高频；不能把可恢复的动态退避误报为额外训练更新。

2026-09-11 22:01（Asia/Shanghai），用户因电脑温度过高要求先暂停。只终止了精确
匹配该配置的训练进程；停止前 GPU 为 76°C、利用率 94%，随后降至 65°C，并继续
降至约 52°C。最近完整的 `last.pt` 与 `best.pt` 均为 epoch 41，best val F1
0.9058338584；epoch 42 仅运行到 train step 178/1780，未写入 checkpoint，已丢弃。
41 个完整 epoch 共记录 28 次 GradScaler overflow skip（28/72980，约 0.038%），
stderr 为空。`status.json` 已明确标为 `paused_user_temperature`，监控 heartbeat 也已
暂停，不会自动恢复。暂停状态不是最终结果，尚未运行 held-out test。

### 2026-09-12 用户授权恢复

用户明确发送 `continue` 后，于 14:49（Asia/Shanghai）使用原 YAML 加 `--resume`
从 epoch 41 checkpoint 恢复，首个继续训练的 epoch 为 42，总预算仍为 100。
恢复前 GPU 为 44°C；checkpoint、当前/归档配置训练签名、全部 pcdc 源码 SHA256
及入口重新执行的全量数据审计均匹配。模型、优化器、调度器、GradScaler（scale
32768）和随机状态一并恢复，没有重置训练或更改超参数。epoch 42 前 178 步的
日志损失与昨日中断前一致。只出现已知 TypedStorage 弃用警告，未见训练错误。

新日志为 `runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42.resume_20260912`
对应的 stdout/stderr 文件；旧日志与暂停状态副本均保留。恢复元数据在 run 目录的
`resume_20260912.json`。原监控已恢复为每 2 分钟一次，基于上次用户报告过热时的
GPU 76°C 设置保守运行暂停线：3 次间隔 5 秒的采样均不低于 76°C 时停止该训练，
保留最近完整 checkpoint 并等待用户决定，不自动重启。不修改功耗/频率/训练源码；
该规则非硬件安全规格，不是实时硬件保护，也不代表 CPU 或整机温度。

用户随后明确把上限放宽为 86°C 或更合理值。只读 `nvidia-smi -q -d TEMPERATURE`
报告 GPU Target Temperature 为 87°C；最终选择 85°C 软件训练暂停线，代替此前
的 76°C 临时规则。每 2 分钟采样，单次读数至少 86°C 时立即停止当前训练，或
连续 3 次间隔 5 秒均至少 85°C 时停止，保留最近完整 checkpoint，不自动重启。
没有修改硬件温控/功耗。目标温度不是安全上限，定期软件检查也不保证采样间隙
不会越线，不能据GPU核心温度推断CPU/显存/整机温度。

恢复后 epoch 42 已正常完成并保存 best/last 选择结果（best 仍为 epoch 41，
val F1 0.905834）；epoch 42 val F1 0.897007，仅为中间结果，目前继续 epoch 43。
本次未改训练源码或 YAML，故没有重复运行会额外占用 GPU 的 smoke/测试；严格
恢复入口的配置/源码/数据校验与首轮训练和验证已通过，最终仍需完整测试与独立评估。

### 2026-09-12 汇报降频与独立本地温控

用户要求温度正常时每小时只汇报到哪个 epoch，避免上下文膨胀。已将同一 heartbeat
改为每小时触发一次，正常也仅汇报一句进度，不再每两分钟唤醒聊天。为保留温控，
15:15 启动独立本地 `scripts.temperature_guard`（配置 `configs/temperature_guard.yaml`），
每 10 秒读 GPU 核心温度；85°C 连续 3 次确认（间隔 5 秒）或单次 86°C 触发暂停。
连续 3 次读温失败也保护性暂停。过热先本地停止训练、CPU 核验最近完整 checkpoint
并写状态，聊天在下一次小时检查时报告，不承诺即时聊天通知或硬件级限温。

守护进程绑定当前训练 PID 18396、创建时间、可执行文件、完整命令行与持有的
Windows 进程句柄，拒绝身份不匹配/重复启动，不跟随 PID 复用。只在该训练运行
期间存在；正常训练退出后自行结束。状态为 run 内 `temperature_guard_state.json`，
错误保留在同名前缀的 `.temperature_guard.stderr.log`。模型、训练源码/参数、设备、
数据和已完成 checkpoint 均未修改，未安装任何新依赖。

新增 8 个工程测试覆盖温度确认/复位、旧 76°C 不触发、86°C 单样本暂停、采样失败、
退出不覆盖状态和重复守护拒绝；完整 `python -m unittest discover -s tests -v` 共
34 项通过。只读实机 smoke 通过，另外用仅休眠的自建 Python 子进程验证创建时间
不匹配拒绝、正确句柄仅停止指定子进程；未停止真实研究进程。该测试是运行工程
验证，不是新增研究结果。

AMP 会改变数值计算，因此稳健加速 run 与旧的未完成 FP32 曲线仍不是严格同协议
结果；不能拼接两者的 epoch 或指标。后续 Phase 1 对照必须统一使用最终配置。

## Phase 1 完成与最终独立复核（2026-09-12）

真实 stable-b4 运行已完成全部 100 epochs，最终 `last.pt` 为 epoch 100；训练预算
完成后才加载 val F1 最佳的 epoch 69 checkpoint，对 held-out test 评估。运行的
`status.json` 于 17:17（Asia/Shanghai）变为 `completed`。源码 SHA256、训练配置
签名、完整 100 项历史与 checkpoint 均核对通过；未更改阈值或用 test 重新选模。

使用归档 `config.yaml` 和同一 `best.pt`，另外分别执行了一次独立 val/test 评估。
两次独立评估也重新检查对应 split 文件的 SHA256。全部指标和混淆计数与训练期
最佳 val / 训练末 test 精确一致：

| Split | Precision | Recall | F1 | IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 0.9135744391 | 0.8985018418 | 0.9059754546 | 0.8281125486 | 0.9921735227 |
| test | 0.9139599558 | 0.8885048049 | 0.9010526363 | 0.8199233794 | 0.9900592119 |

| Split | TP | FP | FN | TN | Pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 2530413 | 239381 | 285845 | 64053225 | 67108864 |
| test | 6075005 | 571900 | 762330 | 126808493 | 134217728 |

最佳 checkpoint SHA256：
`033a0aa14becabc97c60b368b6d7c2e49bc8b386daa61dcc77e761c81ae747f3`。
产物均位于 `runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/`：
`val_independent_20260912.json`、`test_independent_20260912.json`、
`test_metrics.json`、`verification_20260912.json`，没有覆盖早期或失败 run。

耗时统计只累计 100 个完整 epoch：train 13748.455 秒、val 1481.362 秒，总计
15229.817 秒（4 小时 13 分 49.8 秒），平均 152.298 秒/epoch。该累计不含过夜暂停、
数据审计、checkpoint 写盘、丢弃的未完成 epoch 片段及独立评估，也不把被替代的
旧 FP32/batch-8 尝试算进此运行。旧 FP32 run 的 25 个完整 epoch 平均 488.035 秒，
与新运行平均耗时的描述性比值为 3.20 倍；这不是隔离其他因素的正式测速对照。
训练末 test 的纯评估循环为 26.250 秒；独立 val/test 循环为 14.485/26.187 秒。

计划 178000 个训练 step，实际执行 177939 个 optimizer step，GradScaler 记录的
overflow skip 共 61 次（0.0343%）；损失保持有限，未出现未处理的训练错误。
仅有原环境已知 TypedStorage 弃用警告。温控进程随真实训练正常退出自行结束，
最后状态为 `training_process_exited`，温控 stderr 为空；不据最后温度声称全程峰值。

最终重跑 `python -m unittest discover -s tests -v`：34 项全部通过，无跳过/失败。
新的 `configs/smoke_final_verification_20260912.yaml` CUDA smoke 完成训练、验证、
测试及独立重载，产物在 `runs/smoke_final_verification_20260912/run/`，仅为合成
工程验收，不计作本节真实数据指标。小时 heartbeat 已删除，不再唤醒或自动训练。

结论边界：这是原始本地 LEVIR-CD、固定 seed42、随机初始化、batch4+AMP 和显式
`author_uint8_floor255` 解码下的一次 clean baseline。没有多种子置信区间，不能与
不同初始化、解码或预处理协议的论文分数直接归因比较；全量审计不证明本地副本
与官方归档字节相同或地理独立。尚未验证扰动鲁棒性、PSE、PCDC、CPC 或创新性。

## 下一步与研究闸门

以下四项是2026-09-12 Phase 1交接时的状态。用户随后复核新方案并要求执行下一步；
2026-09-13已完成开发隔离、候选扰动工具/清单和工程测试，但尚未通过强度视觉闸门。
最新状态见 [下一步执行记录](PHASE2_PREPARATION_2026-09-13.md)，不把旧的等待状态
误解为基线仍未完成，也不自动开放后续模型训练。

1. Phase 1 真实训练、独立 val/test 复核、完整测试与 smoke 已完成，保持结果归档。
2. 当前停止运行，等待用户复核，不自动续训、启动新 seed 或进入下一阶段。
3. 与用户复核真实基线后，才能进入 Phase 2 扰动协议。不得把运行失败当作
   降低数据完整性要求的理由。
4. Phase 3 后验证强增广/普通一致性的缺口，再复核 PSE/PCDC/CPC；关键对照与
   否证标准已在创新性复核文档列明。当前没有三项已证实创新，也不保证不存在未公开撞车。
