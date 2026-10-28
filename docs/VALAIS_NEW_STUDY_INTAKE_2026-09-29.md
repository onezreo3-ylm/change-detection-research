# ValaisCD 独立研究数据准入与下一步协议（2026-09-29）

状态：**新研究数据工程准入完成，独立 clean 基线已启动**。本文只记录工程检查和运行启动状态，不包含新数据集的最终性能结果。旧 LEVIR-CD 的 Phase 7 整 tile 选择器继续停止，旧 held-out test 不再用于方法开发。此数据集只能按新协议独立实验，不能将其分数与旧 LEVIR 分数直接排名。

## 为什么优先 ValaisCD

- [EPFL-ECEO 数据卡](https://huggingface.co/datasets/EPFL-ECEO/ValaisCD)及[项目页面](https://manonbechaz.github.io/2Player/)说明：SwissTopo 2017/2023 年航空影像、建筑变化二值标注、原始 0.1 m 下采样至 0.5 m、预切成不重叠 256×256。地区是 Brig（dev）、Sion（train）、Sierre（val）、Martigny（test）；不同地区可用于检验空间迁移，但本地 ZIP 不含每块的可恢复地理坐标。
- 数据卡标示 MIT；[swisstopo 原始地理数据使用条款](https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices)允许使用、再分发与处理并要求注明来源。论文和再分发材料须同时注明 ValaisCD 作者与 **Federal Office of Topography swisstopo**。数据卡没有消除原始影像和派生标签来源的引用义务。
- 数据卡承认标签受影像拍摄与矢量更新时差影响，并以监督模型做过样本筛除。本文优先核查 ZIP 中全部提供的影像与标签，不反序列化来源不明的 .pkl，也不把筛除后的验证/测试当作无偏新总体。

## 已完成的准入核对

1. 固定数据仓库修订 dbf58176968d17c964cfda960a314b243f73ca8b，分别固定 dev/train/val 的 URL、长度、HTTP ETag 与发布方 Git LFS **内容 SHA256**。ETag 与内容 SHA256 不相同，不能互换；[配置](../configs/valais_candidate_preflight_20260929.yaml)明确分别记录。
2. [远程预审产物](../runs/valais_candidate_preflight_20260929/audit.json)只读取 dev/train/val ZIP 目录与 dev 的两组像素。三部分分别为 3136/11025/1715 个完整 2017/2023/labels 对；dev 样例是 RGB 256×256、标签 0/1。官方 test 仅曾以 HEAD 查看下载长度；未下载、列举其内部文件或读取图像/标签。
3. val ZIP 已完整下载，SHA256 为 709546544035a7c6158e9fe742e588a434839445b348e4cea09fadea6e10871e，ZIP 所有成员 CRC 检验通过。[全量 val 审计](../runs/valais_candidate_preflight_20260929/full_val_audit.json)解码全部 1715 对：标签值只含 0/1，1440 对没有变化像素，内部没有相同解码双时相影像对。远程预审的跨 split CRC/尺寸筛查为零组，但这不是跨 split SHA/地理重合审计。
4. train ZIP 已完整下载并通过发布方内容 SHA256 928eba45f4111803538dc857ea3b53480ae1007b58254ffb1e507bfcedc949f9。[全量 train 审计](../runs/valais_candidate_preflight_20260929/full_train_audit.json)解码 11025 对：标签只含 0/1，9979 对为空标签，内部无相同解码影像对。[跨 split 比对](../runs/valais_candidate_preflight_20260929/train_val_cross_audit.json)没有相同解码双时相影像对，但缺失地理坐标仍不能证明空间无重叠。下载采用可续接、独立分块、最终全文件 SHA256 核验；原始 ZIP 和先前中断的 partial 文件均保留。
5. 抽查 dev 的无变化及建筑变化样例，影像与二值标注可直观对照；两组样例不能证明全数据标注质量。影像的 TIFF 未提供每块地理坐标，文件名也未证实可还原原始幅面。因此不能把 256×256 块当独立来源作置信区间。
6. 已由[安全转换器](../scripts/prepare_valais_development.py)生成独立 prepared_v2 目录，仅有 train/val 的 A/B/label。现有读取器的[开发期全量审计](../runs/valais_candidate_preflight_20260929/pcdc_development_full_audit.json)通过，源文件数为 11025/1715；[真实 CUDA 单批 smoke](../runs/valais_candidate_preflight_20260929/real_cuda_smoke.json)完成一次有限的训练前向、反向、优化更新和 validation 前向，明确标为工程检查，官方 test 未访问。此前一次转换因未识别数据包自带的筛选清单而提前拒绝，保留空的 prepared_v1.partial；修复后的转换只忽略其 .pkl，不反序列化或使用它。

## 新实验必须单列的协议

- 用独立[Valais clean 开发配置](../configs/valais_clean_development_seed42.yaml)：共享随机初始化 ResNet-18/FPN、输入 256×256、RGB 归一化维持旧数值、标签严格 0/1、阈值 0.5、CUDA、100 轮预算、clean val F1 选 checkpoint、开发期禁止自动 test。**这是新的数据与标签协议，不是旧 LEVIR checkpoint 的续训或直接复现。**
- train=Sion、val=Sierre。Brig 只用于训练区 QA 和后续预先定义的诊断，Martigny 的官方 test ZIP 保持未下载、未列举、未推理。先把已验 SHA 的 train/val ZIP 安全转换为现有读取器要求的 A/B/label 目录，再做全量 audit 与单批真实 CUDA smoke。转换器禁止 test 输入与 ZIP 任意路径。
- 第一轮仅做 clean 基线可行性；若训练稳定且 label QA 通过，再单列 B1 扰动协议。LEVIR 45 条件的强度与效果不能自动移植；所有新强度先在训练影像验收。B1/B2 或局部动作必须保持同数据、初始化、预算和推理门槛，独立记录前向、AMP 跳步及成本。
- **局部动作仍有证据门：**先在新训练来源上比较不操作、始终操作和实际局部动作的收益/误伤，确认受益与受损共存且超越较优常数动作有足够空间；再在新 validation 冻结主指标及 clean/召回/新增漏检限制。没有来源地理 ID 时不使用 tile 级 bootstrap 冒充独立区间；需先取得作者空间映射、可靠重建空间块，或在多个独立地区重新设计评价单位。门不成立就停止，不直接训练 PSE/PCDC/MoE。
- 最终 Martigny test 只在方法、扰动、动作集、阈值、数据/标签协议与报告格式全部冻结后进行一次。此前不依据其图片、标签或分数选方法。

## 下一个可执行检查点

数据、标签、转换和工程 CUDA 检查已通过。100 轮独立 clean 基线已由[启动器](../scripts/run_valais_clean_guarded.py)启动，输出在[运行目录](../runs/valais_clean_development_seed42)，训练 PID 37556、温度守卫 PID 34872；启动时守卫已绑定准确进程身份，85°C 连续确认或 86°C 单次触发即停训。当前仅第 1 轮在执行，尚无最终 checkpoint 或研究结论。完成后只按预定 clean validation F1 选模型。正式扰动和局部机制仍需各自的新协议与证据门。
