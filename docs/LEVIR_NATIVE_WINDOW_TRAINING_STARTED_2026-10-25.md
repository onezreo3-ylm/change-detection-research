# 原尺度窗口配对训练已启动

记录时间：2026-10-25 16:26:53，Asia/Shanghai。

用户明确要求“有提升吗，开启新的训练”。此前仅改变推理窗口没有稳定提升；本轮已完成新的工程、原数据/初始化核对及真实pilot，正式native512训练正在运行，grouped256匹配控制随后串行执行。尚无新训练完成后的提升结论。

协议：`LEVIR_NATIVE_WINDOW_TRAINING_PROTOCOL_2026-10-25.md`；配置：`configs/levir_native_window_training_20261025.yaml` 与 `configs/levir_native_window_runtime_20261025.yaml`。保持原模型、原FP16和原GPU，不添加模块或归一化。两个训练均为100轮、178000次尝试，输入像元等量和256区域的原损失保持匹配。原train445/开发val64准入，test和Martigny关闭。

## 已完成的启动检查

- 最终515项完整unittest通过，日志 `runs/levir_native_window_engineering_20261025/full_unittest_after_fixture_fix.log`。
- 原ResNet18、FPN宽64的显式CPU FP32合成smoke，两臂各一次有限更新，通过；合成结果只作为工程证据。
- 原train/val解码、全部内容hash、划分和原11个pcdc核心源码精确核对；原初始化tensor身份 `eb4fa2550e47d732c5b21acb6899ba5539856bee35b532da58006f56108001a8`；100轮共同源窗口顺序hash预存。
- 真实pilot共98次工程更新尝试、388次物理pair预测，完整模型/优化器/AMP/RNG状态重复精确，下一更新恢复精确，三次输入group前缀一致。pilot不计为正式研究训练，也未按val分数调方案。
- pilot完整owner墙钟53.359秒，温控最高61°C；owner/worker exit0，独立guard/watchdog正常收尾，无换精度、降batch或重试。

首轮完整测试发现两处新增测试调用漏传原loss构造参数；只修正测试fixture，原模型/损失实现和科学协议未改。失败日志、测试副本与74.9163秒费用保留。随后完整检查74.1843秒、CPU smoke2.9779秒、CPU身份准备31.6795秒，全部计入原共享账本。

## 正式训练与预算

native512 worker PID 35656、创建标识 134373901870793734，owner PID 9584。原GPU UUID `GPU-8ada1895-7628-bb88-1dda-c57679117fa2`；独立CPU owner、2秒温度guard和watchdog均运行。85°C确认、86°C立即停；30秒保护心跳、900秒无推进及AMP异常门保持。

本快照实际进度：epoch 1，阶段 `native512_train`，阶段step 1391/1780，正式累计尝试1391。阶段AMP跳步0。这只是启动时快照，当前进度以各session的 `progress.json`、owner/guard及训练status为准。

pilot冻结吞吐规则给出的保守预算为native512 41255.4秒、grouped256 40810.4秒，合计22.80小时。包含每臂6000秒验证/存储余量及1.25倍安全系数，不能当作精确完成时间。每个训练及先行工程/pilot最多24小时，本配对研究共36小时，并受原共享48小时剩余额约束；原账本不重置、不自动延长。

native512正常完整退出并核对保护、身份和预算后，才启动grouped256。完成后按预先冻结门比较两臂原clean-val计数；原B0-E/B1只作历史参考。当前没有通过新方法或效果门，不自动追加seed、512开发推理、扰动评分或阈值搜索。

启动证据为 `runs/levir_native_window_training_20261025/startup_verification.json`、`prepared.json`、`engineering_authorization.json`、`training_authorization.json`、`pilot_session/` 和 `native512_session/`。每小时托管继续绑定本窗口；秒级保护由本地独立进程承担。

## 首轮完整提交核对

2026-10-25 16:31:41（Asia/Shanghai）：已完成第1/100轮，1780次尝试全部成功、0次AMP跳步；共同group顺序与事前第1轮hash精确一致。原64源开发val的1024个256窗口、67108864像素计数完整；best.pt和last.pt各137281795字节已保存。训练已进入第2轮，当前累计3140次尝试。随后正常FP16跳步按原规则单列，未触发异常门；独立温控当前80°C、最高80°C。首轮F1属于早期训练状态，不与100轮成品比较，不宣称提升。具体核对见 `runs/levir_native_window_training_20261025/first_epoch_verification.json`。
