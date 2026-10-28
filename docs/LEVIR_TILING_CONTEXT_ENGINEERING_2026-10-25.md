# 窗口上下文诊断：核心实现与身份准备完成

2026-10-25，Asia/Shanghai。接续本窗口每小时托管，本次只推进已冻结窗口诊断的工程步骤。
**四臂推理核心、训练源隔离和身份准备已完成；真实CUDA probe尚未启动。**
独立CPU owner、温度guard及watchdog的本probe适配仍待完成，当前没有真实推理启动授权。
这是工程里程碑，不是新的研究结果、归档输出的真实CUDA重放通过或方法准入。

## 实现范围

- `scripts/tiling_windows.py`：严格窗口覆盖、双时相同坐标裁块、FP32概率累积/均值、共同支持、
  逐源及池化计数、逐像素归档重放检查、数值参照与双模型资源门。
- `scripts/levir_tiling_probe.py`：只枚举固定train16路径的Dataset、冻结身份准备、四臂执行核心、
  原batch32二值图/计数重放入口、统一batch4比较、模型参数/缓冲不变核对及重复/时相交换检查。
  CLI目前只有prepare和显式CPU synthetic smoke，没有可绕过监督的真实推理launch命令。
  真实执行函数还要求另行封存的authorization和有效监督检查回调；当前不存在该授权文件。
- `tests/test_tiling_windows.py`：22项覆盖/融合/分母/同像素计数/身份与split隔离检查。

不修改pcdc原模型/数据/指标实现；28个pcdc源文件逐项与10月24日原区域诊断准备hash完全一致。
原YAML中的实际模型、FP16、阈值、窗口起点、共同中心、3168主patch-pair和4032像元等量预算均保持。
原始256网格归档重放必须逐像素与计数同时精确通过；相同总F1不能替代。
另一组窗口未覆盖区域保持NaN，不能把缺预测位置计作背景；重叠臂先平均概率再阈值。
共同中心与全图支持分别计分，逐源先记混淆计数再池化，零分母按新诊断YAML记null，旧指标实现不改。

## 已完成验证与一次CPU修复

首次完整测试492项通过，但第一次显式CPU FP32 smoke失败。已确认本地torch在进入
`autocast(device_type=cpu,dtype=float16,enabled=False)`时仍检查不支持的CPU dtype并抛错。
未发生真实CUDA probe、换设备或BF16替代；失败源码、配置、日志和合成fixture原地保留。

修复为CPU FP32显式使用nullcontext，只有请求CUDA FP16才进入autocast；另加一项禁止CPU路径调用
autocast及拒绝CPU FP16请求的检查。新smoke使用独立目录，原失败输出未覆盖。
代码改变后重新运行完整 `python -m unittest discover -s tests -v`：**493项、102.514秒、全部通过**。
最终CPU smoke通过：原ResNet18、FPN宽64，四个缩小的合成窗口臂共30个pair预测，
参数/缓冲hash不变，优化更新0。其结果不证明真实数据、原FP16或真实归档重放已通过。

测试日志在 `runs/levir_tiling_context_engineering_20261025`。
首次完整测试、失败smoke、修复后的完整测试及smoke的全部计算继续扣入原共享账本，未因新目录清零。

## 实际数据及checkpoint准备

先完整验证16源白名单、canonical train/A/B/label路径及无别名，再读取任何选定原始文件。
仅对这48个train文件进行内容hash及1024尺寸头部核对；解码16张train mask并与既有缓存逐像素相同。
两个冻结checkpoint仅在CPU上反序列化，核对原flavor/mode/seed/best epoch、协议、FP16、memory format、
有限tensor及选定train文件hash；未创建真实模型实例或前向，未读raw val/test/Martigny。
校验的是同16个原训练源，没有重新抽样，也未根据新的模型分数挑选样本。

产物为 `runs/levir_tiling_context_probe_20261025/prepared.json` 与source_snapshot。
状态明确为train_only_prepared_not_gpu_authorized；源码和输入清单已封存，当前不允许直接启动CUDA。
本轮原始train访问与前一轮只读保存预测的screen范围已明确区分，不能继续声称本轮没有读原始文件。

## 尚未完成的具体下一步

接入本probe专用的独立CPU owner、guard与watchdog，绑定PID/创建时间、GPU UUID、原精度、实时心跳、
2秒温控和外部墙钟/共享账本，完成其工程测试及最终授权封存。
不得重跑本轮已通过的同一核心检查来代替接线，也不能仅有batch间温度回调就启动真实probe。
监督模块若有代码变化，再按AGENTS完成必要完整测试和smoke；使用新的检查输出目录保留本轮证据。

通过监督与预算条件后才串行做固定首源512窗口pilot，再按原冻结协议执行唯一一次train-only四臂诊断。
没有原始开发val、held-out、长训练、新seed、归一化替换或旧失败方向的追加授权。
原48小时共享预算继续累计；本次没有占用新probe的3600秒真实GPU运行配额。
