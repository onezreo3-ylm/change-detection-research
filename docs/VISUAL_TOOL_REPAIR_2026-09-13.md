# 视觉读取故障诊断与待生效修复

最新状态：2026-09-13 14:59复测再次记录图像省略，不能宣称稳定恢复。
本轮没有重启共享代理或切换模型；详见 [当前复测与QA结果](VISUAL_QA_CONTINUATION_2026-09-13.md)。

日期：2026-09-13。状态：**已定位并写入配置修复，尚未完成运行中代理重载及图像实读验收**。

续接更新：2026-09-13 14时后的当前任务已多次收到真实图片像素，且对应检查窗口
没有新增网关省略事件；详见 [视觉恢复与训练集QA](VISUAL_QA_RESUMED_2026-09-13.md)。
以下保留故障时的历史诊断，不代表当前仍无法读图；本轮未重启或改写共享代理。

## 根因及证据

不是样例路径错误，也不是 PNG 损坏。故障在 Codex 和上游模型之间的
Cockpit provider gateway：它依据自己独立的能力配置，主动删除图片输入。

- `runs/phase2_protocol_v1_20260913` 的 102 张 PNG 全部通过 Pillow
  `verify()` 和完整解码：18 张 1581×977 RGB 联系表，84 张 256×256 原尺寸图。
  102 个 metadata 引用与 REVIEW.md 的 23 个链接均有效；72 张 L5 图的 RGB SHA
  与 `train_calibration.json` 一致。没有重建或修改样例。
- `view_image` 返回了 PNG 字节；原生 `node_repl.emitImage` 也成功返回图像块。
  当前任务的本地会话记录保留了 `input_image`，而不是图片省略文字。
- 下游收到的内容却是
  `[Image omitted because the current model does not support image input.]`。
  此精确字符串只在本机 `E:/Download/Cockpit Tools/cockpit-cliproxy.exe`
  中命中，而非所检查的 Codex 主程序、code-mode host 或 app.asar。
- `C:/Users/81062/.antigravity_cockpit/logs/codex-api.log.2026-09-13`
  明确记录 `provider_gateway_vision_omitted`、`model=gpt-6-astra`，以及
  `omitted ... image input item(s) for text-only model`。例如
  13:33:26 的请求记录省略了 4 个图像输入项。
- 当前模型目录和独立启动的 Codex `model/list` 均返回
  `gpt-6-astra` 的 `inputModalities=["text","image"]`。因此重复修改 Codex
  模型目录无助于解除代理的独立过滤。
  [官方 model/list 字段说明](https://learn.chatgpt.com/docs/app-server#list-models-modellist)。

本机当前代理为 localhost:49389，对应下述 bce7… sidecar。其
`manifest.apiKeys[0].providerGateway.supportsVision` 是 `false`，
此前没有模型级例外。当前服务商的持久配置同样关闭默认视觉能力。
已通过网关地址、模型列表和内存中的密钥相等检查确认对应服务商；未输出密钥。

附带发现：本次 `view_image` 为 PNG 返回了 `data:application/octet-stream`。
在核实 PNG 魔数后，显式改为 `data:image/png` 仍出现相同省略提示；直接发送
`image/png` 的原生图像块亦如此。因此 MIME 问题不能解释本次主要故障，
今后读取已确认的 PNG 时应使用正确 MIME，不能仅把图片 JSON 当文字输出。

## 已实施的最小变更

保留 provider 的默认 `supportsVision: false`，仅加入当前模型例外，
不切换模型、不启用视觉模型自动路由、不改服务地址或凭据：

```json
"modelCapabilities": {
  "gpt-6-astra": {
    "supportsVision": true
  }
}
```

修改位置：

1. `C:/Users/81062/.antigravity_cockpit/codex_model_providers.json` 中当前
   `cmp_1782908798701_1` 服务商。
2. `C:/Users/81062/.antigravity_cockpit/codex_provider_gateway_sidecars/bce7c629cd3c3eb1efb0d580bb2205b4a9cf2adc9902c13be59f5d347afee9c6/manifest.json`
   的 `apiKeys[0].providerGateway`。

两个原文件均已在各自原目录备份为 `原文件名.before-vision-20260913`，
没有覆盖既有 `.bak`。逐对象比较确认每个文件只增加了上述例外，其他字段不变。
备份包含原有本机配置，**不要上传或把其原文贴入报告**。

该 schema 由代理 Go 类型元数据及 Cockpit 内置前端代码核实：
`modelCapabilities` 是 `map<string, {supportsVision: bool}>`，不是列表。
上层账户同步参数则是 `apiModelVisionSupport: {"gpt-6-astra": true}`；
不应把这些字段随意写入 `manifest.accounts`。

## 为什么仍需一次受控生效操作

代理 `loadManifest` 的直接调用位于启动函数；服务保存的是启动时的 manifest
对象。未找到受支持的 manifest reload 命令或热更新接口。写入上述文件后，
复测仍然出现省略提示，故不能宣布“视觉工具已恢复”。

此外 Cockpit 界面会缓存服务商配置；上层 API 账户由应用加密存储。
本次没有解密/改写账户文件，也没有猜测调用管理接口。只修改 sidecar 文件，
不能保证应用以后重新生成配置时继续保留例外。

下一步须在其他任务到达安全停止点后：

1. 在 Cockpit 当前 anyrouter 服务商（`https://anyrouter.top/v1`）的图像能力
   设置中，明确把 `gpt-6-astra` 加入视觉模型列表并保存，使当前 API 账户同步。
   不开启所有模型的默认视觉支持，不另选视觉路由模型。
2. 受控重启该转发服务，使其读取新 manifest；必要时重新启动 Cockpit。
   不要仅保存仍持有旧值的编辑器后就认定修复生效。
3. 重查生成后的 manifest 保留该例外，然后再次调用图像工具读取
   `calibration_tiles/sample1_B.png` 及至少一张原尺寸扰动图。
4. 必须收到可见像素并能描述实际画面，且对应请求不再出现
   `provider_gateway_vision_omitted`，才能判定工具端到端恢复。

诊断时另一个用户任务仍在运行，因此没有强制终止 Cockpit、Codex 或共享代理。
重启/账户同步及上述实读检查尚待执行；不得把配置写入成功等同于视觉验收通过。

## 项目工程回归

使用原解释器 `E:/Download/anaconda3/envs/geosa-basa/python.exe`，未安装依赖，
未替换设备；没有修改研究模块、训练配置、候选扰动或视觉验收状态。

- `python -m unittest discover -s tests -v`：109 项，108 通过、1 error、0 跳过，
  54.876 秒。日志：`runs/vision_tool_repair_unittest_20260913.log`。
  唯一 error 位于 `tests/test_frozen_validation.py:237`：测试只改内存中的
  acceptance 路径，却未同步源 YAML，触发 `pcdc/frozen_validation.py:181`
  的配置一致性保护。本次未修改该独立测试/研究模块，不能报告全套单测通过。
- `python -m pcdc.smoke --config configs/smoke_visual_tool_repair_20260913.yaml`：
  CUDA passed，独立 val 重载一致。日志：`runs/vision_tool_repair_cuda_smoke_20260913.log`。
  新 YAML 仅相对既有 development smoke 改 name/output_dir/data.root 三项，
  保留 CUDA、AMP、channels-last、严格确定性及 num_workers=4。
- smoke 为合成工程检查，train/val-only、`test_evaluated=false`，未生成 test
  数据或指标；未访问真实 held-out test，未开启真实训练或鲁棒性评分。

`configs/corruption_v1_acceptance.yaml` 仍为 `status: pending`、
`visual_review: not_completed`。文件解码、工具传图、人工/模型视觉审核是三个
不同验收层级，以上工程结果不能替代后两者。
