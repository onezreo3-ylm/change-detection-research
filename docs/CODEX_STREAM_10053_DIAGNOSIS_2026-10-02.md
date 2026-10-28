# Codex 10053 连接中断：诊断与修复记录

日期：2026-10-02，Asia/Shanghai。调查对象是 Codex 连接中断，未启动科研训练、校正/门控模型或 held-out test 评估。

本机日志最支持的判断是：聊天中累计嵌入的大量 QA 图片使请求异常膨胀，触发 WebSocket 传输路径中断。Windows 返回的 10053 仅说明连接被中止，不能单独证明某个杀毒软件、代理节点或服务端组件是最终责任方。本次没有捕获网络包或读取代理内部日志，因此不宣称已确定某个组件的具体字节上限。

| 已核实的事实 | 证据与含义 |
| --- | --- |
| 聊天 `续接中断的实验进度`（`01a0fc9d-87f0-7542-962a-dceb836bac6a`）嵌入 28 张图片 | 图像 data URL 合计 87,491,048 字节；最大单张 4,422,286 字节。五次工具输出分别包含 1、6、6、5、10 张图片，最大一批 24,730,708 字节。 |
| 原生客户端反复发送约 88 MB 的请求 | 日志记录 `pre_compression_bytes=88017255`、`post_compression_bytes=65365170`；后续请求仍保留这些图片。图像占据了绝大多数体积。 |
| WebSocket 能建立连接，随后中断 | 21:36:42–21:37:03 的五次重试均报 `os error 10053`，21:37:11 切换至 HTTP。同一时间其他旧聊天的远程压缩也出现 10053。 |
| 常规 HTTPS 与短 HTTP/SSE 请求能完成 | 两种本机代理路径的 HTTPS 探测都获得响应；两次同模型短 SSE 探测均获得 HTTP 200 和 `response.completed`，输出 `OK`。探测不包含项目内容或工具。 |
| 当前模型请求使用官方接口 | 成功连接地址为 `wss://chatgpt.com/backend-api/codex/responses`；当前登录为 ChatGPT。Cockpit 的本地 `65467` 服务仍在，但没有证据将本次官方接口上的 10053 归因于该服务。 |

本次处理已完成：

1. 备份用户配置后，将默认 provider 改为 `openai_sse`。其名称为 `OpenAI`，保持 `wire_api=responses`、`requires_openai_auth=true`，设置 `supports_websockets=false`，使用 HTTP/SSE；不指定 `base_url`，继续由原登录方式选择官方接口。模型 `gpt-6.1-sol`、推理设置及其他配置保持原值。没有修改认证文件或系统代理。
2. 删除日志明确报告为忽略项的 `preferred_auth_method=apikey`，消除无效配置误导。没有添加已经移除的 `features.responses_websockets` 开关，也没有覆盖不可覆盖的内置 `openai` provider ID。
3. 添加 `scripts/prepare_chat_previews.py` 和 `configs/chat_preview_transport_20261002.yaml`，从明确指定的现有 train QA 联系表生成独立的 PNG 概览副本。单张限制 512 KiB，每批最多两张，data URL 总大小每批不超过 2 MiB；保存原图/副本 SHA256、尺寸和分批清单。拒绝 held-out 路径、源输出重叠和覆盖已有输出。
4. 将传输规则写入 `AGENTS.md`：不打印 base64，控制图像累计体积，完成阶段性查看后总结并压缩；压缩失败时先保留磁盘交接，停止继续堆叠图片。

缩小后的联系表**只用于导航和概览**。重采样会影响细节，不能用这些副本重新批准噪声、模糊或 JPEG 的科研可见性。需要判断细节时，逐张查看未改变的原图/原始 tile；原科研图片、协议、checkpoint 和原验收哈希均未由本次处理改写。

验证结果：

- 原生 app-server 的 `--strict-config` 和 `config/read` 验证通过：有效默认 provider 为 `openai_sse`，禁用 WebSocket、要求原 OpenAI 登录且无显式端点覆盖。
- 使用原登录和 `gpt-6.1-sol` 的短 HTTP/SSE 响应两次完成；配置写入后的探测耗时约 3.69 秒。这验证了短 SSE 链路，不能证明未来所有长请求或代理故障都已消失。
- `python -m unittest discover -s tests -v`：262 项通过，约 70.65 秒。其中新增五项检查原图保留、字节预算、输出保护、split 隔离和不可满足预算的失败行为；合成测试图片仅是工程检查。
- 预览生成及 smoke 检查：36 张，91,088,602 → 17,356,983 字节，下降 80.94%；最大单张 499,037 字节、最大分批 data URL 1,312,284 字节。36 份原图 SHA256 全部一致，所有副本 SHA256 均匹配清单。已检查一张副本的实际显示。
- 所有副本一起仍有约 23 MB data URL，**不要将整份清单一次性嵌入聊天**；遵循每批两张及累计 10 MiB 的工作区规则。

正在运行的旧聊天仍保留此前的大图历史和已创建的客户端连接。本次没有直接编辑 session JSONL/SQLite、删除聊天历史或中断其他聊天。待当前任务结束后关闭并重启 Codex，再加载需要继续的聊天，使连接配置对既有会话生效。若旧历史无法压缩，应从磁盘交接续接，避免持续重发约 88 MB 的历史。后续账号管理工具若改写 provider，需要重新核对有效配置。

可复核的文件：

- [当前 Codex 配置](C:/Users/81062/.codex/config.toml)
- [本次修改前的完整配置备份](C:/Users/81062/.codex/tmp/stream-diagnostic-20261002/config.before-215721.toml)
- [仅切换 SSE 前的备份](C:/Users/81062/.codex/tmp/stream-diagnostic-20261002/config.before-sse-220132.toml)
- [配置变更与哈希](C:/Users/81062/.codex/tmp/stream-diagnostic-20261002/config_change.json)
- [短 SSE 与有效配置验证](C:/Users/81062/.codex/tmp/stream-diagnostic-20261002/connection_check.json)
- [图像 payload 审计](C:/Users/81062/.codex/tmp/stream-diagnostic-20261002/image_payload_audit.json)
- [脱敏传输日志摘录](C:/Users/81062/.codex/tmp/stream-diagnostic-20261002/transport_log_excerpt.json)
- [预览传输清单](<E:/new research/detection/runs/chat_preview_transport_20261002/transport_manifest.json>)
- [预览 smoke 检查](<E:/new research/detection/runs/chat_preview_transport_20261002/smoke_check.json>)
- [完整测试日志](<E:/new research/detection/runs/chat_preview_transport_20261002/unittest.log>)

回滚连接配置时，先结束任务并关闭 Codex，再将「仅切换 SSE 前的备份」复制回 `C:\Users\81062\.codex\config.toml`。若此后已修改其他设置，先比较文件，避免覆盖后来的用户配置。传输副本位于独立目录，科研原图无需恢复。

配置依据：[OpenAI 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml)记录了 provider 的 `supports_websockets` 及重试参数；[高级配置](https://developers.openai.com/codex/config-advanced/)说明内置 provider ID 不能通过自定义表覆盖。本记录中的故障体积、时间和验证结果均来自本机证据。
