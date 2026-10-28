# 公开目录读取的一次有界会话修正

2026-10-26。原目录尝试第一步HEAD得到非归档Content-Type，已停止、保留记录；
归档body读取0字节。它不能区分HEAD行为、过期页面字段或匿名会话问题，不宣称已经证明某个原因。

本次只准入一次工程修正：新建内存匿名cookie jar，重新读取作者公开提示HTML并验证同一file id；
在该匿名会话中请求公开确认URL的`bytes=0-3`。必须返回206、归档Content-Type、正确Content-Range
和PK本地头签名，才继续128KiB尾部及最多4MiB目录。最多4个HTTP请求、每个15秒。
配置second_zip_directory_session_20261026.yaml在请求前固定。

不读取用户cookie、登录凭据或其他Drive文件，不跳过登录/CAPTCHA，不读member或解码影像/标签，
不下载完整归档。服务器返回HTML、忽略Range、内容不匹配即停止；该修正失败后不继续重试此入口。
目录元数据不是完整包校验或数据准入。旧LEVIR-test/Martigny、GPU队列及原预算保持。
