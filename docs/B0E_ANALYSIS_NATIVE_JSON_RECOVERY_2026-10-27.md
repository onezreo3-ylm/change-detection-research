# 原Windows文本格式的序列化恢复补充

2026-10-27。首版布尔类型adapter通过609项测试，但它以LF写bytes，原Windows
text writer以CRLF写出partial，因此严格字节前缀检查拒绝发布。原partial未改变。
只读诊断已确认第一处差异在byte1；恢复原CRLF后5799byte前缀完全相同。

此补充复用首版所有统计/身份重建函数与原冻结规则，只在新派生adapter中以原CRLF
文本格式写candidate，再以原始字节逐个匹配完整旧prefix。仅在验收与匹配成功后，
替换已确认无效的canonical partial；两版失败源码、日志、stage和partial全部保留。
统计数字、阈值、bootstrap/seed、输入、评分和原源码均不改变。新测试实际模拟
Windows text writer遭遇np.bool_时的partial，不放宽为语义相等或忽略换行的比较。

恢复费用继续累计到原analysis修复费用项，包括每次失败、诊断、测试与发布，
原3600/172800秒上限不变。没有新推理或新研究分数。
