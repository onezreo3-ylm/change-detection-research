# B0-E分析JSON序列化修复

2026-10-27。两组评分及实际四角色正常退出审计已完成，certificate已生成。
旧分析按原规则完成计算，在写analysis.json时遇到NumPy bool_无法JSON序列化；
原partial和失败日志必须保留，不重跑评分，不改旧分析源码、规则seal、阈值、bootstrap
次数/seed、历史七格或新两格。当前局部探针未启动，串行coordinator已停止且不重试。

新adapter仅将递归结果中的np.bool_变为同值Python bool。其他不支持类型仍报错，
NaN/Inf仍拒绝。复用旧完整身份检查函数、原paired_bootstrap及primary_gate，按原
64×3×2×45×4轴、20k及seed20261026确定性重建保存结果，不读原始数据或checkpoint。
必须验证重新序列化全文以原partial完整字节为前缀，确保失败前已写出的数字未改变。

先复制原partial、原源码/规则/config和错误日志到独立recovery目录并记hash；新的
adapter/测试/smoke/全项目验证通过后才发布完整analysis.json。仅替换该已确认无效的
partial，旧partial字节保存不变；任何已有有效JSON或条件CSV会被拒绝，不覆盖完整结果。
新结果明示recovery执行与源码，原统计规则seal保持有效且重新核对。

实际失败分析、修复验收、重新分析和协调开销计入原固定评价3600秒及共享172800秒；
先写逐阶段实测费用，再一次结算，不遗漏失败费用、不扩展预算。该修复不是研究效果、
新统计分析或允许更改决策的机会。
