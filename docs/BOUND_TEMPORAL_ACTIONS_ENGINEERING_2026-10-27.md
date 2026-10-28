# 互斥时间动作的绑定接口工程

2026-10-27。pcdc/bound_temporal_actions.py只扩展已封存BoundLocalActionSession，
不修改原文件、R18/FPN、训练源或已经运行的固定四区域探针。没有增加模型参数、
专家或门。真实研究与学习仍需独立协议、源码、计数/精度与预算准入。

父接口持有实际归一化A/B、样本/源/实例/checkpoint/协议身份与实际tensor hash。
临时只读hook捕获父接口本来就要执行的两次adapter输入/输出，随后立即移除，
从同一实际编码生成A-only、B-only及原both差分；encoder/adapter没有重复执行。
所有候选固定于这次观测，不接受外部同形状features替换，不随选择集合重算残差。
动作注册显式指定placement；同一区域的不同placement和其他影响支持重叠的动作
互斥，联合执行拒绝冲突。每次调用核验同一个reference binding、placement映射、
模型、变体版本与外部hook状态，close还核对全部变体完整值hash并清理cache。

8项CPU接口测试覆盖旧a0/both精确重放、encoder/adapter实际次数、所有两区域混合
选择、A/B交换、互斥/错误身份、variant/mapping篡改、绕过version的.data写入、
constructor后半段异常清理。单元fixture的窄FPN32仅用于接口检查；单独smoke使用
原FPN64/GN8及4区域全部256个实际组合，并通过新绑定接口验证三组all-placement
解码后拼合强对照。明确CPU/FP32/随机非零fixture、无数据/权重加载/优化，不是研究结果。

这是对已有参考统计/cache机制的操作适配，非新归一化或新对称性原则。三个变体可以
batch或共享计算，decoder调用数不是延迟下界。CPU检查不建立真实任务增益、CUDA
FP16身份、风险保证或新颖性。此前四区域二动作候选的效用失败保持关闭。
