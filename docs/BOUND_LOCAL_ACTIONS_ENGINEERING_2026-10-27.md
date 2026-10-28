# 样本、输入、模型与参考绑定的局部动作接口

2026-10-27。新文件pcdc/bound_local_actions.py，不修改已封存统计重放原型、原模型、
原训练/评价源码或活动B0-E输入。此批仅CPU显式合成工程；真实探针另行准入。

接口要求原AdapterCorrectedDetector全部frozen/eval，持有并复制实际归一化A/B张量，
自行调用原encode_pair，计算原四层绝对差和两次基准stage2 adapter残差。局部动作只由
已固定的完整stage2校正差在指定Boolean区域替换，不从调用者接收第二批features，
不随动作集合重算残差。六个原GN捕获同一无动作参考。

身份包含有序train tile、原A/B文件hash、逐实例hash、actor/adapter checkpoint与协议hash；
另记录**实际归一化输入逐样本hash、实际加载模型张量hash**。外层真实loader仍须验证
所声明文件/checkpoint与加载值；这里不把调用者给的字符串当文件真实性证明。
每次调用必须携带同一session/reference binding；改次序、证据、模型版本或缓存会拒绝。
模型、缓存的完整值hash在close再次核验，覆盖绕过version的.data改写；只有成功close后
才能发布该批结果。异常也清理缓存，重放context自行保证移除hooks。

actual encoder/adapter/decoder调用次数分别记录。三个臂为原GN、原GN联合动作加输出
支持掩膜、共享参考重放。联合动作要求**传播后的卷积影响支持互不相交**。逐动作完整
解码后拼合仍是强现成对照，其K次解码不得藏去。输出mask不能证明区域内组合可加。
对完整stage2 mask的原GN结果需bit-exact复现原a1；原a0与无动作重放也必须bit-exact。

CPU需要明确engineering_only且AMP关闭。真实CUDA将固定原FP16 autocast，没有自动
CPU、FP32、其他模型或随机权重替代。CPU工程fixture明确随机权重，并在各臂前将
adapter最后层一次性设为非零随机值以实际施加动作，不构成学习或任务结果。
smoke采用原ResNet18/FPN64/GN8、2个128输入；单元检查部分使用较窄FPN32，仅作接口检查。

运行：`python -X utf8 -m unittest discover -s tests -p test_bound_local_actions.py -v`；
`python -X utf8 -m scripts.smoke_bound_local_actions_cpu`。
新目录拒绝覆盖。全项目测试需等当前真实CUDA训练退出，不能与旧CUDA测试并发。
