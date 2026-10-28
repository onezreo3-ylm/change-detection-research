# P1 A-S第96轮后再次人工恢复

2026-10-19，用户在19:13主动暂停后明确要求“恢复并继续”，解除该暂停。
仅完成原A-S的97–100轮及既定可行性验证，C-S100轮结果保留，不新增训练预算或超参数。

入口：configs/valais_p1_resume_as_epoch96_20261019.yaml；状态目录：
runs/valais_p1_resume_as_epoch96_20261019。复用未修改的scripts.resume_valais_p1_as，
原trainer、模型、配置、FP32/CUDA、初始张量、采样/扰动实例、优化器/scheduler、loss、
阈值和首次最高clean-val F1选模均保持。新配置只指定本次保存轮数、暂停收据和独立输出。

previous_recovery字段按现有loader继续指向已封存的C-S94恢复链；实际最近的A-S73恢复链
另行完整核验，且其owner/status/authorization/训练与队列日志复制到新saved_prefix/prior_as。
本次暂停收据所列6个owner/训练/数据worker须重新逐个核对PID创建身份和退出状态。
A-S当前guard_resumes/session_0001完整复制并核验；不以更早的原始guard代替本次会话。
上述额外链证据、人类新授权、暂停收据和checkpoint哈希核验均加入saved_prefix，
在启动前由现有seal一并绑定。原脚本、旧YAML、旧封存和暂停现场不修改。

恢复准入：检查96轮history/actual sampling/优化预算、best选模、模型/优化器有限值、
完整Python/NumPy/Torch/CUDA/loader RNG和原train/val数据身份；生产trainer仍全量核对
train/val。完整unittest及两次真实96轮保存状态的下一CUDA更新须通过，模型/优化器/
loss/RNG逐位一致，每次4 source、8 pair views、1成功更新和1 BN更新。
这些smoke属于工程验证，不是研究效果，也不写入研究checkpoint。

原A-S已提交264672次成功更新、2116800 pair views；剩余4轮、11028成功更新、
88200 pair views，最终仍为100轮/275700更新/2205000 pair views。
暂停前97轮日志确认至少1100步，精确丢弃步数未知，全部从96轮checkpoint重算。
这些丢弃计算、恢复smoke和旧C-S95轮重算均额外披露，不宣称两行总物理开销相同。

使用原85°C确认/86°C立即停止规则，恢复会话独立存为A-S guard_resumes/session_0002。
一次启动额度，不自动重试。完成后核对两行采样、预算、guard/hash，再显式适配新flavor
的原45条件val，随后固定64个Sion train样本的残余耦合诊断。已有C-S clean改善门失败
仍有效，不能由可恢复性或P0现象宣称创新或方法收益；P2仍须原冻结证据门。
test继续关闭。恢复此前每90分钟托管核查，健康时静默；新的人类停止/改向始终优先。
