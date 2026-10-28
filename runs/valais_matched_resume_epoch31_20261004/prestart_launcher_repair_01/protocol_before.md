# Valais B0-E：第31轮暂停后的恢复协议

用户在已核验保存的暂停后要求“可以继续”，授权完成原B1/B0-E队列的剩余阶段。
本次从原B0-E第31轮checkpoint恢复第32至100轮，再做同一Sierre clean/45条件验证与比较。
已完成的B1训练、B1验证、原授权和暂停证据保持原样。原一次性任务继续禁用；恢复使用新入口。

## 不变的研究设置与恢复状态

原study YAML、两项训练YAML、训练器、模型、损失、算子、初始化、seed42、数据切分、
FP32、batch4、eval32、num_workers4、归一化、阈值0.5、100轮预算、选模规则及预声明门不变。
原checkpoint载入model、optimizer、cosine scheduler、scaler、Python/NumPy/Torch/CUDA和
训练DataLoader随机状态；第32轮先前的部分更新不在checkpoint中，重新执行完整第32轮。
只增加恢复控制与顺序记录，数值训练仍调用原guarded_train(resume=True)。

暂停记录、原checkpoint及备份、history和封存输入在恢复前核对SHA256。
新代码、配置、此协议、launcher、完整测试日志、smoke与顺序重建产物另建authorization。
每个后续阶段重新核对不可变输入。原59文件seal不修改。

## 样本顺序证据的恢复

原B0-E wrapper未落盘的SHA256内部状态无法直接恢复。本次根据checkpoint内的已审计
训练manifest生成唯一、排序一致的train/name@x_y列表，只在内存重放PyTorch DataLoader的
随机采样，不打开影像或标签，不调用模型。

重放的每一个已完成epoch必须精确恢复保存的B0-E exposure_stream_sha256及源数、view数、
batch数、更新数和零跳步；clean instance字符串保持为原值“clean”。第31轮结束的loader
generator状态必须逐字节等于checkpoint中的rng.loader。完整100轮source顺序哈希及数量
还必须等于已完成B1的实际source_order记录，证明相同seed/采样规则的匹配顺序。

metadata重放使用num_workers0、无pinning，只为重建顺序；实际续训仍为原num_workers4。
重放配置不会写入研究训练YAML或checkpoint。与原四worker运行的每轮实际stream及最终
loader RNG一致是接受前提，不能以“应该相同”替代验证。

恢复worker把经验证的31轮前缀SHA256状态接到实际第32至100轮sample ID记录，逐轮核对
预期source顺序，最后同B1核对完整顺序、初始化、1102500 source、275700成功更新及
2205000 pair views。source_order_progress仅证明训练batch顺序，不证明本轮checkpoint已保存。
source_order最终只在guarded训练完成和匹配核验成功后写入。

## 守卫、测试与smoke

原中断guard记录完整保留；原守卫wrapper已支持temperature_guard_resumes/session_NNNN。
本次调用该入口建立新会话，不把旧monitoring快照伪装成完成guard。85°C连续确认或86°C
单次及传感器/身份/持久化异常仍停止；无自动重试或恢复。

恢复前运行全量python -m unittest discover -s tests -v，包含CUDA检查，不屏蔽CUDA。
新增测试对一次性顺序接续、singleton末批、saved stream/RNG篡改、未来实际顺序变化、
未完成状态及原训练器CPU数值精确恢复做工程验证。合成fixture不作为研究结果。

单独真实CUDA smoke从第31轮载入模型、optimizer、scheduler与scaler，对metadata重建的
第32轮首个真实train source batch执行一次FP32双view更新，核对8 views、1更新、0跳步、
一次BN更新及有限损失。该更新只在独立内存模型中，原研究checkpoint不写回。
smoke不声称完整逐位CUDA训练确认；正式运行由原训练器恢复全部RNG与真实DataLoader状态。

## 执行与结果

恢复入口：[script](../scripts/resume_valais_matched_study.py)。
配置：[YAML](../configs/valais_matched_resume_epoch31_20261004.yaml)。
会话证据：runs/valais_matched_resume_epoch31_20261004。
隐藏一次性Windows任务Codex-Valais-Matched-Resume-E31-20261004：无trigger、无retry、IgnoreNew，
最长72小时；技术失败或守卫停止即止后续阶段。

队列：B0-E32至100轮→训练/数据/实际顺序核验→同一clean与45条件验证→原描述性门→
原研究目录comparison.json、RESULT.md、completion.json。完成标记附恢复授权与暂停证据哈希。
峰值GPU allocation和wall time仅可记录本次恢复阶段；原前缀未测值不能补成“完整实测”。
单seed/Sierre单城限制不变；不自动启动B2、平衡、校正/门控、独立seed或test。
