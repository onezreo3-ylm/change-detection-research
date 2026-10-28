# 下一步执行记录：候选扰动基准与开发保护

执行日期：2026-09-13；工作于2026-09-12开始，中断后续接。
依据用户在“后续实验流程详细一点”中复核的最新版方案，以及本任务
“现在按最新实验方案进行下一步”的请求。本批完成第1–3步的工程准备，
**强度视觉验收尚未完成**，未开始新的真实训练或鲁棒性评分。

## 已完成

- 固定 [十步新方案](EXPERIMENT_PLAN_2026-09-12.md) 和
  [六篇公开近邻对照](PUBLIC_PRIOR_ART_2026-09-12.md)。主线是实际校正动作的
  收益/损伤监督；PSE暂作诊断，MoE不加入。未实现校正器或门控。
- 新增显式开发配置 `evaluation.run_test_after_training: false`。训练/审计只用
  train和val；独立test和test dataset构建会提前拒绝。旧配置默认行为及旧
  推理签名保留。开发checkpoint没有test审计时，不允许静默补验后测试。
- 实现九个非几何RGB算子、独立随机流、时相对称选择接口、标签无关椭圆alpha，
  以及固定实例/样例准备工具。尚未接入真实训练。
- 候选参数、数学定义与限制见 [扰动协议v1](CORRUPTION_PROTOCOL_V1.md)。

## 实际产物与复核

目录为 [phase2_protocol_v1_20260913](<E:/new research/detection/runs/phase2_protocol_v1_20260913/status.json>)。

| 项目 | 实际记录 |
| --- | --- |
| val条件 | 9个命名类型×5档＝45，每条件1024tile |
| 固定实例 | 46,080条JSONL；实际行数、文件哈希与manifest核对通过 |
| 分层 | 每条件A/B各512、全局/局部各512；方向/时相/空间联合层固定且均衡 |
| 训练校准 | 4个不同训练源图各1tile，共360次变换、18张联系表 |
| 不变量 | 标签、另一时相、alpha=0区域不变；未原位修改输入；360次全部通过 |
| 原始内容 | 准备前核对4对train和64对val的A/B/label SHA；准备后再核对12个训练原文件，均未改变 |
| 源码 | 15个当前Python源文件与本批归档快照SHA全部一致 |
| test | 未读真实test图像/标签，未创建test实例，未重新计算test分数 |

原stable-b4配置和best checkpoint均保持原SHA：

```text
baseline YAML: 48edeabc313e97bc06c6073f435e16be794879e44bfbe5e7ce6fbdc327201e41
best.pt:       033a0aa14becabc97c60b368b6d7c2e49bc8b386daa61dcc77e761c81ae747f3
candidate:     98f5e62efb60fa0cefd17c0191522cb11b0083755f1215484206a8b5e8469296
```

候选SHA不是“验收通过”的标志。源配置、绝对路径配置、源码快照、环境版本、
每实例参数/seed/alpha几何和输出RGB SHA均已归档；旧实验产物未覆盖。

## 工程验收

使用既有解释器 `E:/Download/anaconda3/envs/geosa-basa/python.exe`，没有安装或
升级依赖，没有把要求的CUDA smoke换成CPU。

- `python -m unittest discover -s tests -v`：**74项通过，0失败、0跳过**，
  报告用时46.258秒。包含原baseline/CUDA确定性/续训测试、新开发隔离和扰动回归。
  [完整日志](<E:/new research/detection/runs/phase2_unittest_20260913.log>)。
- `python -m pcdc.smoke --config configs/smoke_development_20260912.yaml`：
  **CUDA通过**，AMP、channels-last、严格确定性；训练/val/best/last保存与独立val
  重载一致，没有生成合成test目录或test指标。
  [smoke日志](<E:/new research/detection/runs/phase2_cuda_smoke_20260913.log>)。
  配置名日期为创建日2026-09-12，实际执行于2026-09-13。
- 仅出现已有PyTorch TypedStorage弃用警告。以上全部是工程结果，不能作为
  clean/扰动鲁棒性或创新性研究证据。单元测试里的test字样指合成夹具，不是
  本轮重新访问真实LEVIR held-out test。

机器可读复核记录见
[verification_20260913.json](<E:/new research/detection/runs/phase2_protocol_v1_20260913/verification_20260913.json>)。

## 尚未通过的视觉闸门

已生成 [18张强度样例索引](<E:/new research/detection/runs/phase2_protocol_v1_20260913/REVIEW.md>)，
但图像工具未向两个检查者返回可读取的图片内容，不能声称已检查画面、小建筑
可见性或视觉伪影。打开预览面板也因连接关闭失败；文件本身已生成，索引可手动打开。

选定4tile的变化像素分别为0、2738、0、0（每tile65536像素）；只有一张包含
非空变化标签。即使视觉工具可用，这样的有限校准也不足以担保全训练集小目标
可观测性。可用训练图像补充预先固定的校准样本，但须记录选择规则与版本。

数值统计不替代视觉审核，`status.json`仍是
`completed_preparation` + `candidate_pending_review`，不伪称正式基准验收。
brightness暗化与shadow衰减在部分参数下数学同构，九个名称不等于九种独立物理
机制；等权mF1只代表此协议构造的汇总，限制已写入协议。

## 下一步

续接更新：用户要求继续后，固定验证评估器已实现，112项测试和45条件合成CUDA
验收通过，原best的真实clean val也精确复现。详见
[本轮评估器记录](VALIDATION_EVALUATOR_STATUS_2026-09-13.md)。视觉门槛仍未通过；
下文“实现评估器”部分已完成，但真实扰动评分仍在等待明确的强度确认。

先复核训练样例的强度/目标可见性，记录候选修订或正式接受。然后才实现并运行
基于现有best checkpoint的固定验证集退化评估，不用分数反调强度。
B1/B2、新训练seed、actor/gate划分、校正器、收益门控及最终test均未启动。
不自动恢复旧run或创建监控；本批命令已正常结束。
