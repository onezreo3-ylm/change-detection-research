# 视觉链路复测与补充QA执行结果

日期：2026-09-13。本记录续接中断后的实际文件状态，不重做已完成工作。
**当前会话再次无法获得图片内容；不能称为已稳定修复。**

## 视觉状态：历史成功与当前失败分开

此前 [14时恢复记录](VISUAL_QA_RESUMED_2026-09-13.md)保留了实际画面观察，属于
当时的成功窗口。本次用原样例重新测试，常规传递和显式`image/png`图像块均收到：

```text
[Image omitted because the current model does not support image input.]
```

本地网关省略事件采样计数为154，最近一条为2026-09-13 14:59:18.471122
（Asia/Shanghai），模型标签为`gpt-6-astra`。此前成功窗口记录为138条。
这里只提取事件计数、时间和模型标签，没有输出请求正文、凭据或账户文件。
新事件与当前图像省略提示一致，不能用先前成功记录替代现在的端到端验收。

本次按 [OpenAI Docs图像分析说明](https://developers.openai.com/api/docs/guides/images-vision#analyze-images)
核对分析与生成的边界及PNG输入格式；官方能力说明不被当作本地恢复证据。
没有用生成图片替代科研输入，没有切换模型/服务商、改写共享代理配置或重启代理。
进一步修改并重载共享转发服务可能影响其他任务，需要明确授权及独立实读验收。

## 已推进：原参数不变的训练集补充QA

执行 [补充配置](<E:/new research/detection/configs/train_qa_supplement_20260913.yaml>)，
按既定seed=730194的训练文件名hash顺序，跳过原4个来源，新增12个源影像对各1tile。
选择ID先落盘，再读标签统计，不根据标签面积、模型分数或val/test表现挑样。

| 项目 | 已执行结果 |
| --- | --- |
| 新增训练源样本 | 12；其中5个tile有非空变化标注 |
| 覆盖 | 每sample/family/global-local均有A、B两个作用时相，五档完整 |
| 变换实例 | 2160 |
| 联系表 | 108张，展示L1/L3/L5 |
| PNG验证 | 432张原尺寸L5输出的RGB SHA全部一致；108张联系表完整解码通过 |
| 原文件 | 36个选中train的A/B/label文件前后SHA一致，最终复查也一致 |
| 冻结内容 | 原15个源文件、checkpoint、v1参数及val manifest/JSONL均未改动 |
| 像素不变量 | 标签、另一时相、alpha=0区域、输入不被原位修改全部通过 |
| 研究执行 | 未读真实val/test图像；未做模型推理或新正式训练；未写入接受记录 |

5个阳性tile变化像素为289、18685、9885、1024、5157（每tile65536像素）。
其90个local/L5/类型/时相组合中63个与变化标注至少有1像素交集，A侧32/45、B侧31/45。
这是生成后的空间覆盖统计，不等于目标完全覆盖或可观测性已证明。

文件名中的`rotation`只指联系表的行顺序循环移位，**不是图像旋转**。
轮次之间还可能改变多方向类型及mask/noise seed，不当作仅时相不同的配对实验。

[补充样例索引](<E:/new research/detection/runs/train_qa_supplement_20260913/REVIEW.md>)按6个批次
提供全部9类global/local图片及原尺寸A/B/标签链接；当前仍待真实视觉检查。

## 修复的工程问题

补充脚本原先把准备期15个源文件索引与当前完整17文件索引直接比较，新增的
独立评估器也会导致误拒绝。现改为：原15文件逐项严格匹配；另外记录本次完整
17文件索引并在结束时复核。没有放松任一原算子哈希，也没有修改冻结源码。
同时把原val实例JSONL纳入前后哈希保护，只读元数据、不读取val图像。
新增两个回归测试覆盖“允许新增独立文件”和“拒绝原文件改变/缺失/无效哈希”。

## 完整工程验收

- `python -m unittest discover -s tests -v`：**145项通过，0失败、0跳过**，63.784秒。
  [日志](<E:/new research/detection/runs/vision_continuation_full_unittest_20260913.log>)。
- 新的显式CUDA smoke通过：1 epoch合成train/val、best/last及独立val重载一致。
  [日志](<E:/new research/detection/runs/vision_continuation_cuda_smoke_20260913.log>)。
- 进程内温控入口完成45条件合成CUDA评分；guard状态与最后完成标记均为completed，
  summary与guard_state的提交哈希匹配。93次callback、6次温度采样，最高采样82°C。
  这不是全程峰值或硬件安全保证。温控不绑定旧训练PID、不停止其他进程、不自动恢复。
  [温控smoke日志](<E:/new research/detection/runs/guarded_evaluator_verified_cuda_smoke_20260913.log>)。

以上smoke均明确为合成工程检查；未产生真实鲁棒性结果。
只有已有TypedStorage弃用警告，未安装新依赖、换设备或降低数据完整性要求。
[机器复核记录](<E:/new research/detection/runs/train_qa_supplement_20260913/verification_20260913.json>)
保存运行证据、文件哈希和边界。此前14:27的温控smoke保留，没有覆盖。

## 当前停止点

`configs/corruption_v1_acceptance.yaml`仍为pending/not_completed。
真实45条件验证评分、B1/B2、校正器、门控、PSE/PCDC/CPC和held-out test均未启动。
下一步是恢复当前图像输入链路并实际审核补充图像，不能通过把状态改成passed来绕过。
本批没有留下训练、评分或温控后台进程。
