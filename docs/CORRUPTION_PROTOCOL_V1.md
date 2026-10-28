# LEVIR 切片扰动协议 v1：候选定义与复现边界

日期：2026-09-13。`schema_version: 1`；名称 `levir_tile_corruptions_v1`；
状态 **`candidate_pending_review`**。这些参数是本项目的候选选择，不是已接受的
正式基准或公开标准。本说明不宣称测试已通过、视觉校准已完成或获得了鲁棒性结果；
下文计数是当前默认配置规定的规模，实际执行证据由独立运行报告记录。

定义来源：[候选 YAML](<E:/new research/detection/configs/corruption_protocol_v1.yaml>)、
[RGB算子](<E:/new research/detection/pcdc/perturbations.py>)、
[实例/清单规则](<E:/new research/detection/pcdc/corruption_protocol.py>)、
[准备入口](<E:/new research/detection/pcdc/prepare_corruptions.py>)。
模型训练路线见 [十步计划](<E:/new research/detection/docs/EXPERIMENT_PLAN_2026-09-12.md>)。

## 1. 作用位置与统一数值顺序

保持原LEVIR-CD划分、`author_uint8_floor255`标签解码和原归一化。
在1024原图切成**256×256非重叠tile之后、模型归一化之前**，对A/B中恰好一个
时相的uint8 HWC RGB施加非几何扰动。另一时相与二值语义标签逐位保留；
不缩放、不裁掉边界、不旋转、不平移，不改原始文件。

令`x = RGB.astype(float64)/255`，`m`为H×W注入alpha，`T`为下节算子。
计算顺序固定为：

```text
float64 变换 T(x)
    → b = (1-m)*x + m*T(x)       # m沿RGB通道广播
    → uint8(np.rint(255 * clip(b, 0, 1)))
```

除JPEG编码器固有量化外，算子与alpha混合之间不先clip，不先转uint8；最终
只作一次clip与`np.rint`。之后才走原模型的float32和mean/std归一化。
alpha恰为0的像素还会显式恢复原RGB，确保支持域外逐位不变。

`none`、空alpha、brightness/contrast factor=1、gamma exponent=1、零色偏、
零haze opacity、零shadow attenuation、零noise sigma、零blur radius均返回
输入副本。**JPEG quality=100不是no-op**。即使走identity路径，也先校验参数
schema、数值、seed和alpha；不接受语义标签数组冒充浮点alpha。

## 2. 九个命名算子、方向与五档参数

所有数列按L1→L5排列；RGB偏移、噪声和混合量均在[0,1] RGB尺度下定义，
不是0–255整数单位。档位表示各自参数逐渐增强，不保证不同类型的同档损伤相等，
也不保证每张图像的任务损失随档位严格单调。

| 名称 | `T(x)`与单位 | 命名方向及L1→L5 |
| --- | --- | --- |
| brightness | `f*x`；`factor=f`无量纲 | darken：`[0.95,0.90,0.85,0.80,0.70]`；brighten：`[1.05,1.10,1.15,1.20,1.30]` |
| contrast | `(x-0.5)*f+0.5`；固定pivot=0.5，不按图像均值自适应 | reduce：`f=[0.90,0.80,0.70,0.60,0.50]`；increase：`[1.10,1.20,1.30,1.40,1.50]` |
| gamma | `x**g`；`exponent=g`无量纲 | darken：`g=[1.10,1.25,1.45,1.70,2.00]`；brighten：`[0.9090909090909091,0.80,0.6896551724137931,0.5882352941176471,0.50]` |
| color_bias | `x+offset_rgb`；每通道加性偏移 | `d=[0.02,0.04,0.06,0.08,0.12]`；warm=`(d,0,-d)`，cool=`(-d,0,d)`，magenta=`(d/2,-d,d/2)`，green=`(-d/2,d,-d/2)` |
| haze | `(1-q)*x+q*A`；固定`airlight_rgb=A=(1,1,1)` | neutral：`opacity=q=[0.05,0.10,0.15,0.20,0.30]` |
| shadow | `(1-a)*x`；`attenuation=a`无量纲 | neutral：`a=[0.10,0.20,0.30,0.40,0.50]` |
| noise | `x+sigma*Z`；H×W×3标准高斯场`Z`，sigma是归一化RGB标准差 | neutral：`sigma=[0.01,0.02,0.03,0.05,0.08]` |
| blur | 可分离高斯卷积；`radius`在本实现中就是标准差sigma，单位为tile像素 | neutral：`radius=[0.30,0.60,0.90,1.20,1.60]` |
| jpeg | 原uint8 RGB在内存中JPEG编码/解码，再转float64/255 | neutral：`quality=[95,85,70,55,40]`；固定`subsampling=2`、`optimize=False`、`progressive=False` |

高斯模糊不调用Pillow模糊近似。每轴一维核为归一化的
`exp(-0.5*(k/sigma)**2)`，半宽`ceil(3*sigma)`，依次沿高度、宽度卷积；
使用NumPy `reflect` padding，边缘像素不重复。当前五档核宽分别为3、5、7、9、11。
卷积在float64中进行，不改变像素网格、不在两轴之间转uint8。

noise只用局部`Generator(PCG64(seed))`，不修改全局随机状态。
JPEG从RGB数组新建图像，不继承源EXIF/ICC；其codec量化发生在alpha混合之前，
不能把JPEG当成无量化浮点算子。

## 3. 注入alpha：全局或局部椭圆，不由标签选择

`global`的`m=1`。`local`用规范化坐标椭圆；候选采样范围为：

- `center_x, center_y`分别均匀采样于`[0.30,0.70]`。
- `radius_x, radius_y`分别均匀采样于`[0.25,0.45]`。
- `feather=0.25`固定。

对于列`j`、行`i`，像素中心坐标为`u=(j+0.5)/W`、`v=(i+0.5)/H`：

```text
r = hypot((u-center_x)/radius_x, (v-center_y)/radius_y)
m = clip((1-r)/feather, 0, 1)
```

`r>=1`处alpha为0，`r<=0.75`处为1，中间线性羽化。先以float64计算几何，
返回float32 alpha；混合时再转float64。椭圆可在tile边界被截断，不会移动影像。
mask仅由规范化几何与局部随机种子生成，不读图像内容、真实变化标签或预测。

局部算子先在完整tile上生成`T(x)`，再按alpha混合；例如局部blur的卷积仍会
使用完整tile上下文。注入区域不是已识别的真实特征损伤区域；不能由alpha
直接断言其中所有像素都受损，或其中都没有真实变化。

## 4. 已知建模限制：九个名字不等于九种独立物理机制

brightness与shadow存在明确重合。相同alpha下，若`factor=1-attenuation`，
两者的全局和局部数学运算都相同；当前brightness的0.90/0.80/0.70与shadow
前三档的剩余亮度系数重合。局部清单按family分别产生mask，未必对应完全同一个
实例，但这不消除算子层面的等价。**本轮只披露该限制，不擅自修改候选参数。**

因此，九类不是九个独立物理因素；后述每类等权mF1是对本候选构造的
**协议加权汇总**，不是对真实退化发生概率或独立物理机制的无偏平均。

haze是固定白色airlight的简单alpha混合，没有深度、透射率场或天气观测验证，
不构成真实雾霾物理建模。局部JPEG是“整tile经过codec后再区域混合”的受控
干预，不是在原图局部单独运行的物理JPEG编码过程。阴影也不包含真实照明几何。
合成扰动保持标签并不保证严重条件下小目标仍可观测。

## 5. 可重放实例、种子与训练接口

当前固定种子为train=310942、val=310943、test=310944。test仅保留数值；
实例接口拒绝`stream="test"`，不创建test清单，不读test图像/标签，不评分。

sample ID形如`val/原文件名@x0_y256`。`keyed_seed`对规范化JSON作SHA256，
取前8字节的大端uint64；不依赖Python `hash()`或全局RNG。
实例seed包括协议名、split种子、stream、sample ID、family、方向、作用时相、
空间模式、epoch、view，**不含severity level**。mask另用`keyed_seed(seed,"mask")`。

因此，同一tile/family跨五档保持方向、时相、mask和标准噪声场一致，改变的
仅是该档参数；noise在重放时按不同sigma缩放同一个标准高斯场。不同family
不承诺共用mask。val不允许随训练epoch/view变化。

`sample_training_instance()`是未来集成hook：无状态地按sample/epoch/view
独立选择family、level、该family内方向、A/B、global/local，各自均匀采样。
它**没有挂接到当前训练实验**；不会因存在该函数就自动执行增广、B1或B2，
也没有替用户决定未来双视图前向与BN更新方式。

## 6. val清单：45个条件，联合分层恰好均衡

64对val原图产生1024个256tile。每个`family × level`条件为每tile分配
**一个**方向/时相/空间模式实例，形成45×1024=**46,080条**记录。
这不是让每tile遍历全部方向×时相×空间模式；不能将记录数按这些因素再乘一遍。

每个family将`directions × (A,B) × (global,local)`组成联合层。
按`keyed_seed(val_seed,family,sample_id)`排序tile，依次轮转分配联合层；
同一分配在五个level复用。分配顺序不看图像、标签或模型输出。

| family方向数 | 联合层数 | 每个level、每个联合层的tile数 |
| --- | --- | --- |
| brightness/contrast/gamma：2 | 2×2×2=8 | 128 |
| color_bias：4 | 4×2×2=16 | 64 |
| haze/shadow/noise/blur/jpeg：1 | 1×2×2=4 | 256 |

1024恰好可整除上述层数，因此各条件A/B各512、global/local各512；
方向也严格等量。若换数据规模，一般只保证层间相差不超过1个tile，必须另记协议。
清单保存`condition_id`、方向、目标时相、空间模式、参数、seed、mask几何，
并在manifest中保存condition与联合层计数。

准备阶段只生成val实例描述，不运行模型，也不生成val鲁棒性分数。
会读取val源文件以核对SHA256，但不据val图像/标签内容挑扰动强度。

## 7. 后续评分规则：先条件内混淆计数，再等权45个F1

保持baseline检测阈值0.5；checkpoint仍只按clean val F1选择。
对每个条件`c`，先累计该split所有tile的`TP_c, FP_c, FN_c, TN_c`，再计算
该条件的precision、recall、F1、IoU、OA。零分母遵循既有指标实现。

```text
F1_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
corruption_mF1 = (1/45) * sum(F1_c for each family/level condition)
```

不得平均tile F1，也不得把45个条件的混淆计数混池后计算一个F1来替代mF1。
同时单列clean指标与方向/时相/global-local分层结果，不由总平均掩盖分项。
每个命名family权重为1/9、各level为1/5；这包含第4节说明的算子重合与协议选择。
当前准备工具只把该规则写入manifest，**没有实现一次评分运行或报告任何成绩**。

## 8. train-only校准设计与局限

使用`selection_seed=730194`，先按训练源文件名的keyed hash排序，从445对中
选择4个不同源影像对；每对再由同源ID的keyed hash从其16个tile中选1个。
选择发生在读取标签像素之前，不按变化面积、建筑大小、亮度或模型难度挑例。

默认设计生成18张联系表（9family×2空间模式），每表4个来源行，展示clean A、
clean B、语义标签、注入alpha以及L1/L3/L5。缩略图仅展示时nearest缩至192，
实际算子始终处理256tile。五档都会计算和记录，所以设计规模为
4×9×2×5=360个校准实例；保存原尺寸clean A/B/标签与各组合L5输出。

每行方向按有序方向列表轮转；目标时相随行和global/local切换。该小样本展示
不是val的全联合均衡设计，同方向/场景组合覆盖有限。四对源图可能不足以涵盖
小变化、强阴影等关键情况，不能据此保证所有源影像在全部强度下都可观测。
标签可用于**选定之后**的视觉判断，但不反过来决定主基准mask或样例选择。

记录的不变量是标签不变、另一时相不变、alpha=0区域不变、输入数组未被原位修改。
数值诊断包括注入支持率、实际RGB变化像素率、平均通道绝对差、输出边界通道比例、
新增0/255边界通道比例。后两者不等同于“真实裁切率”或“信息可观测性”：
边界值也可能来自原始图像或量化，不能用这些数代替视觉审查。

必须先审查train-only强度样例及失败情况，再决定是否接受正式参数。
如需增补校准样例或修改参数，要明确版本、选择规则和理由；不能以val/test分数
反向优化严重度。`completed_preparation`仅表示准备流程结束，仍保留candidate状态。

## 9. 内容校验、版本与输出SHA

参考checkpoint固定为
[stable-b4 best.pt](<E:/new research/detection/runs/levir_clean_r18_author_floor255_fast_stable_b4_seed42/best.pt>)，
SHA256：

```text
033a0aa14becabc97c60b368b6d7c2e49bc8b386daa61dcc77e761c81ae747f3
```

准备入口核对checkpoint内容、推理/预处理/指标协议签名、与checkpoint绑定的历史audit、
train/val inventory；实际内容SHA核对仅覆盖选定的4对train源文件及全部64对val
源文件的A/B/标签。它不修改baseline、不重新训练、不读test源文件；读取旧
checkpoint/audit中已有的历史元数据不等于重新访问test图像或标签。

协议fingerprint包含候选设置、种子、checkpoint SHA和相关基座协议/运行参数；
路径可搬迁，代码hash另行记录。原YAML、归档基座配置、JSONL实例文件也各有
内容SHA。准备开始/结束及源码复制时检查变化；中途修改配置或代码必须失败，
不能将变化前后产物混成同一份已完成候选。

`train_calibration.json`的`output_rgb_sha256`是变换结果**HWC uint8像素字节**
的SHA256，不是PNG文件的SHA，也不是JPEG压缩码流的SHA。重放对比应使用同一含义。

JPEG像素可能随Pillow、libjpeg/libjpeg-turbo实现或版本变化。`environment.json`
记录Python、NumPy、Pillow、JPEG codec和libjpeg-turbo版本；结合源码快照、参数、
实例seed、原文件SHA与输出RGB SHA复现。不承诺不同codec版本逐位一致；
发生变化必须作为版本差异复核，不能静默沿用旧输出或旧分数。

## 10. CLI与产物位置

基座配置为 [开发期配置](<E:/new research/detection/configs/levir_development_seed42.yaml>)，
保持stable-b4学习/预处理设置，并设置`evaluation.run_test_after_training: false`。
候选加载器会拒绝训练后自动test的基座配置。

在项目目录`E:/new research/detection`使用已有Python环境运行准备命令：

```powershell
python -m pcdc.prepare_corruptions --config "E:/new research/detection/configs/corruption_protocol_v1.yaml"
```

这条命令只做CPU图像准备与内容校验，不是把CUDA研究推理切换成CPU。
它不包含训练、评分或test选项；没有模型前向。指定资源缺失或校验不符时失败，
不替换数据或权重。

当前YAML的输出根目录为`E:/new research/detection/runs/phase2_protocol_v1_20260913`。
以下是该目录下的**设计产物**，是否实际生成及通过检查由运行记录确认：

| 文件/目录 | 内容 |
| --- | --- |
| `candidate_source.yaml`、`candidate_resolved.yaml` | 原候选配置及解析后路径；状态仍是candidate |
| `base_config.yaml` | 归档的绝对路径基座配置 |
| `source/`、`source_sha256.json` | 源码快照及内容hash |
| `val_instances.jsonl` | 46,080条默认val候选实例，非预测结果 |
| `val_manifest.json` | 条件/联合层计数、参数指纹、输入/配置/实例hash、聚合规则；注明未创建test清单 |
| `train_calibration.json` | 样例ID、360个默认实例的数值/不变量/输出像素SHA、已核对源文件SHA |
| `previews/` | 18张默认train-only联系表 |
| `calibration_tiles/` | 256原尺寸clean A/B/标签及L5样例输出 |
| `environment.json` | Python、NumPy、Pillow、JPEG实现等版本 |
| `status.json` | preparing、failed或completed_preparation；不等于正式基准验收 |

输出根目录已存在时拒绝覆盖；失败产物保留为失败记录，不自动删除后重跑。
需要重跑或变更时另定输出目录并记录版本。原始数据、split文件、既有checkpoint
和原baseline结果目录保持不变。验收、单元测试、CUDA smoke与后续评分的真实
状态由主执行记录另行报告，本文件不预先填入通过项或研究结论。
