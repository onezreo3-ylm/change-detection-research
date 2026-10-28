# LEVIR-CD 官方来源与接入准备

更新时间：2026-09-11。用户已要求按现有 LEVIR-CD 继续实验；不自动下载、不改
原始标签。当前只需要原版 LEVIR-CD；暂不要求 WHU-CD、SYSU-CD 或预训练权重。

## 官方入口

- [作者项目主页](https://justchenhao.github.io/LEVIR/)
- [主页提供的百度网盘](https://pan.baidu.com/s/1fzNiOE7elGRmIo2h6MIhZw)，提取码 `l7iv`
- [主页提供的 Google Drive](https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim)
- [作者 STANet 仓库与数据目录说明](https://github.com/justchenhao/STANet#prepare-datasets)

链接来源已从作者主页交叉核实。百度网盘页面在当前浏览工具中打不开；Google
Drive 只返回动态页面壳，因此没有声称已验证网盘中每个归档可下载。
若网盘失效，以作者主页更新为准。不要把 Hugging Face/网盘转载镜像称为官方源。
主页上未见官方逐归档 SHA256；本地计算的 SHA256 只能用于留档与版本对照，
不能单独证明来自作者。

## 下载要求

下载 **原版 LEVIR-CD**（637 对、原始 1024×1024），不是 LEVIR-CD+，也不是
他人预裁切的 256 版本。保留官方 train/val/test 划分及原始归档，不自行打乱
或重新拆分。作者说明限制为学术用途，使用时按主页要求引用数据集论文。
[版本与使用说明](https://justchenhao.github.io/LEVIR/)

目前旧副本位于 `E:/thesis/datasets/LEVIR-CD`。建议把新副本放在独立目录
`E:/new research/detection/data/LEVIR-CD`，**不要覆盖旧目录**。该新路径已对应
`configs/levir_baseline.yaml` 的默认 data.root；`configs/levir_local.yaml`
继续指向旧副本，以便做对照。下载压缩包时应保留其原文件名、来源 URL 和日期。

期望解压布局（不是额外一层同名 LEVIR-CD，也不要混入 LEVIR-CD+）：

```text
data/LEVIR-CD/
  train/  A/ B/ label/
  val/    A/ B/ label/
  test/   A/ B/ label/
```

## 到货后验收顺序

1. 保存下载来源、日期、归档名及 SHA256；保留原始归档。
2. 验证实际 split 目录和文件名清单，不能只看总数。项目预期 445/64/128 对。
3. 验证 A/B/label 一一配对、RGB、尺寸一致和原始 1024×1024。
4. 先只读检查标签值，再与旧副本逐文件核对（尤其 32 张非二值标签）。
5. 若官方副本也存在 156/254，记录证据并确认统一解码规则，不自动阈值化，
   不反复重下载“期待修好”。若不同，进一步核对图像和全部划分，而不是只替换异常标签。
6. 解码协议明确后，运行全量内容 SHA256 与跨 split 重复双时相对检查。
7. 所有数据/协议通过，且用户要求开始实验后，才启动真实 GPU 训练。

准备好的只读命令（在仓库根目录运行；输出文件名需未被占用）：

```powershell
$py = 'E:/Download/anaconda3/envs/geosa-basa/python.exe'
& $py -m pcdc inspect-labels --config configs/levir_baseline.yaml --output runs/official_label_inspection.json
& $py -m pcdc audit --config configs/levir_baseline.yaml --full --output runs/official_full_audit.json
```

第二条只有在严格标签协议成立时才通过。当前旧副本 32 张 mask 共 709 个
非 0/255 像素，不足以证明数据损坏，也不足以认定重新下载一定必要。
真实运行改用单独的 `configs/levir_local_author_floor255.yaml`，其作者兼容解码
规则和局限见实验状态记录；严格配置仍用于诊断，不被静默放宽。
完整统计见 `EXPERIMENT_STATUS_2026-09-11.md` 与
`runs/label_encoding_20260911.json`。

## 准备验收记录

- 23 个 unittest 全部通过，含明确的 CUDA 严格确定性计数和解码策略测试。
- 两个 CUDA 合成 smoke 完成训练/val/test/best 保存及独立重载评估，仅是工程验证。
- 本地数据及原始标签未被修改；未自动获取数据；真实训练按独立解码协议继续。
- 真实数据准确性仍待官方副本对照；真实 256×256/batch4 的训练资源验收未完成。
