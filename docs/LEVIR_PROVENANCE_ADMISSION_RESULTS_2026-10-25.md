# LEVIR-CD 来源元数据准入审查（2026-10-25）

本次有界审查完成。**509/509 个原 train/开发 val 文件名能精确对应官方位置记录；这些记录只能作为名义位置索引，不能作为已经核实的原始场景分组，也不构成新训练准入。** 此结论不改变已结束的尾部损失候选及其停止门。

## 范围与可追溯版本

执行配置为 `configs/levir_provenance_admission_screen_20261025.yaml`。白名单直接来自已保存的 `runs/levir_tail_risk_control_20261025_iofix/data_audit.json`，仅取原 train445 和开发 val64 的 `entries.name`，未读取原影像或标签。先检查本地元数据文件名和数据根目录/允许 split 的元数据文件名，未找到地理对照；复用此前 `source_screen.json` 的来源筛查。

[LEVIR 官方发布页](https://justchenhao.github.io/LEVIR/)于 2023-03-11 公告补充坐标，并链接官方 STANet。固定仓库版本为 `6152d72f55df00a43d341580970be7a58c6b0cd5`；依据保存的官方目录 HTML 中 `currentOid` 固定，后续文件均从该不可变 commit 获取。GitHub API 首次返回限流403，改用公开目录 HTML；未循环请求 API。

来源文件为该版本的 [README](https://raw.githubusercontent.com/justchenhao/STANet/6152d72f55df00a43d341580970be7a58c6b0cd5/LEVIR_CD_spatial_info/README.md)、[JSON](https://raw.githubusercontent.com/justchenhao/STANet/6152d72f55df00a43d341580970be7a58c6b0cd5/LEVIR_CD_spatial_info/LEVIR_CD_name_coords.json)（62,276字节）和 [GeoJSON](https://raw.githubusercontent.com/justchenhao/STANet/6152d72f55df00a43d341580970be7a58c6b0cd5/LEVIR_CD_spatial_info/LEVIR_CD_name_coords.geojson)（345,880字节）。逐项 URL、获取时间、版本、原始传输 SHA256 和白名单副本 SHA256 见 `retrievals.json`。

公开元数据是合并 split 的文件，传输字节用于来源校验；只提取、保存和分析白名单509条记录，未保存合并文件，未检查白名单外的坐标、数量或空间关系。未访问本地封闭 test/Martigny 数据，未进行其映射或评价；本次传输的公开元数据与原始影像/标签访问应明确区分。为补充重复键与顶层 schema 核查，两个元数据文件各再次获取一次，固定版本的 SHA256 均相同。

## 精确映射结果

唯一规则是：已存清单原文件名 = JSON键 = GeoJSON的 `properties.name`。未按编号相似、重命名、位置或模型预测推断对应关系。

| split | 白名单 | JSON匹配 | GeoJSON匹配 | 覆盖率 | 缺失/重名歧义 |
|---|---:|---:|---:|---:|---:|
| train | 445 | 445 | 445 | 100% | 0 |
| 开发 val | 64 | 64 | 64 | 100% | 0 |

509个坐标框均为有限数且经纬度范围/上下左右顺序合理；两个格式的范围在6位小数舍入误差内一致，最大差 `4.994316071815774e-7` 度。509个精确坐标框各不相同；这项键与范围核查不能证明场景独立。逐样本结果见 `mapping.json`，汇总见 `mapping_summary.json`。

JSON 的明确字段说明为左上经度、左上纬度、右下经度、右下纬度，即本次记录的 `[west,north,east,south]`。GeoJSON README 示例有左右角文字注释不一致，本次只以实际点集范围核对明确的 JSON 字段语义。

## 可用范围与限制

- 所选 GeoJSON feature 的属性只有 `name`。两个文件均不提供采集区域ID、两期父场景ID、采集日期或切片血缘。发布页提到20个采集区域，但检查的元数据没有逐样本区域对应表。
- 509条 `Polygon.coordinates` 都直接放置一圈坐标点，缺少外层线性环数组；与 [RFC7946 §3.1.6](https://www.rfc-editor.org/rfc/rfc7946#section-3.1.6) 的 Polygon 嵌套要求不符。白名单副本保留原结构，本次没有修复几何、重投影或直接将其作为合格 GIS Polygon 导入。
- GeoJSON数值最多6位小数，JSON解析后的十进制表示为14–15位小数；这只是数字表示精度。发布说明未给出原始地理配准误差或独立的基准核实。RFC7946对标准 GeoJSON 的WGS84约定不能替代对该发布文件精度的验证；该标准也明确区分位数与不确定度。
- 精确文件名对应证明了索引层面的匹配；所检查的官方发布没有逐影像内容校验表，本次没有另行确证官方影像与本地像元身份。

因此，允许保留**可追溯的名义位置查询表**用于未来方案讨论；拒绝把现有位置表直接当作**已经验证的独立来源分组**。本次未进行距离聚类、区域推断、空间泄漏判定或重新划分。位置邻近不能单独证明泄漏，坐标框或文件名不同也不能单独证明场景独立。

仓库附有 [BSD-2-Clause LICENSE](https://raw.githubusercontent.com/justchenhao/STANet/6152d72f55df00a43d341580970be7a58c6b0cd5/LICENSE)；[数据发布页的 Usage](https://justchenhao.github.io/LEVIR/)另规定影像/标注供学术用途并受 Google Earth 条款约束。检查的元数据说明未另列专门许可。本次仅作内部学术来源审查，不扩大数据使用范围。

## 唯一外部资源与交接

需要一份**发布者确认的509个原 train/开发 val 文件名到原采集区域及两期父场景ID的对应表**，说明共享父场景与切片关系，并附可追溯版本或确认依据。资源应来自原始采集血缘的保管者；不能由AI、距离聚类或用户凭印象补写。尚未联系发布者。

当前已授权的训练、评价和本次元数据审查均已完成。后续等待这份可核实对应表或用户明确的新方向；不重复同一缺项检查，不为使用剩余预算追加实验。未来方案仍须另有可验证问题、合格数据、同架构普通对照及明确有限协议。

本轮GPU新增为0，无模型加载、训练或重新评分。公共账本保留 `30708.1026326` 秒（保守计费约8.53小时，非纯GPU kernel时间）。没有修改项目代码，因此未重复全量测试或 smoke。旧尾部候选、区域诊断和旧实验保持结束；独立真值的既有限制保持。

完整证据目录：`runs/levir_provenance_admission_screen_20261025`。`admission.json` 为准入决定，`verification.json` 为白名单/映射/边界核查，`artifact_sha256.json` 仅封存本次新证据和本报告，不重哈希旧训练成果。
