# SECOND发布ZIP目录读取范围

2026-10-26。承接已完成的公开HTML核查：作者链接的second_dataset.zip显示3.5G，页面为
大文件扫描提示而非登录页。其公开确认表单只用于此处的包头/目录请求，不执行完整归档下载。
本批在任何请求前固定configs/second_zip_directory_20261026.yaml：最多3个HTTP请求，
每次15秒，1次HEAD、最多128KiB尾部和4MiB central-directory Range。

先核对归档Content-Type/精确大小；Range必须返回206及完全匹配的Content-Range，
忽略Range或返回HTML就停止。请求失败不循环重试。只解析EOCD与目录记录，不打开任何member，
不读取/解码影像或标签，不提取文件，不运行模型。尾部传输可能包含未解释的压缩载荷，
不能把它误说为零归档字节传输或完整包内容审计。旧LEVIR-test/Martigny完全不访问。

目录元数据只用于确定split/命名/README或使用说明是否存在，不据此认证数据质量、
版权许可、地理独立或building编码。保存范围、输入和目录片段hash；这些不是完整ZIP hash。
本批工程代码可先做CPU fixture与受限读取smoke，全项目测试仍等活动CUDA训练结束后执行。
原共享48小时与原训练源码/配置/数据均不改，重叠CPU工作含在现有owner墙钟。
