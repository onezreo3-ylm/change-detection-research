# 作者训练归档的暂存与格式审查

2026-10-26。本协议在传输前冻结；本步是资源取得及来源/说明审查准备，不是SECOND科学准入。
公开目录已确认作者发布ZIP中只有SECOND_train_set.rar、SECOND_total_test.zip和系统附加文件。
实际HTTP Range元数据证明可只取训练member；不读取第二个test ZIP member。

按configs/second_release_quarantine_20261026.yaml，最多4次请求：新匿名公开提示HTML、
首4KiB本地ZIP头、同一central-directory片段、精确的训练member压缩字节范围。
每次读取超时15秒、整个暂存最多1800秒；至少需本member体积外另留2GiB可用磁盘。
源archive总字节、目录片段hash、训练member名称/尺寸/CRC在写入前核对。
Range被忽略、类型/大小/内容身份变化、超时或校验失败即停止，保留partial，无自动重试或换源。

只在CPU流式解开外层ZIP对训练RAR的deflate封装，校验本member长度、CRC32和RAR头，
计算完整训练RAR的本地SHA256。它不是整个含test ZIP的发布包SHA；不把CRC说成加密来源认证。
只把结果放入独立quarantine目录，不替换现有数据集、不提取RAR、不解码图像/标签、不调用模型。
完成后仍须单独核对数据使用说明、完整train清单、标签编码/变化掩膜、来源与train-only QA，
之后才能讨论数据准入；论文许可不自动覆盖影像许可，暂存不意味着已取得任何额外再发布权。

暂存的CPU/网络阶段与现有训练owner墙钟重叠，不并发GPU、不改活动账本、原48小时限额不变。
现有Python环境不安装/升级包；可用系统tar只在后续明确的list-only范围内检查RAR目录。
当前仍不打开任何旧LEVIR-test或Martigny文件。
