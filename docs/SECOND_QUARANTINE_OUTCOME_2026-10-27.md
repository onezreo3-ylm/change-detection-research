# SECOND归档核查结果与停止记录

2026-10-27，Asia/Shanghai。SECOND尚未科学准入，未产生模型或独立评价结果。

作者公开ZIP的稀疏目录读取成功，外层大小3785055247字节，目录306字节、3个条目：
SECOND_train_set.rar、macOS元数据、SECOND_total_test.zip。外层没有README/许可文件；
该事实不证明嵌套训练RAR内也没有。目录片段SHA256为
`d32f4707a73c2305a8e4219c8777a29dd21be910a37b73559b325ad52d7aabac`，
不是整个ZIP的摘要。原始目录和界限见runs/second_zip_directory_session_20261026。

首次单个大范围训练member请求得到HTML，在写入任何RAR内容前停止。固定64MiB
分段修正成功接收前三段，第四段返回HTTP200、text/html、2040字节。公开可见文本明确为
Google Drive配额限制，而非数据归档。两次合计11个请求、159.360秒，应用读取
201335396个归档字节；保留的RAR partial为201317918字节，未达到完整成员长度，
没有通过最终CRC/长度/EOF校验，不得更名、解压、用作数据或宣称下载完成。

失败状态和公开错误页保存在runs/second_release_quarantine_chunked_20261026，
原失败保存在runs/second_release_quarantine_20261026。没有再次请求该归档，
没有打开嵌套test ZIP，没有像元或标签解码。部分文件只作失败现场保存。
错误页称高访问量可能需等待；这是服务提示，不是已确定的恢复时刻。

后续公开schema核查仅阅读作者文本。项目页仍把模型代码标为coming soon；
指标链接是Metric.zip，不是标签定义说明，本次未下载。页面仅明确white代表non-change，
未给出可据以编程的building数值/颜色及发布包路径规范。作者论文明确同一语义类变化
可以被标注，并报告2968/1694 train/test；不足以由两期语义不相等推导建筑变化。
本轮未核实数据专属许可，不把论文或别处实现的代码许可替代数据条件。

一手入口：[作者项目页](https://captain-whu.github.io/SCD/)、
[作者论文](https://captain-whu.github.io/SCD/SCD_files/Semantic_Change_Detection.pdf)。
搜索只返回第三方汇总等补充入口，未用其颜色表或许可作为权威。不存在已核验的作者代码
标签转换器，不据此声称任何SECOND模型结果已经具备可比性。

当前后续条件是归档服务的实际恢复或作者提供可验证的发布说明/数据来源；不循环更换
下载方法重试同一对象，不静默换第三方重打包数据。B0-E队列和原45条件评价继续，
共享48小时上限不变。独立数据证据缺口保持open，而不是把下载失败算作负效果证据。
