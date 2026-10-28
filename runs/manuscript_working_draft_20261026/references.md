# 参考资料与阅读范围

检索核对日2026-10-26；以下为一手来源。摘要/元数据阅读不等同于完整方法复现；不据此宣称全面排重。

1. Hao Chen, Zhenwei Shi. *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing 12(10),1662,2020. DOI:10.3390/rs12101662。核对[作者数据页](https://justchenhao.github.io/LEVIR/)中的来源、任务、许可及书目信息，未将数据页介绍当作本地空间独立性证明。
2. Hussein Mozannar, David Sontag. *Consistent Estimators for Learning to Defer to an Expert*. ICML,PMLR119:7076–7087,2020。[正式会议页](https://proceedings.mlr.press/v119/mozannar20b.html)。本轮核对正式书目和摘要；先前笔记另读过引言，未声称本轮阅读全文或证明本地策略等同其方法。
3. Qiuyu Tian, Haoliang Sun, Yunshan Wang, Yinghuan Shi, Yilong Yin. *DeferredSeg: A Multi-Expert Deferral Framework for Medical Image Segmentation*. arXiv:2604.12411v2,2026。[固定v2](https://arxiv.org/abs/2604.12411v2)。本轮核对作者、版本和摘要；arXiv comments记载Accepted by Pattern Recognition，尚未独立核对期刊DOI/卷页，因此以v2引用。
4. Isaac Corley, Caleb Robinson, Anthony Ortiz. *A Change Detection Reality Check*. arXiv:2402.06994v2,2024。[arXiv](https://arxiv.org/abs/2402.06994)、[作者机构页](https://www.microsoft.com/en-us/research/publication/a-change-detection-reality-check/)。实际读摘要与版本记录，不借其不同协议成绩证明本地基线达到SOTA。
5. Xiang Li等. *REOBench: Benchmarking Robustness of Earth Observation Foundation Models*. NeurIPS38,Datasets and Benchmarks Track,2025。[正式会议页](https://proceedings.nips.cc/paper_files/paper/2025/hash/58af38e2fba04b5e00c3450018c56406-Abstract-Datasets_and_Benchmarks_Track.html)。实际读摘要与会议元数据，未运行其基础模型或采用几何干预。
6. Elena Buglakova等. *Tiling artifacts and trade-offs of feature normalization in the segmentation of large biological images*. ICCV2025,pp.13109–13118。[CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Buglakova_Tiling_artifacts_and_trade-offs_of_feature_normalization_in_the_segmentation_ICCV_2025_paper.html)、[arXiv:2503.19545v1](https://arxiv.org/abs/2503.19545v1)。CVF检索返回正式条目，直接HTML曾403；arXiv摘要可读。没有声称阅读全文或已复现BatchRenorm。
7. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. arXiv:1512.03385,2015技术报告。[原始版本](https://arxiv.org/abs/1512.03385)。本轮核对原始元数据和摘要；主文只说明采用已有残差编码器，不借本文的外部指标作比较。
8. Tsung-Yi Lin等. *Feature Pyramid Networks for Object Detection*. arXiv:1612.03144v2,2017。[固定v2](https://arxiv.org/abs/1612.03144v2)。本轮核对作者、版本和摘要。FPN风格指top-down/lateral组织，本地变化检测解码器不是原物体检测器的直接复现。

本稿的实验数值来自表格verification.json列明的本地保存产物，不能把外部论文的成绩混入本地结果。
