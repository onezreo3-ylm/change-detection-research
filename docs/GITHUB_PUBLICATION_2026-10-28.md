# GitHub 公开交接准备记录

2026-10-28，Asia/Shanghai。用户选择公开仓库，目标地址为
`https://github.com/onezreo3-ylm/change-detection-research`。
此批只整理交接、保存证据和验证工程安装，没有新的研究训练或 held-out 评价。

## 发布范围

原代码、配置、文档、测试和 4938 份小型 run 文本记录按原字节保存。
454 份大型 JSON/JSONL 原合计 3022599550 字节，无损 gzip 后 173056360 字节，
逐文件原路径、大小和压缩前后 SHA256 均保留。
原模型权重、数据、数组、QA 图片、下载 partial 和导出包保留在本地；
科研产物清单不包含 node_modules 等可重建依赖缓存。

`.gitattributes` 禁止自动换行转换，避免 Windows/Linux 克隆改写原研究绑定。
原科研 YAML 不做路径替换；新 CPU smoke 使用单独配置。
原研究账本、既有失败和停止结论均保持，科学源模块未为迁移改写。

## 原工作区实测检查

- `python -m unittest discover -s tests -v`：641 项，107.894 秒，全部通过。
- `python -m pcdc.smoke --config configs/smoke_cpu_portable.yaml`：通过；
  原架构、显式 CPU、合成 train/val、单轮和独立 val 重载；无真实数据/test。
- 命令退出码和完整输出位于 `evidence/release_checks/`。

## 独立检出验证

从仅含已提交文件的 `309e8ae77df28037c28474bf9c92fad04f88b23d` 检出到
独立目录，使用同一 Windows 解释器，未复制原权重、数据、数组或 QA 图片。

- 5999 份发布文件原字节校验和 454 份压缩前原字节的深校验全部通过。
- CPU 合成 smoke 通过，独立 val 重载与选模计数一致，不依赖原数据或权重。
- 恢复器实际恢复全部 454 份大型文本，原 SHA256 匹配；额外验证了同内容文件
  保留、不同内容拒绝覆盖、路径越界拒绝及损坏压缩文件拒绝。
- 首次完整测试运行 641 项，有 23 个错误：其中 3 个来自尚未解压的审计文件。
  解压后复测 641 项（78.742 秒），余下 20 个错误均为冻结 robust 协议对原机
  `base_config` 绝对路径的身份保护，不是缺代码或原文件字节变化。

**不宣称新路径的完整测试通过。** 本次交接保留原科研协议与校验条件；
真实移机续跑仍需另行审查路径映射和身份绑定。原机测试全绿与新检出安装/smoke
可用分别记录。完整日志与命令退出码在 `evidence/release_checks/`，
`portability_findings.json` 列出剩余错误。未实测 Linux、CPU-only 全套测试或
不同 Python/依赖环境；这些工程检查不构成研究效果复现。

随后提交只补交接文档、清单和验证日志，不改变上面被检查的科学源码或配置。
公开远端的实际提交应以 GitHub `main` 为准；没有把尚未执行的远端检查预写成通过。

## 续接方式

接收者在 Codex 中获取仓库后，先读 `AGENTS.md`、`docs/CODEX_HANDOFF.md`、
`docs/RESEARCH_SPEC.md`、`docs/PORTABLE_SETUP.md`，验证快照清单，
汇报真实资源缺口后再推进当前获准范围。原聊天、进程和机器文件不会自动迁移。
