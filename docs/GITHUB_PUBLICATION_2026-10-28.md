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

独立检出的结果与后续公开访问核对将在发布提交前补齐到本记录。
原机器的 CUDA 检查通过不代表 Linux、CPU-only 或不同依赖环境已复现全部测试。

## 续接方式

接收者在 Codex 中获取仓库后，先读 `AGENTS.md`、`docs/CODEX_HANDOFF.md`、
`docs/RESEARCH_SPEC.md`、`docs/PORTABLE_SETUP.md`，验证快照清单，
汇报真实资源缺口后再推进当前获准范围。原聊天、进程和机器文件不会自动迁移。
