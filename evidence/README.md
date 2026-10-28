# 发布证据快照

此目录为 2026-10-28 GitHub 交接准备保存的原样证据索引，不是新的研究结果。

- 原代码、配置、文档、测试和不超过 1 MiB 的选定文本记录原样存入仓库。
- `runs/` 中较大的 JSON/JSONL 在 `large_text/` 逐文件无损 gzip 保存；
  `snapshot_manifest.json` 映射原路径、原 SHA256 与压缩文件 SHA256。
- 数据集、checkpoint、numpy 数组、QA 图片、下载归档、临时文件和导出包保留在
  原机；`external_assets_manifest.json` 记录未发布本地产物的大小和 SHA256。
- 历史失败、partial、监督证明、源快照及已消费 test 的既有文本记录按规则保留。
  阅读已有结果不授予重新评价 test 或启动旧命令的权限。

在仓库根目录运行：

```text
python -m scripts.repository_snapshot verify --deep
python -m scripts.repository_snapshot restore --all
```

恢复命令只恢复已发布的大型文本；不会取得未上传资源。清单中的所有历史记录
保持原机器路径和时间原貌。清单不把历史 `running` 标记解释成新机器的实时状态。
`runs/` 仍被 `.gitignore` 忽略，发布快照仅按明确文件清单加入 Git，避免以后把
新训练数据或权重意外提交。请先读 `docs/CODEX_HANDOFF.md` 理解最新状态。
