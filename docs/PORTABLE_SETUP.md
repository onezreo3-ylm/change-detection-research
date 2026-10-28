# 新机器获取与安装

从项目根目录操作。仓库是可读的代码与研究交接快照，真实实验资源需另行取得。
GitHub 链接不会自动恢复原聊天、权重、设备或活进程。

## 获取与读取

```text
git clone https://github.com/onezreo3-ylm/change-detection-research.git
cd change-detection-research
```

先读 `AGENTS.md`、`docs/CODEX_HANDOFF.md` 和 `docs/RESEARCH_SPEC.md`。
Codex 本地可在克隆目录打开项目；使用云环境时，需在该账户的环境配置中
连接/选择仓库并准备 Python 依赖。只粘贴链接不等于获得原机器文件或仓库写权限。
官方说明：[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)、
[Cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments)。

## 安装工程依赖

原机已验证 Python 3.11.14、PyTorch 2.0.1、torchvision 0.15.2、CUDA 11.7、
NumPy 1.26.4、Pillow 12.0.0、PyYAML 6.0.3、SciPy 1.16.3、Matplotlib 3.10.8。
`requirements-reference.txt` 记录这些直接依赖版本，不是完整传递依赖锁。
`requirements-engineering.txt` 补全完整工程测试及保存结果绘图/区域工具的依赖。

在单独的 Python 3.11 虚拟环境中安装；依据目标机器明确选择 PyTorch CPU/CUDA
构建，不修改原科研环境或声称不同后端复现相同数值轨迹。例如基础工程安装：

```text
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-engineering.txt
```

Linux/macOS shell：

```bash
source .venv/bin/activate
python -m pip install -r requirements-engineering.txt
```

若需按原机直接版本安装，可改用 `requirements-reference.txt`；CUDA 构建与驱动
仍需独立核对。新环境版本差异必须记录，安装成功不代表研究复现成功。

## 校验保存证据

下列程序只用 Python 标准库，不读取原数据或载入模型：

```text
python -m scripts.repository_snapshot verify
python -m scripts.repository_snapshot verify --deep
```

第一条验证发布文件的原始字节 SHA256；第二条还流式解压大型文本并验证原文件
的 SHA256，不落地。`.gitattributes` 禁止自动转换换行，以保留原研究绑定。
Git commit 自身锚定清单文件；清单不自包含其自身哈希。

需分析某个较大原始 JSON 时，仅恢复需要的文件：

```text
python -m scripts.repository_snapshot restore --path runs/valais_p1_posttraining_v1_20261019/cs_val/conditions.json
```

具体可恢复路径以清单的 `compressed_records[].original_path` 为准。
`restore --all` 可恢复全部大型文本，需要约 3 GiB
额外空间。恢复器核验压缩与解压哈希，拒绝覆盖不同内容，也拒绝越出仓库路径。
它不恢复数据、模型、数组或图片，不执行任何归档命令或进程。

## 工程验证

CPU smoke 不需要解压历史证据。完整测试会读取部分历史审计 JSON，应先执行
`python -m scripts.repository_snapshot restore --all`。

```text
python -m unittest discover -s tests -v
python -m pcdc.smoke --config configs/smoke_cpu_portable.yaml
```

CPU smoke 是专门命名的合成工程 fixture，使用原 baseline 架构，检查单轮
train/val、best checkpoint 保存和独立 val 重载。`run_test_after_training: false`，
不读取真实 LEVIR/Valais 或任何真实 held-out test，也不使用下载权重。
它是显式 CPU 检查，绝不是请求 CUDA 运行失败后的回退。
输出 `runs/smoke_cpu_portable/` 必须尚不存在；重跑应在新的独立 YAML 中指定
新的输出和数据路径，保留旧记录。

完整测试包含显式 CUDA 检查，CPU 机器会按原声明跳过相应项目。
**当前完整测试不能在新路径直接全绿。** 发布时的独立 Windows 检出恢复全部
454 份文本后运行 641 项，仍有 20 个错误，均由冻结 robust 协议绑定原机绝对
`base_config` 路径触发：`Robust training must use the exact base config archived by the protocol`。
原工作区同一套 641 项通过，独立检出的 CPU smoke 和所有发布字节校验通过。
这些事实区分记录，不能把原机通过说成任意机器全量通过。

这次交接保留了科学身份保护；未改原路径、重算旧协议指纹或放宽校验来消除错误。
若下一步确需移机执行这些冻结流程，须单独设计并审查迁移配置及身份绑定，保留
原记录。完整失败和修复缺失文本后的复测日志见 `evidence/release_checks/`，
汇总见 `docs/GITHUB_PUBLICATION_2026-10-28.md`。Linux/其他依赖环境未实测。

## 真实研究迁移

原 YAML 和归档 JSON 中的 `E:/...`、Windows 路径及进程身份保留原样，以避免
改变科学记录和哈希。需要移机运行时创建独立、可审查的新机器配置，明确记录
路径映射，核对原数据 split、内容、标签语义、checkpoint 与源身份。
原运行目录中的授权不在另一台机器上自动生效；没有 GPU/权重/数据时不替换资源。

数据入口及限制见 `docs/DATA_PREPARATION.md` 和
`docs/VALAIS_NEW_STUDY_INTAKE_2026-09-29.md`。未上传文件的本地清单在
`evidence/external_assets_manifest.json`；其中哈希能核对所有者另行提供的文件，
不代表这些文件在 GitHub 可下载。请按 `docs/CODEX_HANDOFF.md` 保持预算和准入边界。
