# REPOSITORY_CLEANUP_PLAN.md

> 正式实验前「代码与文档收口」计划。**本文件是计划，尚未执行任何移动/删除/重命名。**
> 日期：2026-10-01
> 原则：不改模型科学设计、不改正式超参、不重设计 WiFo/Full；只清理历史 pilot/ablation、建立唯一协议来源、消除文档冲突、让仓库结构清晰。

---

## 1. 四类划分

- **ACTIVE**：当前正式实验（WiFo vs Full）真正需要的文件，保留并整理进目标目录。
- **ARCHIVE**：历史 A/B/C/D、pilot、D4 preliminary、旧 campaign/plan/audit/JSON，移动到 `docs/archive/2026-10-01-preformal/`（或 `archive/configs`、`archive/scripts`、`archive/tests`）。**不得删除，不得再作为 AI 判断当前协议的依据。**
- **GENERATED**：`__pycache__`、`src/*.egg-info`、`.pytest_cache`、`*.pyc`，清理。
- **REMOVE**：临时测试文件，删除。

---

## 2. ACTIVE（保留 + 整理到目标结构）

### 2.1 源码（不改，14 个文件）

```
src/wifo_upa/
  __init__.py  attention.py  baseline.py  config.py  data.py  embed.py
  evaluate.py  geometry.py  initialization.py  masks.py  model.py  pe.py
  smoke_test.py  train.py
```

### 2.2 配置（重命名为 formal / debug 两层）

| 现在 | 目标 | 说明 |
|---|---|---|
| `configs/wifo_base_paper.yaml` | `configs/formal/wifo.yaml` | 正式 WiFo（Base 512 / 3 masks） |
| `configs/full_base_geometry.yaml` | `configs/formal/full.yaml` | 正式 Full（Base 512 / 4 masks, antenna-only） |
| `configs/debug_wifo.yaml` | `configs/debug/wifo.yaml` | DEBUG ONLY |
| `configs/debug_full.yaml` | `configs/debug/full.yaml` | DEBUG ONLY |

### 2.3 脚本（重命名，active 只保留入口）

| 现在 | 目标 |
|---|---|
| `scripts/pretrain.py` | `scripts/train.py` |
| `scripts/evaluate_checkpoint.py` | `scripts/evaluate.py` |
| `scripts/bench.py` | `scripts/benchmark.py` |
| `scripts/download_data.py` | 保留（HuggingFace D1–D18 `X_test.mat`） |
| `scripts/download_api.py` | 保留（AnyShare D1–D16 `X_train/X_val/X_test`，正式数据唯一来源） |

> 注：`download_api.py` 不在你列出的 4 个 active 脚本内，但正式训练需要它下载 AnyShare 的 `X_train/X_val`；建议保留（否则正式数据无法获取）。

### 2.4 测试（保留核心，删除旧入口测试）

保留：`test_attention.py test_data.py test_evaluate.py test_masks.py test_pe.py test_train_loop.py test_baseline.py test_embed.py test_geometry.py test_model.py test_formal_protocol.py`

需微调（引用被归档的 config）：
- `test_train_loop.py::test_frozen_ablation_d_and_full_configs`（L191-196，引用 `ablation_d.yaml`/`full.yaml`）→ 归档或改为 inline config。
- `test_model.py` L120-129（引用 `ablation_a/b/c/d.yaml`+`full.yaml`）→ 归档或改为 inline config。
- `test_formal_protocol.py`（引用 `wifo_base_paper.yaml` 等）→ 更新为 `configs/formal/*.yaml`、`configs/debug/*.yaml` 新路径。

### 2.5 文档（收敛为 4 个 active 文档）

| 目标 | 内容来源 |
|---|---|
| `docs/protocol.md` | 唯一正式训练协议（来自 `TRAINING_PROTOCOL.md`，冻结超参） |
| `docs/architecture.md` | 模型架构（来自 `model-spec.md`，精简） |
| `docs/data.md` | 数据来源/形状/split（来自 `anyshare-api.md` + `paper-results.md` 数据表） |
| `docs/runbook.md` | 运行手册（来自 `execution-runbook.md`，精简到 WiFo vs Full） |

> protocol.md 是唯一正式训练协议来源；其他 active 文档不得重新定义冲突超参。README 只作入口，不复制协议。

### 2.6 根目录

`README.md`（重写为入口）、`pyproject.toml`、`requirements.txt`、`.gitignore`、`.env.example`。

---

## 3. ARCHIVE（移动，不删除）

### 3.1 旧实验/设计文档 → `docs/archive/2026-10-01-preformal/`

```
ablation-campaign.md  experiment-plan.md  formal-pilot.md  public-split-pilot.md
verification.md  d4-trial-results.md  wifo-vs-full-ablation.md  execution-runbook.md
implementation-plan-v1.md  model-spec.md  tensor-contract.md  anyshare-api.md
TRAINING_PROTOCOL.md  WIFO_VS_FULL_PLAN.md  RESOURCE_CONSTRAINT_AUDIT.md
```

### 3.2 论文/源码审计 → `docs/literature/`

```
PilotWiMAE_UPA_Audit.md  paper-results.md  source-audit.md  references.md
```

### 3.3 旧 config → `archive/configs/`

```
ablation_a/b/c/d.yaml  pilot_a/b/c/d/full/wifo.yaml
base.yaml  full.yaml  little.yaml  small.yaml
wifo_base.yaml  wifo_little.yaml  wifo_small.yaml
```

### 3.4 旧脚本 → `archive/scripts/`

```
run_ablation.py  run_campaign.py  run_formal_pilot.py  run_public_split_pilot.py
verify_cuda.py  download_pku.py  download_pku_v2.py  setup_server.sh
```

### 3.5 旧实验 JSON → `docs/archive/2026-10-01-preformal/experiments/`

```
pilot_d17_full_smoke.json  pilot_d17_full_256_64_64_e30.json
pilot_d17_small_256_64_64_e12.json  trial_d4_small_e20.json  pilot_d8_server_e30.json
```

### 3.6 开题材料 → `manuscript/proposal/`

```
proposal/（FIGURE_NOTES.md  REVISION_NOTES.md  figures/generate_figures.py）
neu_logo.png  _gen_opening_report.py
```

---

## 4. GENERATED（清理）

```
__pycache__/（所有）  src/*.egg-info/  .pytest_cache/  *.pyc
```

---

## 5. REMOVE（删除）

- `anyshare_probe.png`（临时测试探针文件）

---

## 6. KEEP AS-IS（已下载数据，gitignored）

- `data/`（D1–D18 `X_test.mat`，~4.4GB，HuggingFace 公开测试文件；正式训练所需的 `X_train/X_val` 需另经 AnyShare 下载）

---

## 7. 目标目录树（执行后）

```
WiFo/
├── src/wifo_upa/            # 14 个源码文件（不变）
├── configs/
│   ├── formal/{wifo,full}.yaml
│   └── debug/{wifo,full}.yaml
├── scripts/
│   ├── train.py  evaluate.py  benchmark.py
│   ├── download_data.py  download_api.py
├── tests/                   # 核心测试（微调 config 引用）
├── docs/
│   ├── protocol.md  architecture.md  data.md  runbook.md
│   ├── archive/2026-10-01-preformal/   # 旧实验/设计文档 + experiments/
│   └── literature/           # 论文/源码审计
├── archive/
│   ├── configs/  scripts/  tests/
├── manuscript/proposal/      # proposal + neu_logo + _gen_opening_report
├── data/                    # 已下载测试数据（gitignored）
├── README.md  pyproject.toml  requirements.txt  .gitignore  .env.example
```

---

## 8. 测试依赖影响（关键）

归档旧 config 会**破坏**以下测试，需在整理时一并处理：

| 测试 | 引用 | 处理 |
|---|---|---|
| `test_train_loop.py::test_frozen_ablation_d_and_full_configs` | `ablation_d.yaml` `full.yaml` | 归档该测试（或改 inline config） |
| `test_model.py` L120-129（ablation config 参数化测试） | `ablation_a/b/c/d.yaml` `full.yaml` | 归档该测试（或改 inline config） |
| `test_formal_pilot.py` | `pilot_wifo.yaml`（且测 run_formal_pilot.py） | 整体归档 |
| `test_campaign.py` | 测 run_campaign.py（内部引用 pilot_*.yaml） | 整体归档 |
| `test_formal_protocol.py` | `wifo_base_paper.yaml` 等 | 更新为 `configs/formal/*` `configs/debug/*` 新路径 |

脚本引用更新：`benchmark.py`（原 bench.py）、`verify_cuda.py`（若保留则改 config 路径；本计划归档 verify_cuda.py）。

---

## 9. 执行顺序与验收

1. 建目录：`configs/formal` `configs/debug` `docs/archive/2026-10-01-preformal` `docs/literature` `archive/configs` `archive/scripts` `archive/tests` `manuscript/proposal`。
2. 移动 + 重命名（按 §2/§3 清单）。
3. 更新 config 引用（`test_formal_protocol.py`、`benchmark.py`）。
4. 删除 GENERATED + REMOVE。
5. 收敛 4 个 active 文档（protocol/architecture/data/runbook），重写 README 为入口。
6. 验收：`python -m compileall -q src tests scripts` 与 `pytest -q` 必须通过（现有模型行为不变）。
7. 输出：最终目录树、所有移动/删除/重命名清单、唯一正式协议、未解决阻塞项。

> ⚠️ 本地 Windows Python 3.14 alpha 环境 torch/PyYAML 损坏，`pytest` 本机跑不了；验收需在服务器（Python 3.10 / torch 2.5.1）执行。`compileall` 本机可跑。
