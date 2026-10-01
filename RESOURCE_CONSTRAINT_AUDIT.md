# RESOURCE_CONSTRAINT_AUDIT.md

> 阶段一：系统审计 —— 找出「因资源不足而引入的工程妥协」，**只审计、不修改代码**。
> 日期：2026-10-01
> 依据：当前代码 + git 历史（21 commits，单分支 main）+ 协议文档。
> 「原始值」的权威来源：`docs/paper-results.md`（WiFo 论文提取）、`docs/source-audit.md`（官方源码审计）、`docs/execution-runbook.md`（正式运行手册）、`docs/experiment-plan.md`、`docs/model-spec.md`。
>
> ⚠️ git 里 `configs/` 全部在 v1（`788dc5f`）一次性创建、之后未变，**git 中不存在「优化前」的正式配置**；原始协议以论文/文档为准。

---

## 0. 一句话结论

当前代码里**没有**梯度累积、activation/gradient checkpointing、CPU/optimizer offload、token 截断、人工可见 token 上限这类「重量级」省显存手段——已确认不存在（见 §5）。

真正的「资源妥协」集中在**六个隐蔽地方**，且有一个**公平性隐患**：

1. **脚本 argparse 默认值被写成 smoke-test 规模**（`batch=2/8`、`epochs=1~10`、`samples=8~128`），正式训练必须靠命令行覆盖，极易误用。
2. **batch size 严重低于论文原始协议**（论文 **128**；UPA 扩展为省显存降到 8/16；当前默认更是 2/8）。
3. **精度从论文的 TF32 换成了 FP16 AMP**（且无 BF16 选项）。
4. **DataLoader 性能全为最小值**（`num_workers=0`、无 `pin_memory`/`prefetch`/`persistent_workers`）。
5. **任务调度用了省钱模式 `sample`**（每 batch 1 个任务），而最终严格对比必须 `sequential`。
6. **训练范围停留在 D4、20 epoch**，而原始协议是 D1–D16 预训练 + D17–D19 零样本、200 epoch。

**公平性隐患**：`execution-runbook.md` 给出的正式命令里，**baseline 用 batch=128、UPA 用 batch=8**——这是「UPA attention 16× 后为省显存被迫降 batch」的产物。现在 4090D/H800 显存足够，应把 UPA 恢复到与 baseline 相同的 batch。

**科学协议零改动**——mask 比例、patch size、UPA bias、`(t,k,r,c)`、数据划分、优化器、LR 调度、seed 都完好（见 §4）。

---

## 1. 审计范围与方法

| 维度 | 覆盖 |
|---|---|
| 配置 | `configs/` 全部 17 个 yaml |
| 训练/启动脚本 | `scripts/run_*.py`、`pretrain.py`、`bench.py`、`setup_server.sh`、`download_*.py` |
| 核心源码 | `src/wifo_upa/`（config/train/model/baseline/data/evaluate/masks/attention/pe/embed/geometry/initialization/smoke_test） |
| 协议文档 | `paper-results.md`、`source-audit.md`、`execution-runbook.md`、`experiment-plan.md`、`model-spec.md`、`formal-pilot.md`、`ablation-campaign.md`、`wifo-vs-full-ablation.md` |
| git 历史 | 21 commits；`git log -p -- configs/`、`scripts/` |

---

## 2. 论文原始协议（权威基准，用于 OLD/ORIGINAL 列）

来源：`docs/paper-results.md §2/§3`、`docs/source-audit.md §8`。

| 参数 | 论文值 |
|---|---|
| 优化器 | AdamW（β1=0.9, β2=0.999, weight_decay=0.05） |
| **Batch size** | **128** |
| **Epochs** | **200** |
| LR 调度 | Cosine decay，**warmup = 5 epochs** |
| 基础学习率 | 5×10⁻⁴ |
| Patch | (4,4,4) |
| 掩码比例 | random 85% / time 50% / freq 50% |
| **精度** | **TF32** |
| 数据 | 每数据集 12000 样本（9000/2000/1000 = 训练/验证/测试）；D1–D16 预训练，D17–D19 零样本 |
| 硬件 | 4× RTX 4090 + EPYC 7763 64 核 + 256GB |

**论文模型五档**（`paper-results.md §2`，均为论文自己的合法规模）：

| 档 | enc.width | enc.depth | 参数量 | 本仓库对应 config |
|---|---|---|---|---|
| WiFo-Tiny | 64 | 6 | 0.3M | — |
| WiFo-Little | 128 | 6 | 1.4M | `little.yaml` / `wifo_little.yaml` |
| WiFo-Small | 256 | 6 | 5.5M | `small.yaml` / `full.yaml` / `ablation_*.yaml` |
| WiFo-Base | 512 | 6 | 21.6M | `base.yaml` / `wifo_base.yaml` |
| WiFo-Large | 768 | 8 | 86.1M | — |

> **重要澄清**：`little.yaml`（128）不是「被压缩坏了」的模型，而是论文自带的 **WiFo-Little** 档。同理 `base.yaml`（512）是 WiFo-Base。宽度档位本身不是资源妥协；真正的资源妥协是**用哪个档 + 多大 batch + 多少 epoch 跑正式实验**。

---

## 3. 逐项审计

### 3.1 Batch size 与 gradient accumulation

**没有 gradient accumulation（已确认不存在）。** 问题在「默认值太小」+「UPA 被降到 8」。

| 文件 | 当前值 | 原始值 | 建议 |
|---|---|---|---|
| [run_formal_pilot.py:41](scripts/run_formal_pilot.py#L41) | `--batch-size` default **2** | **128**（论文）；UPA 可行性档 8/16 | **128**（或与 baseline 一致的有效 batch） |
| [pretrain.py:49](scripts/pretrain.py#L49) | `--batch-size` default **8** | 128 | 128 |
| [run_campaign.py:73](scripts/run_campaign.py#L73) | `--batch-size` default **2** | 128 | 128 |
| [run_ablation.py:41](scripts/run_ablation.py#L41) | `--batch-size` default **2** | 128 | 128 |
| [run_public_split_pilot.py:34](scripts/run_public_split_pilot.py#L34) | `--batch-size` default **2** | 128 | 128 |

- **原始值溯源**：论文 batch=128（`paper-results.md:50`）。`source-audit.md:71-73` 明确：因 UPA token 4×、attention 16×，**「可行性」建议降到 batch 8/16 + mixed precision + gradient accumulation when necessary**。所以 8/16 是 UPA 的显存妥协，128 才是原始目标。
- **公平性隐患（关键）**：`execution-runbook.md:143` baseline 用 `--batch-size 128`，`:170` UPA 用 `--batch-size 8`——两者 effective batch 不一致，违反你的 #十四。**现在必须统一。**
- **显存可行性**：D4 消融 Full 峰值 ~1.34GB@batch16（A10），batch128 ≈ 8× ≈ ~11GB，**H800(80GB) 轻松、4090D(24GB) 大概率可行**。故应把 UPA 从 8 恢复到 128（或至少 16→用 accumulation 补到有效 128）。
- **建议**：`full.yaml` 里 `batch=128`；若 4090D 上 UPA 放不下真实 128，用 **real batch 16 + accumulation 8 = 有效 128**，且 baseline 也同步用同一套（保证 global effective batch 完全一致）。**不要**各自拉满到不同值。

### 3.2 Epochs / warmup / 早停

| 文件 | 当前值 | 原始值 | 建议 |
|---|---|---|---|
| [run_formal_pilot.py:40](scripts/run_formal_pilot.py#L40) | `--epochs` default **5** | **200** | max=200 + early stopping |
| [pretrain.py:48](scripts/pretrain.py#L48) | `--epochs` default **1** | 200 | 同上 |
| [run_campaign.py:72](scripts/run_campaign.py#L72) | `--epochs` default **1** | 200 | 同上 |
| [run_public_split_pilot.py:33](scripts/run_public_split_pilot.py#L33) | `--epochs` default **10** | 200 | 同上 |
| [run_formal_pilot.py:43](scripts/run_formal_pilot.py#L43) | `--warmup-epochs` default **1**（D4 用 3） | **5** | 5 |
| [pretrain.py:51](scripts/pretrain.py#L51) | `--warmup-epochs` default **0** | 5 | 5 |

- **现状**：无 `early_stopping`、无 best-checkpoint 持久化（`best_state` 只是脚本内存变量）。D4 的 20 epoch 是 preliminary，且 val NMSE 在 epoch 20 仍在降（`wifo-vs-full-ablation.md` 曲线）——远未收敛。
- **建议**：正式协议改为「以收敛为准」：`max_epochs=200` + `early_stopping` + 每 epoch 存 best checkpoint + 记录完整 val 曲线。不要固定 20。

### 3.3 数据集范围 / sample cap / subset

| 文件 | 当前值 | 原始值 | 建议 |
|---|---|---|---|
| [run_formal_pilot.py:37-39](scripts/run_formal_pilot.py#L37-L39) | samples default **128/32/64** | 全量 9000/2000/1000 | 全量 |
| [run_campaign.py:69-71](scripts/run_campaign.py#L69-L71) | **64/16/32** | 全量 | 全量 |
| [run_public_split_pilot.py:29-31](scripts/run_public_split_pilot.py#L29-L31) | **32/8/16** | 全量 | 全量 |
| [run_ablation.py:37-40](scripts/run_ablation.py#L37-L40) | **D5 / synthetic / 8** | — | 保留为 debug |
| [evaluate_checkpoint.py:33-34](scripts/evaluate_checkpoint.py#L33-L34) | 评估 `--samples` default **8**、`--batch-size` default **2** | 完整 held-out test split | 全量测试集 |
| 训练数据范围 | 仅 **D4** | **D1–D16 预训练 + D17–D19 零样本** | D1–D16 + D17–D19 |

- `pretrain.py --samples-per-dataset` 默认 `None`（= 全量，正确），是唯一默认正确的一个。
- ⚠️ **命名分歧**：你的指示写「D1–D18」，但论文/文档/data.py 是 **D1–D19**（D1–D16 训练 + D17–D19 零样本；`DATASET_SHAPES` 含 D1–D19）。第二阶段请确认以论文的 D1–D19 为准。
- 每数据集样本：论文 12000 = 训练 9000 / 验证 2000 / 测试 1000（`wifo-vs-full-ablation.md §9` 的 D4 文件实测 9000/2.28GB、2000/508MB、1000/254MB）。

### 3.4 模型宽度 / 深度

- **宽度档本身不是资源妥协**：64/128/256/512/768 是论文自带五档（Tiny→Large），本仓库的 `little/small/base` 一一对应。
- **真正的妥协**：正式对比该用哪一档。论文主表是 **WiFo-Base(512)**；UPA 扩展在可行性阶段用 **Small(256)/Little(128)**（`source-audit.md:71-73`）。`execution-runbook.md` 的正式命令两者都用 **base(512)**。
- **深度/头/MLP**：所有 config 一致（depth=6 / decoder_depth=4 / heads=8 / mlp_ratio=2），与论文一致，**无压缩**。`pt=pf=4` 是 paper 兼容冻结（`config.py:58-60` 强制），非省资源。
- **建议**：正式消融主档 = **Base(512)**（匹配 WiFo-Base 已发表数字，便于对标）；`Small(256)` 作 scale 对照；`Little(128)` 留作 debug/quick。**WiFo-like 与 Full 必须同档**——只放大 Full 是禁止的。

### 3.5 精度 / AMP / BF16

| 位置 | 当前值 | 原始值 | 说明 |
|---|---|---|---|
| [train.py:88](src/wifo_upa/train.py#L88) | CUDA AMP = **fp16**（CPU = bf16），`--amp` 才开启 | 论文 **TF32** | 精度选择 |
| [train.py:84-89](src/wifo_upa/train.py#L84-L89) | `_autocast_context` 无 fp32/bf16 选项 | — | — |

- **重要**：论文精度是 **TF32**（`paper-results.md:56`），不是当前代码的 FP16 AMP。FP16 是更激进的精度选择。
- **建议**：按你的指示 H800 上**优先 BF16**（原生支持、数值更稳）。第二阶段给 `--precision {tf32,bf16,fp16,fp32}` 四选（或至少 bf16/fp32），并实测数值稳定性。**不要**机械切 FP32——AMP/TF32 是硬件效率不是纯省显存。
- **保留项（不要动）**：`model.py:165-167` 与 `baseline.py:193-195` 的 `encoded.float()` + `torch.autocast(enabled=False)` 包裹 decoder，是**数值正确性修复**（fp16/fp32 索引写入 dtype 不匹配），不是省显存。切 BF16 后可重新评估是否还需保留，但**先跑数值验证再改**。

### 3.6 DataLoader 性能（num_workers / pin_memory / prefetch / persistent_workers）

| 文件 | 当前值 | 建议 |
|---|---|---|
| [run_formal_pilot.py:102](scripts/run_formal_pilot.py#L102) | `num_workers=0`（硬编码） | 4–16 + `pin_memory=True` + `persistent_workers=True` + `prefetch_factor` |
| [run_campaign.py:129,147](scripts/run_campaign.py#L129) | `num_workers=0`（硬编码） | 同上 |
| [pretrain.py:62,71-72](scripts/pretrain.py#L62) | `--num-workers` default 0，**`!=0` 直接报错**「HDF5 lazy loading currently requires --num-workers 0」 | 解除限制（见下） |
| [data.py:129-135,219](src/wifo_upa/data.py#L129-L135) | `LazyMatCSIDataset` 长驻 HDF5 句柄、逐样本懒读 | 正式数据改 eager 入内存 |

- **性质**：为省 CPU/RAM 把加载压成单进程；论文硬件是 64 核 EPYC，隐含并行加载。
- **根因与解法**：`LazyMatCSIDataset` 懒读 HDF5 导致 fork 不安全 → 强制 `num_workers=0`。解除需：**(a)** 正式数据用 `load_mat_csi` 一次读入内存（D4 全量 ~3GB；D1–D16 全量 ~34GB，论文 256GB 内存轻松容纳）；或 **(b)** worker 内延迟打开 HDF5。属第二阶段工程。
- **公平性**：无（纯吞吐）。
- `run_public_split_pilot.py`、`run_ablation.py` 的 DataLoader 连 `num_workers` 参数都没有（PyTorch 默认 0）。

### 3.7 任务调度 / 评估覆盖

| 项 | 当前值 | 原始值 | 建议 |
|---|---|---|---|
| D4 消融 `--task-schedule` | **`sample`** | **`sequential`**（final strict，`source-audit.md:155-158`） | 正式改 `sequential` |
| 评估任务覆盖 | `evaluate_task` 逐任务；WiFo-like 无 spatial（正确标 N/A） | 每模型统一评估 Temporal/Frequency/Random/(Spatial 若支持) | 保持 |
| spatial 子类（antenna/row/column/block） | 全部实现（`masks.py`），无裁剪 | 全保留 | 保持 |

- **性质**：文档原文「sample … reduces cost … must be disclosed」。`sample` 是省钱模式；`sequential` 是严格对比模式。正式结果必须 `sequential`。
- ⚠️ **公平性提示（需第二阶段显式拍板，非资源问题但影响公平）**：
  - `available_tasks()`（[train.py:67-71](src/wifo_upa/train.py#L67-L71)）Full 返回 4 任务（含 spatial），WiFo-like 返回 3 任务。`sample` 下 Full 每 batch 有 25% 概率训 spatial、WiFo-like 恒 0，且 Full 的 temporal/freq/random 每 step 密度低于 WiFo-like（25% vs 33%）。这是研究设计固有（只有 Full 支持空间掩码），但训练分布不对称，正式报告需显式说明或约定「训练任务池统一为 {random,temporal,frequency}，spatial 仅作评估」。
- `trainer.evaluate` 的 `_next_eval_task` 是**跨 batch 轮转任务**，其单一 `nmse` 是混合任务均值，不适合正式报告；正式流程应像脚本那样用 `evaluate_task` 逐任务逐 dataset 报告（脚本已这么做）。

### 3.8 多卡 / DDP

- **当前**：全仓库无 DDP/DataParallel。
- **建议**：本课题是 dataset×model×seed 大组合，**实验级并行**（不同 seed/dataset 放不同卡）比 DDP 简单；`pretrain.py` 已支持 resumable checkpoint。DDP 仅当单一大实验需要更大 effective batch 时再加。**不要为「看起来高级」强制 DDP。**

### 3.9 其他（已确认不存在）

| 手段 | 是否存在于代码 | 证据 |
|---|---|---|
| gradient accumulation / micro-batch | ❌ 无 | `train.py:train_epoch` 每 batch 一次 backward+step |
| activation / gradient checkpointing | ❌ 无 | 全 `src` grep 无 `checkpoint`（仅 `save/load_checkpoint` 持久化） |
| CPU / parameter / optimizer offload | ❌ 无 | grep 无 `offload` |
| token 截断 / 天线采样 / 位置丢弃 | ❌ 无 | `model.py` L=Tp·Kp·Nh·Nv 全量；`masks.py` 无截断 |
| 人工可见 token 上限 | ❌ 无 | 仅 `R_visible>=0.5`（研究设计） |
| sequence chunking / 分块 forward | ❌ 无 | — |
| 仅省显存引入的低精度 | ⚠️ 见 §3.5 | FP16 是精度选择，非纯省显存 |

> 附注：`attention.py:45-46` 的 bias 只物化瞬态 `[H,N,M]`、不缓存 `[H,L,L]`，是 `model-spec.md §6` 明确要求的**协议级内存约束**（数值无影响、只是每次重算），**保留**，不算需恢复的妥协。

---

## 4. OLD / ORIGINAL vs CURRENT vs RECOMMENDED 总表

| 参数 | ORIGINAL（论文/文档） | CURRENT（现状） | RECOMMENDED（建议恢复） | 性质 |
|---|---|---|---|---|
| batch size | **128** | 默认 2/8；runbook 里 UPA=8、baseline=128 | **128（全模型统一；放不下则 16+accum=有效128）** | 资源收缩 + 公平隐患 |
| gradient accumulation | 无（=1） | 无（=1） | 保持 1（仅放不下时按需引入） | — |
| epochs | **200** | 默认 1~10，D4 用 20 | **max=200 + early stopping + best ckpt** | 资源收缩 |
| warmup epochs | **5** | 默认 0/1，D4 用 3 | **5** | 资源收缩 |
| train/val/test 样本 | 全量 9000/2000/1000 | 默认 8~128 | **全量** | 资源收缩 |
| 数据集范围 | D1–D16 预训练 + D17–D19 零样本 | 仅 D4 | **D1–D16 + D17–D19** | 资源收缩 |
| 模型档 | 论文五档，主表 **Base(512)** | 512/256/128 三档并存，正式跑过 256 | **主档 Base(512)，Small(256) 对照，Little(128) 留 debug** | 可行性降档 |
| 精度 | **TF32** | FP16 AMP | **BF16（H800）/ TF32 / fp32 可选** | 精度选择 |
| num_workers | 未指定（64 核隐含并行） | 0（硬编码/报错限制） | **4–16 + pin_memory + persistent_workers** | 资源收缩 |
| pin_memory / prefetch | 未指定 | 无 | **开启 + profiling** | 资源收缩 |
| 任务调度 | 最终 **sequential** | D4 用 sample | **正式 sequential** | 资源收缩 |
| activation checkpointing | 未设计 | 无 | 保持无（显存够） | — |
| 多卡 | 4×4090（论文） | 无 DDP | 实验级并行优先，DDP 按需 | — |

---

## 5. 研究设计（审计确认 **不应改动**，列出以免误删）

以下属于科学协议/研究设计，**资源充足也不改**：

- mask 比例：random `0.85`、temporal `0.50`、frequency `0.50`、spatial antenna `0.25`（主对比再跑 `0.50`）。
- UPA 空间掩码：antenna/row/column/block 四策略；`R_visible_space >= 0.5` 不变式。
- `(t,k,r,c)` 四维位置编码；`pt=pf=4` patch（`config.py` 冻结，paper 兼容）。
- UPA relative bias：`max_delta_r=3 / max_delta_c=7`；分离式 `b_r+b_c`，不缓存全矩阵。
- 训练/验证/测试划分、noise、normalization。
- 优化器族 AdamW、lr=5e-4、weight_decay=0.05、cosine 调度、seed 协议。
- model-spec §10「第一版限制」：不 warm-start、不可学习 PE、不引入额外 loss/token 聚合（「让第一版可解释」的研究限制）。

---

## 6. 建议的第二阶段修改清单（等你确认后执行）

1. **加 `configs/debug.yaml` + `configs/full.yaml` 两层**：
   - `debug.yaml`：Little(128) 或更小、synthetic/小样本、1~5 epoch、sample schedule、fp32——快速 smoke test（现有 `smoke_test.py` 思路保留）。
   - `full.yaml`：Base(512)、D1–D16 全量、batch=128、200 epoch + 早停、warmup=5、sequential schedule、BF16、完整评估。
2. **把脚本默认值从 smoke 规模改为 full 规模**，或改为「从 config 读 batch/epochs/samples」，杜绝正式运行静默退回 batch=2/epochs=1。
3. **统一 batch=128**：UPA 从 runbook 的 8 恢复到与 baseline 一致；4090D 放不下就 16+accum=有效 128，且 baseline 同步。
4. **解除 `num_workers=0`**：数据 eager 入内存（或 worker 内延迟开 HDF5），开 `pin_memory/persistent_workers/prefetch`，按 profiling 定 `num_workers`。
5. **精度加 `--precision`（tf32/bf16/fp16/fp32）**，H800 上默认 BF16，验证数值。
6. **训练循环加 early stopping + best-checkpoint 持久化 + val 监控**，以收敛为准而非固定 epoch。
7. **正式运行统一 `sequential`**，并显式约定训练任务池（见 §3.7 公平性提示）。
8. **统一 experiment runner**：每实验记录 config/git commit/seed/dataset/model variant/epoch/best val/test metrics/training time/peak GPU mem/param count/token count/checkpoint path/log path（对应 D1–D19 × {WiFo-like,A,B,C,D,Full} × ≥3 seed 矩阵）。

---

## 7. 附：本次审计未做（阶段边界）

- 未改任何代码、config、脚本（阶段一=只审计）。
- 未重新设计模型、未新增网络结构、未动 mask/PE/bias/划分/优化器/调度/seed。
- 仅新增本审计文档 `RESOURCE_CONSTRAINT_AUDIT.md`。

---

## 8. 阶段二（已执行，2026-10-01）

按用户正式方案（WiFo vs Full）完成如下修改，详细见 `WIFO_VS_FULL_PLAN.md` 与 `TRAINING_PROTOCOL.md`：

- **精度**：`train.py` 加 `precision`（tf32/bf16/fp16/fp32），默认 tf32（论文精度）；`use_amp` 保留为 `fp16` 别名。
- **DataLoader**：`data.py` 加 `preload` 解除 `num_workers=0` 限制；`pretrain.py` 加 `--num-workers/--preload/--pin-memory/--persistent-workers/--prefetch-factor`。
- **正式默认值**：`pretrain.py` 默认 batch=128 / epochs=200 / warmup=5 / tf32 / sequential；新增 `--seed`（3-seed 协议）。
- **配置**：新增 `wifo_base_paper.yaml`、`full_base_geometry.yaml`（antenna-only spatial）、`debug_wifo.yaml`、`debug_full.yaml`（DEBUG ONLY）。
- **Benchmark**：`bench.py` 重写为 WiFo/Full × batch 32/64/128 × 精度 的吞吐/显存 benchmark。
- **测试**：新增 `tests/test_formal_protocol.py`（锁定 WiFo=3 任务、Full=4 任务 antenna-only、TF32/FP32 语义）。

> ⚠️ 本地 Windows Python 3.14 alpha 环境 torch/PyYAML 均损坏，`pytest` 无法在本机运行；`compileall` 已全部通过，测试需在服务器（Python 3.10 / torch 2.5.1）执行 `python -m pytest -q`。
