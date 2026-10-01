# WIFO_VS_FULL_PLAN.md

> 正式实验方案：**WiFo vs Full**。
> 阶段二产物：取消此前因算力/显存/预算引入的工程妥协，恢复原 WiFo 论文正式训练协议，只比较两个模型。
> 日期：2026-10-01
> 关联：`TRAINING_PROTOCOL.md`（冻结的超参协议）、`RESOURCE_CONSTRAINT_AUDIT.md`（阶段一审计）。

---

## 1. 目标与范围

在**尽量保持原 WiFo 论文训练协议不变**的前提下，比较加入显式二维平面阵列（UPA）几何建模与 Antenna 空间重建后，模型性能与泛化如何变化。

**只比较两个模型**：

| 模型 | 说明 |
|---|---|
| **WiFo** | 严格按原论文：flattened-antenna、3D patch `(4,4,4)`、STF 位置编码、3 种 mask |
| **Full** | per-antenna TF embedding、`(t,k,r,c)` 二维坐标、UPA relative bias、4 种 mask（第 4 种为 antenna） |

**不再做** A/B/C/D 中间消融（旧代码保留，但不进入正式训练计划与正式结果）。

---

## 2. 模型配置

### 2.1 WiFo（原始设计，不改动）

- encoder：width=512、depth=6、heads=8
- decoder：depth=4、width=512、heads=8
- 输入：`T × K × N`，天线维保持 flattened antenna axis
- patch：`(4,4,4)`（`Conv3d`）
- 位置编码：原始 STF（time/frequency/space 三维 sincos，D=512 时 170/170/172 分配）
- 配置文件：`configs/wifo_base_paper.yaml`

### 2.2 Full（核心结构保持不变）

- 同一骨干尺寸：width=512、depth=6、decoder_depth=4、heads=8（与 WiFo 公平）
- Per-antenna Time-Frequency embedding（`Conv2d (pt,pf)`，跨天线共享）
- 显式二维坐标 `(t,k,r,c)`
- UPA relative bias：`B_h(Δr,Δc) = b_r^h(Δr) + b_c^h(Δc)`（分离式，行/列查表）
- 配置文件：`configs/full_base_geometry.yaml`

**不新增** Mamba / MoE / GNN / RoPE / 新 decoder / 新 loss / token merge 等结构。

---

## 3. 掩码任务

| 模型 | 任务数 | Mask | 比例 |
|---|---|---|---|
| **WiFo** | 3 | Random | 0.85 |
| | | Temporal | 0.50 |
| | | Frequency | 0.50 |
| **Full** | 4 | Random | 0.85 |
| | | Temporal | 0.50 |
| | | Frequency | 0.50 |
| | | **Antenna** | 0.25 |

- WiFo **不加** spatial mask。
- Full 的第四种任务**只有 Antenna Mask**；**删除** Row / Column / 2D Block（代码保留，但正式配置 `spatial_types: [antenna]` 不再混入）。

---

## 4. 训练数据

- **联合预训练**：D1–D16 共同组成一个多数据集训练集合，训练**一个** WiFo 和**一个** Full（不是每数据集单独训模型）。
- 每数据集 12000 样本 = 训练 9000 / 验证 1000 / inference 2000（论文原文）。
- **预训练集合 = 训练(9000) + 验证(1000) = 10000/数据集，共 160000**（论文原文：WiFo pre-trained on all training and validation samples）。
- **文件映射（官方源码已核实，见 DATA_PROTOCOL_AUDIT.md §1）**：
  - `X_train.mat`（9000）= training
  - `X_test.mat`（1000，内部变量 `X_val`）= **validation** → 进入预训练
  - `X_val.mat`（2000，内部变量 `X_test`）= **inference** → D1–D16 评估
  - ⚠️ 文件名与内容交叉：预训练要接 `X_train.mat` **和 `X_test.mat`**，不是 `X_val.mat`。
- **D17 / D18**：用于 zero-shot evaluation。
- **D19 当作不存在**（不下载、不生成、不建 config、不在正式计划中提及）。
- **噪声 / 标准化（官方源码已核实，见 DATA_PROTOCOL_AUDIT.md §3）**：标准化已 baked（数据已按 dataset mean/var 标准化，无需 runtime 再加）；**20 dB 高斯噪声需 runtime 加**（训练 + 推理都加）。

---

## 5. 训练协议（对齐原 WiFo 论文）

| 参数 | 值 |
|---|---|
| Model scale | **Base**（width 512） |
| Batch size | **128** |
| Epochs | **200** |
| Warmup | **5 epochs** |
| Optimizer | AdamW（β1=0.9, β2=0.999, weight_decay=0.05） |
| Base LR | 5e-4 |
| LR schedule | Cosine decay |
| Precision | **TF32**（主结果；BF16 仅作工程加速，不混入主结果） |
| Task schedule | **sequential**（每 batch 逐 task `forward → backward(loss/n_tasks)`，最后仅一次 `optimizer.step()`） |

- 不因 GPU 更强而修改 batch / lr / epoch。
- 全模型 **effective batch = 128** 一致（优先 physical batch=128；某卡放不下才用 gradient accumulation 补到有效 128）。

---

## 6. 任务执行逻辑（每 batch 一次 optimizer.step）

- **WiFo**：`L = (L_random + L_temporal + L_frequency) / 3`，一次 backward + 一次 step。
- **Full**：`L = (L_random + L_temporal + L_frequency + L_antenna) / 4`，一次 backward + 一次 step。
- 两者每 batch 的 optimizer update 次数一致（各 1 次）。

> 实现（`Trainer.train_epoch` 的 `sequential` 模式，已按逐 task 反向落地）：
> 一次 `zero_grad` → 逐 task `forward` → **立即 `backward(loss / n_tasks)` 并释放该 task 计算图** → 最后仅一次 `optimizer.step()`。数学协议不变（梯度 = 各 task loss 的均值梯度），峰值显存从 n_tasks×激活降到 1×激活。

---

## 7. 3-Seed 协议

- WiFo × 3 seeds，Full × 3 seeds，共 **6 个完整训练**。
- 每个 seed：完整 D1–D16 联合预训练 200 epochs → D17/D18 统一评估。
- 报告 mean ± std，不报单 seed。

---

## 8. 评估方式

- **D1–D16**：分别报告 Temporal NMSE、Frequency NMSE，并给出 D1–D16 Average。
- **D17**：Zero-shot Temporal NMSE、Zero-shot Frequency NMSE。
- **D18**：Zero-shot Temporal NMSE、Zero-shot Frequency NMSE。
- **Antenna Reconstruction NMSE**：仅 Full 报告；WiFo 写 **N/A**（不修改 WiFo 来强行支持）。评估范围 = D1–D16 inference 集（`X_val.mat`）与 D17/D18 zero-shot 集，antenna mask ratio=0.25，评估被掩天线位置的 NMSE（与 temporal/frequency 同评估协议）。

---

## 9. 正式比较表

| Model | D1–D16 Temporal | D1–D16 Frequency | D17 Temporal | D17 Frequency | D18 Temporal | D18 Frequency | Antenna Recon. | Params | FLOPs | Train Time | Peak GPU Mem |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WiFo | | | | | | | N/A | | | | |
| Full | | | | | | | | | | | |

只比较 WiFo 与 Full。

---

## 10. GPU Benchmark 方法

正式大规模训练前，先在 4090D / H800 上分别 benchmark WiFo Base 与 Full Base：

```bash
python scripts/bench.py --config configs/wifo_base_paper.yaml --model-type baseline --batch-size 32,64,128 --precision tf32
python scripts/bench.py --config configs/full_base_geometry.yaml --model-type upa --batch-size 32,64,128 --precision tf32
```

记录：Peak GPU Memory、Step Time、Samples/s、GPU Utilization。重点确认 **Full Base + physical batch=128 能否直接运行**（决定是否需要 gradient accumulation）。

---

## 11. 正式训练命令

> ⚠️ **PLACEHOLDER / DO NOT RUN**：以下命令仅示意结构，**当前不可直接执行**。
> 原因：预训练集合必须是 160000 样本 = `X_train.mat`(9000) + `X_test.mat`(1000 validation)，而下面只接了 `X_train.mat`。且数据尚未解压到 `data/`、20 dB 噪声 runtime 实现尚未落地、D17/D18 zero-shot 文件待核实。映射确定后按 DATA_PROTOCOL_AUDIT.md §1 补齐 `--data` 里的 validation 文件。

**WiFo-Base（D1–D16 联合预训练，seed=17）：**

```bash
python scripts/pretrain.py \
  --config configs/wifo_base_paper.yaml --model-type baseline \
  --data D1=data/D1/X_train.mat --data D2=data/D2/X_train.mat \
  --data D3=data/D3/X_train.mat --data D4=data/D4/X_train.mat \
  --data D5=data/D5/X_train.mat --data D6=data/D6/X_train.mat \
  --data D7=data/D7/X_train.mat --data D8=data/D8/X_train.mat \
  --data D9=data/D9/X_train.mat --data D10=data/D10/X_train.mat \
  --data D11=data/D11/X_train.mat --data D12=data/D12/X_train.mat \
  --data D13=data/D13/X_train.mat --data D14=data/D14/X_train.mat \
  --data D15=data/D15/X_train.mat --data D16=data/D16/X_train.mat \
  --epochs 200 --batch-size 128 --warmup-epochs 5 \
  --precision tf32 --task-schedule sequential \
  --num-workers 8 --preload --pin-memory --persistent-workers \
  --seed 17 \
  --checkpoint checkpoints/wifo_base_d1_d16_seed17.pt
```

**Full-Base（D1–D16 联合预训练，seed=17）：**

```bash
python scripts/pretrain.py \
  --config configs/full_base_geometry.yaml --model-type upa \
  --data D1=data/D1/X_train.mat --data D2=data/D2/X_train.mat \
  --data D3=data/D3/X_train.mat --data D4=data/D4/X_train.mat \
  --data D5=data/D5/X_train.mat --data D6=data/D6/X_train.mat \
  --data D7=data/D7/X_train.mat --data D8=data/D8/X_train.mat \
  --data D9=data/D9/X_train.mat --data D10=data/D10/X_train.mat \
  --data D11=data/D11/X_train.mat --data D12=data/D12/X_train.mat \
  --data D13=data/D13/X_train.mat --data D14=data/D14/X_train.mat \
  --data D15=data/D15/X_train.mat --data D16=data/D16/X_train.mat \
  --epochs 200 --batch-size 128 --warmup-epochs 5 \
  --precision tf32 --task-schedule sequential \
  --num-workers 8 --preload --pin-memory --persistent-workers \
  --seed 17 \
  --checkpoint checkpoints/full_base_d1_d16_seed17.pt
```

> 其余 seed（18、19）只改 `--seed` 与 `--checkpoint` 路径即可。

**D17/D18 zero-shot 评估（示例，Full）：**

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/full_base_geometry.yaml \
  --checkpoint checkpoints/full_base_d1_d16_seed17.pt \
  --dataset D17 --source mat --mat-path data/D17/X_test.mat \
  --samples 1000 --batch-size 16
```

---

## 12. 文件清单

**新增 config：** `configs/wifo_base_paper.yaml`、`configs/full_base_geometry.yaml`、`configs/debug_wifo.yaml`（DEBUG ONLY）、`configs/debug_full.yaml`（DEBUG ONLY）。

**代码改动：**
- `src/wifo_upa/train.py` — 加 `precision`（tf32/bf16/fp16/fp32），TF32 默认；checkpoint 记录 precision。
- `src/wifo_upa/data.py` — `LazyMatCSIDataset` 加 `preload`（预载入内存，解除 `num_workers=0` 限制）。
- `scripts/pretrain.py` — 正式默认值（batch=128/epochs=200/warmup=5/tf32）、`--precision`、`--preload`、`--pin-memory`、`--persistent-workers`、`--prefetch-factor`、`--num-workers`、`--seed`。
- `scripts/evaluate_checkpoint.py` — `--precision`（默认取 checkpoint 存储精度）。
- `scripts/bench.py` — 重写为正式 benchmark 脚本。
- `tests/test_formal_protocol.py` — 锁定 WiFo=3 任务 / Full=4 任务（antenna-only）与 TF32/FP32 语义。

**文档：** 本文件、`TRAINING_PROTOCOL.md`、`RESOURCE_CONSTRAINT_AUDIT.md`（更新）。
