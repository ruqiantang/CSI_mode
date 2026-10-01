# TRAINING_PROTOCOL.md

> 冻结的正式训练协议（单一事实来源）。
> 本文只定义「正式主实验」的超参与协议；debug/smoke 配置见 `configs/debug_*.yaml`，不与正式结果混用。
> 详细方案与命令见 `WIFO_VS_FULL_PLAN.md`。

## 1. 模型

只比较 **WiFo** 与 **Full**（Base 档）。

| | WiFo | Full |
|---|---|---|
| config | `wifo_base_paper.yaml` | `full_base_geometry.yaml` |
| encoder | width 512 / depth 6 / heads 8 | 同左 |
| decoder | depth 4 / width 512 / heads 8 | 同左 |
| embedding | `Conv3d` `(4,4,4)`，flattened antenna | per-antenna `Conv2d (4,4)` |
| 位置编码 | STF（t,f,s 三维 sincos） | `(t,k,r,c)` 4D |
| relative bias | 无 | `B(Δr,Δc)=b_r(Δr)+b_c(Δc)` |
| mask 任务 | Random / Temporal / Frequency | Random / Temporal / Frequency / Antenna |

## 2. 掩码比例

| task | ratio |
|---|---|
| random | 0.85 |
| temporal | 0.50 |
| frequency | 0.50 |
| antenna（仅 Full） | 0.25（可见率 ≥ 0.5） |

Full 不使用 Row / Column / 2D Block。

## 3. 训练超参（对齐原 WiFo 论文）

| 参数 | 值 |
|---|---|
| batch size | **128**（全模型一致；放不下才 grad-accum 到有效 128） |
| epochs | **200** |
| warmup | **5 epochs** |
| optimizer | AdamW（β1=0.9, β2=0.999, weight_decay=0.05） |
| base LR | 5e-4 |
| schedule | cosine decay |
| precision | **TF32**（主结果）；BF16 仅工程加速，不混入主结果 |
| task schedule | sequential（每 batch 逐 task `forward → backward(loss/n_tasks)`，最后仅一次 `optimizer.step()`） |

## 4. 数据

- D1–D16 联合预训练（一个 WiFo + 一个 Full）。
- 每数据集 12000 样本 = 训练 9000 / 验证 1000 / inference 2000（论文原文）。
- **预训练集合 = 训练(9000) + 验证(1000) = 10000/数据集，16 数据集共 160000**（论文原文：pre-trained on all training and validation samples）。
- **文件映射（官方源码已核实，见 DATA_PROTOCOL_AUDIT.md §1）**：
  - `X_train.mat`（9000）= training
  - `X_test.mat`（1000，内部变量 `X_val`）= **validation** → 进入预训练
  - `X_val.mat`（2000，内部变量 `X_test`）= **inference** → D1–D16 评估
  - ⚠️ 文件名与内容交叉：预训练要接 `X_train.mat` **和 `X_test.mat`**，不是 `X_val.mat`。
- D17 / D18 zero-shot 评估。
- **忽略 D19**。
- **噪声 / 标准化（官方源码已核实，见 DATA_PROTOCOL_AUDIT.md §3）**：标准化已 baked（数据已按 dataset mean/var 标准化，无需 runtime 再加）；**20 dB 高斯噪声需 runtime 加**（训练 + 推理都加，按 batch 功率）。

## 5. Seed 协议

- WiFo × 3 seeds + Full × 3 seeds = 6 个完整训练。
- 报告 mean ± std。

## 6. 评估

- D1–D16：Temporal NMSE、Frequency NMSE（逐数据集 + D1–D16 Average）。
- D17 / D18：zero-shot Temporal / Frequency NMSE。
- Antenna Reconstruction NMSE：仅 Full；WiFo 标 N/A。评估范围 = D1–D16 inference 集（`X_val.mat`）+ D17/D18 zero-shot 集，antenna mask ratio=0.25，评估被掩天线位置的 NMSE（与 temporal/frequency 同协议）。

## 7. 精度语义

| precision | 含义 |
|---|---|
| `tf32` | fp32 + TF32 tensor core（`torch.backends.cuda.matmul.allow_tf32=True`） |
| `bf16` | bfloat16 autocast |
| `fp16` | float16 autocast + GradScaler |
| `fp32` | 纯 fp32（关闭 TF32） |

`Trainer(precision=...)`；旧 `use_amp=True` 等价于 `precision="fp16"`（已弃用）。

## 8. DataLoader

正式训练建议：`--num-workers 8 --preload --pin-memory --persistent-workers`，`prefetch_factor` 按 profiling 定。`--preload` 将 D1–D16 全量读入内存（~34GB），解除 HDF5 lazy loading 的 fork 安全限制。
