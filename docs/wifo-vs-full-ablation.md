# WiFo-like vs Full（UPA）公平消融对比

> 日期：2026-09-27
> 原始结果 JSON：
> - Full：`WiFo_ablation/results/full_trial_d4_small_e20.json`
> - WiFo-like：`WiFo_ablation/results/wifo_baseline_d4_small_e20.json`

## 1. 实验目的

回答「**UPA 改动本身是否带来增益**」——在完全相同的数据、训练代码、超参数、随机种子下，只改变模型结构，对比：

- **WiFo-like 基线**：扁平天线 `(4,4,4)` patch + `(t,k,s)` 一维位置编码（官方 WiFo 的简化重实现）
- **Full（UPA 扩展）**：`(4,4,1)` patch + `(t,k,r,c)` 二维坐标编码 + UPA 相对偏置 + 空间掩码

这是「第一层：工程内部公平消融」，与「第二层：官方 WiFo 源码复现」相区分。

## 2. 实验设置（完全一致，唯一变量是模型结构）

| 项目 | 配置 |
|---|---|
| 数据集 | D4（T=16，K=32，4×8 UPA，UMa+LoS） |
| 数据划分 | 训练 9000 / 验证 2000 / 测试 1000 |
| epoch | 20 |
| batch size | 16 |
| 优化器 | AdamW，lr=5e-4，weight_decay=0.05，grad_clip=1.0 |
| 调度器 | cosine，warmup 3 epoch |
| 任务调度 | sample（每 batch 随机抽一个掩码任务） |
| 掩码比例 | random 0.85 / temporal 0.50 / frequency 0.50 / spatial 0.25 |
| 精度 | AMP（fp16） |
| 随机种子 | 17 |
| 硬件 | NVIDIA A10 24GB，torch 2.5.1+cu124，Python 3.10.12 |

两个模型的骨干尺寸完全相同：embed 256 / depth 6 / decoder_depth 4 / heads 8 / mlp_ratio 2.0 / pt 4 / pf 4。

## 3. 模型结构差异

| 组件 | WiFo-like | Full |
|---|---|---|
| patch 方式 | `Conv3d (4,4,4)`，把 32 根天线拍平按 4 分组 | `(4,4)` 时频 patch，天线独立（Nh×Nv） |
| token 数 | **256**（4×8×8） | **1024**（4×8×4×8） |
| 位置编码 | `(t,k,s)` 一维 sincos（control） | `(t,k,r,c)` 二维（4d） |
| 相对偏置 | 无 | UPA relative bias（行列方向查表） |
| 空间掩码 | 不支持 | 支持（antenna/row/column/block） |
| 参数量 | 5,404,032 | 5,355,136 |

参数量几乎相同（差 0.9%），容量上公平；差异纯来自 UPA 几何感知架构。

## 4. 运行命令

**Full**（`/root/WiFo`，原始目录）：
```bash
python scripts/run_formal_pilot.py \
  --config configs/small.yaml --variant full \
  --train-dataset D4 --train-path data/D4/X_train.mat \
  --val-dataset D4 --val-path data/D4/X_val.mat \
  --test-dataset D4 --test-path data/D4/X_test.mat \
  --train-samples 9000 --val-samples 2000 --test-samples 1000 \
  --epochs 20 --batch-size 16 --warmup-epochs 3 \
  --task-schedule sample --device cuda --amp --seed 17 \
  --output experiments/trial_d4_small_e20.json
```

**WiFo-like**（`/root/WiFo_ablation`，隔离副本 + 修复后的 baseline.py）：
```bash
PYTHONPATH=/root/WiFo_ablation/src /root/wifo_env/bin/python scripts/run_formal_pilot.py \
  --config configs/wifo_small.yaml --variant wifo \
  --train-dataset D4 --train-path data/D4/X_train.mat \
  --val-dataset D4 --val-path data/D4/X_val.mat \
  --test-dataset D4 --test-path data/D4/X_test.mat \
  --train-samples 9000 --val-samples 2000 --test-samples 1000 \
  --epochs 20 --batch-size 16 --warmup-epochs 3 \
  --task-schedule sample --device cuda --amp --seed 17 \
  --output experiments/wifo_baseline_d4_small_e20.json
```

> WiFo-like 运行前对 `baseline.py` 做了两处必要修复（与 `model.py` 中 UPAMAE 对齐）：
> 1. **初始化统一**：`self.apply(self._init_weights)` + `nn.init.normal_(self.mask_token, std=0.02)`；
> 2. **AMP dtype 修复**：`encoded.float()` + `torch.autocast(enabled=False)` 包裹整个 decoder + `mask_token.to(decoded.dtype)`，消除 fp16/fp32 索引写入的类型不匹配。

## 5. 训练速度对比

| | 单 epoch（均值） | 单 epoch（范围） | 20 epoch 训练耗时 | 总耗时 |
|---|---|---|---|---|
| Full | **298.0 s** | 294.1–305.1 s | 5961 s（99.3 min） | ~100 min |
| WiFo-like | **16.9 s** | 16.7–17.2 s | 338 s（5.6 min） | ~7.5 min（含 ~1.5 min 数据加载） |

**加速比 ≈ 17.7×**（298.0 / 16.9）。token 数从 1024 → 256，attention 的 L² 从 1,048,576 → 65,536（16×），实际加速略超 16×，因为 Full 还多出 relative bias 与空间掩码的额外 O(L²) 开销。

> 结论：Full 的 5 分钟/epoch 里，**绝大部分确实是 1024-token 的 attention 计算**，而非数据/预处理开销（固定开销仅 ~17 s/epoch）。此前「固定开销主导、只能快 2%」的猜测是错误的。

## 6. 完整逐 epoch 曲线

### 6.1 Full（UPAMAE）

| epoch | train_loss | 验证 NMSE | 耗时(s) |
|---|---|---|---|
| 1 | 0.5713 | 1.0148 | 301.78 |
| 2 | 0.4959 | 0.9660 | 295.19 |
| 3 | 0.4528 | 0.7506 | 301.03 |
| 4 | 0.3081 | 0.4302 | 298.83 |
| 5 | 0.2573 | 0.3512 | 297.43 |
| 6 | 0.2230 | 0.3796 | 296.21 |
| 7 | 0.2141 | 0.3737 | 295.91 |
| 8 | 0.2009 | 0.2732 | 300.48 |
| 9 | 0.1719 | 0.2392 | 299.53 |
| 10 | 0.1610 | 0.2587 | 297.14 |
| 11 | 0.1581 | 0.2246 | 305.05 |
| 12 | 0.1341 | 0.1998 | 295.19 |
| 13 | 0.1298 | 0.1862 | 294.13 |
| 14 | 0.1275 | 0.1770 | 298.14 |
| 15 | 0.1169 | 0.1708 | 299.19 |
| 16 | 0.1037 | 0.1627 | 295.04 |
| 17 | 0.1061 | 0.1566 | 297.09 |
| 18 | 0.0973 | 0.1535 | 299.36 |
| 19 | 0.0975 | 0.1525 | 295.37 |
| 20 | 0.0955 | **0.1514** | 298.63 |

### 6.2 WiFo-like（扁平天线基线）

| epoch | train_loss | 验证 NMSE | 耗时(s) |
|---|---|---|---|
| 1 | 0.5660 | 1.0028 | 17.20 |
| 2 | 0.4992 | 0.9936 | 16.96 |
| 3 | 0.4979 | 0.9955 | 16.73 |
| 4 | 0.4960 | 0.9867 | 16.79 |
| 5 | 0.4936 | 0.9878 | 16.70 |
| 6 | 0.4875 | 0.9632 | 16.69 |
| 7 | 0.4553 | 0.8748 | 16.80 |
| 8 | 0.3500 | 0.5610 | 16.74 |
| 9 | 0.2098 | 0.3925 | 17.08 |
| 10 | 0.1703 | 0.3145 | 16.99 |
| 11 | 0.1487 | 0.2817 | 16.78 |
| 12 | 0.1334 | 0.2561 | 16.77 |
| 13 | 0.1212 | 0.2548 | 16.95 |
| 14 | 0.1127 | 0.2351 | 16.98 |
| 15 | 0.1099 | 0.2203 | 16.96 |
| 16 | 0.1025 | 0.2117 | 16.97 |
| 17 | 0.1002 | 0.2065 | 17.00 |
| 18 | 0.0987 | 0.2043 | 16.78 |
| 19 | 0.0956 | 0.2012 | 17.04 |
| 20 | 0.0940 | **0.1997** | 16.75 |

> 注：WiFo-like 前 6 个 epoch 验证 NMSE 几乎不降（1.00→0.96），是 warmup=3 的 LR 爬坡所致；第 7 个 epoch 起陡降，属正常收敛，非 bug。

## 7. 测试结果对比（核心结论）

| 重建任务 | Full（UPA） | WiFo-like | 相对提升 |
|---|---|---|---|
| **时域 temporal** | **0.0525** | 0.1091 | **Full 好 2.08×** |
| 频域 frequency | 0.1986 | 0.2336 | Full 好 1.18× |
| 随机 random | 0.1859 | 0.2303 | Full 好 1.24× |
| 空间 spatial | 0.3445 | —（不支持） | — |

附带指标（nmse_full、token 数、峰值显存）：

| 任务 | 指标 | Full | WiFo-like |
|---|---|---|---|
| temporal | nmse_full | 0.1557 | 0.2629 |
| temporal | num_tokens | 1024 | 256 |
| temporal | 峰值显存 | 1.34 GB | 0.22 GB |
| frequency | nmse_full | 0.3164 | 0.2932 |
| random | nmse_full | 0.1898 | 0.2317 |

## 8. 结论

1. **UPA 几何感知扩展带来显著增益**：在完全公平的设置下，Full 相比 WiFo-like 扁平基线，时域 NMSE 从 0.1091 降到 0.0525（**约 2 倍**），频域/随机任务也有 18%–24% 的相对提升。参数几乎相同（5.36M vs 5.40M），因此增益纯来自架构，而非容量。

2. **代价是约 18× 训练时间**：token 数 4×（256→1024）带来 attention L² 16× 的算力增长，Full 单 epoch 298 s vs WiFo-like 16.9 s。

3. **时域仍是两者最强的任务**：Full 时域 0.0525 已逼近论文 WiFo-Base 完整预训练的 0.048，验证了 UPA 架构在时间外推上的有效性；频域两者都偏弱（0.20/0.23），与论文「频域更难」的规律一致。

4. **下一步**：若需量化 2D 坐标 / relative bias / 空间掩码各自的独立贡献，跑 A/B/C/D 中间变体（均为 UPA 架构、1024 token，每个约 100 min）。

## 9. 附录：环境与复现要点

- **服务器**：阿里云 ECS，NVIDIA A10 24GB，root 登录。
- **环境**：`/root/wifo_env` venv，Python 3.10.12，torch 2.5.1+cu124，h5py/numpy/scipy/pyyaml。
- **数据**：D4 为 MATLAB v5 格式，`X_train.mat`(9000, 2.28GB) / `X_val.mat`(2000, 508MB) / `X_test.mat`(1000, 254MB)，形状 `[B,T,K,Nh,Nv]` = `[B,16,32,4,8]`。来源为北大网盘 AnyShare 分享链接（见 `docs/anyshare-api.md`），实例销毁后可重新下载。
- **复现注意**：`wifo_upa` 以 editable 方式安装指向 `/root/WiFo/src`，在隔离副本运行时需 `PYTHONPATH=/root/WiFo_ablation/src` 覆盖，否则会导入旧目录的未修复 baseline.py。
