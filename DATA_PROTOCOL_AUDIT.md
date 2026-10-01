# DATA_PROTOCOL_AUDIT.md

> 正式大规模训练前的数据协议审计。**已通过官方 WiFo 源码核实**（本地备份 `WiFo_backup_20260927/src/`，对应官方 `github.com/liuboxun/WiFo`）。
> 日期：2026-10-01

---

## 0. 官方源码核实结论（一句话）

| 问题 | 结论 | 证据 |
|---|---|---|
| 20 dB 噪声 baked 还是 runtime | **runtime** | `model.py:1095-1098`：`snr_db=20; noise=randn_like(imgs)*sqrt(mean(imgs²)*10**(-2))`，在 `forward()` 内每次前向都加 |
| dataset-wise 标准化 baked 还是 runtime | **baked**（发布前已标准化） | 全 `src/` 无 mean/var 标准化代码；`normalize_csi_batch` 只是 real/imag 格式转换；论文用词「pre-standardized」 |
| 1000 validation vs 2000 inference 文件映射 | **文件名与内容交叉**（见 §1） | 官方 `DataLoader.py` 读 `X_test.mat` 却取变量 `X_val` |

---

## 1. split 文件映射 —— ✅ 已解决

**论文原文**：每数据集 9000 training / 1000 validation / 2000 inference（共 12000）。

**公开文件大小比例**：`X_train : X_val : X_test = 9 : 2 : 1`（D1–D16 一致）→ X_train=9000、X_val=2000、X_test=1000。

**官方源码证据**（`WiFo_backup_20260927/src/DataLoader.py:32-35`）：
```python
folder_path_test = '../dataset/{}/X_test.mat'.format(dataset)
X_test = hdf5storage.loadmat(folder_path_test)
X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex))...
```
→ 文件 `X_test.mat` 内部变量名是 `X_val`。（`references.md` 也记录「D4 `X_val.mat` 内部变量名是 `X_test`」。）

**结论（文件名 ↔ 内容交叉）**：

| 文件（名字） | 大小（样本数） | 内部变量名 | 论文角色 |
|---|---|---|---|
| `X_train.mat` | 9000 | `X_train` | training |
| `X_val.mat` | **2000** | `X_test` | **inference** |
| `X_test.mat` | **1000** | `X_val` | **validation** |

→ **文件名被对调了**：`X_val.mat` 实为 inference（2000），`X_test.mat` 实为 validation（1000）。

**正式 pipeline 映射**：
- **预训练集合 = `X_train.mat`(9000) + `X_test.mat`(1000 validation) = 10000/数据集 × 16 = 160000**。
- **D1–D16 评估 = `X_val.mat`(2000 inference)**。

> 即：`--data` 要接 `X_train.mat` **和** `X_test.mat`（不是 `X_val.mat`）；D1–D16 评估用 `X_val.mat`。

---

## 2. 预训练集合组成 —— ✅ 已解决

- 论文：pre-trained on all training + validation samples = **160k**。
- 预训练 = 9000 training + 1000 validation = 10000/数据集。
- 结合 §1 文件映射：预训练 = `X_train.mat` + `X_test.mat`（validation）。

---

## 3. 标准化 + 20dB 噪声 —— ✅ 已解决

- **标准化 = baked**：论文「pre-standardized using mean and variance of the corresponding dataset」；官方 `src/` 无任何 mean/var 标准化（`normalize_csi_batch` 仅 real/imag 格式转换）。→ 发布的 MAT 已标准化，data pipeline **不需要**再加标准化。
- **20 dB 噪声 = runtime**：`model.py:1095-1098`（`forward()` 内）每次前向对输入加 `snr_db=20` 高斯噪声（`noise_power = mean(imgs²) × 10^(-2)`，按 batch 功率）。→ **需在 data/model pipeline 的 runtime 加 20dB 噪声**，训练与推理都加。

---

## 4. D17 / D18 zero-shot —— ⚠️ 需注意（不阻塞 benchmark，但阻塞最终 eval）

- 官方 README：**inference 数据集**指向 `huggingface.co/datasets/liuboxun/WiFo-dataset`（与本地 `download_data.py` 用的 `pku-pcni-lab/RF-only_channel_dataset_for_WiFo` **不同**）。
- 本地 D17/D18 只有 `X_test.mat`（`verification.md` 实测 `[1000,...]`）。
- 结合 §1 的交叉命名：D17/D18 的 `X_test.mat`(1000) 可能是 validation，真正的 inference(2000) 需从官方 `WiFo-dataset` 确认。**正式 zero-shot 前需核实 D17/D18 的 inference 文件与样本数。**

---

## 5. 已确认需要落地的 pipeline 改动

1. **预训练集合** = `X_train.mat` + `X_test.mat`（validation），共 160k。
2. **D1–D16 评估** = `X_val.mat`（inference，2000）。
3. **runtime 加 20 dB 噪声**（对应 `model.py` 的 `snr_db=20`，按 batch 功率，训练+推理都加）。
4. **不加标准化**（数据已 baked）。
5. **Antenna Reconstruction NMSE**：在 D1–D16 inference 集（`X_val.mat`）与 D17/D18 zero-shot 集上，用 antenna mask ratio=0.25，评估被掩天线位置的 NMSE（与 temporal/frequency 同评估协议）。

---

## 6. 正式训练阻塞项（剩余）

| # | 项 | 状态 |
|---|---|---|
| 1 | split 文件映射 | ✅ 已解决（§1） |
| 2 | 预训练集合组成 | ✅ 已解决（§2，160k = train+validation） |
| 3 | 标准化 / 噪声位置 | ✅ 已解决（§3，baked / runtime） |
| 4 | sequential 显存优化 | ✅ 已完成（train.py 逐 task backward） |
| 5 | D17/D18 zero-shot 的 inference 文件与样本数 | ⚠️ 需从官方 `WiFo-dataset` 核实（不阻塞 GPU benchmark，阻塞最终 zero-shot 结果） |

> 不启动 200 epoch 正式训练。
