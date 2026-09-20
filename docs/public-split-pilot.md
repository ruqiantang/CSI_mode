# Public-Split Pilot

This pilot was created as a stopgap while the official D1-D16 training archive
was believed to be unavailable. It splits the public D17 `X_test.mat` file
into deterministic train, validation, and test subsets, then compares the
WiFo-like baseline with the Full UPA model on the same split.

## Why This Split Exists

The public Hugging Face dataset provides only `X_test.mat` for D1-D18. The
official PKU cloud archive has since been verified as active and contains
D1-D16 formal training and validation splits, so use
`scripts/run_formal_pilot.py` or `scripts/pretrain.py` for new controlled
experiments. This document preserves the earlier same-distribution pilot.

Because the pilot reuses the public test file, it is **not** the paper's
zero-shot protocol and must not be reported as a zero-shot result. It is only
a controlled, same-distribution sanity check for whether the UPA-aware design
shows any early signal on the original CSI data.

## Command

Run from the repository root after downloading D17:

```bash
.venv/bin/python scripts/download_data.py --datasets D17 --output data

.venv/bin/python scripts/run_public_split_pilot.py \
  --config configs/pilot_wifo.yaml \
  --variant wifo \
  --dataset D17 \
  --mat-path data/D17/X_test.mat \
  --train-samples 128 \
  --val-samples 32 \
  --test-samples 64 \
  --epochs 5 \
  --batch-size 2 \
  --warmup-epochs 1 \
  --output experiments/pilot_d17_wifo_128_32_64_e5.json

.venv/bin/python scripts/run_public_split_pilot.py \
  --config configs/pilot_full.yaml \
  --variant full \
  --dataset D17 \
  --mat-path data/D17/X_test.mat \
  --train-samples 128 \
  --val-samples 32 \
  --test-samples 64 \
  --epochs 5 \
  --batch-size 2 \
  --warmup-epochs 1 \
  --output experiments/pilot_d17_full_128_32_64_e5.json
```

The split seed is fixed at `17`. Model selection uses the average masked NMSE
over the three tasks shared by both models: `random`, `temporal`, and
`frequency`. The Full model is additionally evaluated on `spatial`, but that
task is not part of the shared selection metric.

## D17 Pilot Results

Configuration: 128 train, 32 validation, 64 test samples, 5 epochs, batch size
2, and best-epoch selection on the validation set.

| Model | Random NMSE | Temporal NMSE | Frequency NMSE | Spatial NMSE |
|---|---:|---:|---:|---:|
| WiFo-like baseline | 1.0010245 | 1.0009179 | 1.0009426 | N/A |
| Full UPA | 1.0009044 | 1.0006885 | 1.0008865 | 1.0009442 |

Average over the three shared tasks:

- WiFo-like baseline: `1.0009617`
- Full UPA: `1.0008265`

The Full UPA model is slightly lower on all three shared tasks in this run,
but the margin is only about `1.35e-4` in absolute NMSE. That is too small to
treat as a reliable improvement without repeated seeds, more data, or the
official training split.

## Interpretation

This pilot suggests the UPA-aware model is not obviously worse on the original
data once the training subset is increased, but it does **not** establish a
meaningful gain. The absolute NMSE values are all close to `1.0`, which is
consistent with the model still struggling to reconstruct held-out CSI from a
small public-test-only split.

For a decision-grade comparison, use the official D1-D16 training data, then
evaluate the saved checkpoint on D17-D18 with `scripts/evaluate_checkpoint.py`.
