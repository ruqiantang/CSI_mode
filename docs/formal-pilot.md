# Formal-Data Pilot

This pilot uses the official D4 training and validation files and evaluates on
the public D17 held-out test file. It is a controlled real-data check of the
UPA-aware pipeline before committing to mixed D1-D16 pretraining on CUDA.

## Protocol

- Training data: official `D4/X_train.mat`, first 512 samples in this run.
- Validation data: official `D4/X_val.mat`, first 32 samples.
- Test data: public `D17/X_test.mat`, first 64 samples.
- Seed: `17`.
- Batch size: `2`.
- Epochs: `2`.
- Task schedule: `sequential`.
- Shared model selection: average masked NMSE over `random`, `temporal`, and
  `frequency`.
- The Full model additionally reports `spatial`, but spatial is not used for
  shared model selection.
- The baseline uses `configs/pilot_wifo.yaml`; Full uses
  `configs/pilot_full.yaml`.

Reproduce the two runs with:

```bash
.venv/bin/python scripts/run_formal_pilot.py \
  --config configs/pilot_wifo.yaml --variant wifo \
  --train-dataset D4 --train-path data/D4/X_train.mat \
  --val-dataset D4 --val-path data/D4/X_val.mat \
  --test-dataset D17 --test-path data/D17/X_test.mat \
  --train-samples 512 --val-samples 32 --test-samples 64 \
  --epochs 2 --batch-size 2 --warmup-epochs 1 \
  --output experiments/formal_d4_to_d17_wifo_512_32_64_e2.json

.venv/bin/python scripts/run_formal_pilot.py \
  --config configs/pilot_full.yaml --variant full \
  --train-dataset D4 --train-path data/D4/X_train.mat \
  --val-dataset D4 --val-path data/D4/X_val.mat \
  --test-dataset D17 --test-path data/D17/X_test.mat \
  --train-samples 512 --val-samples 32 --test-samples 64 \
  --epochs 2 --batch-size 2 --warmup-epochs 1 \
  --output experiments/formal_d4_to_d17_full_512_32_64_e2.json
```

The official MATLAB files are not committed. The D4 validation file stores its
sole variable as `X_test`, and the public D17 test file stores its sole
variable as `X_val`; the loader intentionally reads the sole variable rather
than requiring the variable name to match the filename.

## 128-Sample Warmup

Before the 512-sample run, both models were run with 128 training samples,
32 validation samples, 64 D17 test samples, five epochs, batch size two, and
the same selection rule.

| Model | Best D4 validation NMSE | Random | Temporal | Frequency | Spatial |
|---|---:|---:|---:|---:|---:|
| WiFo-like baseline | 1.0011797 | 1.0015450 | 1.0016645 | 1.0012462 | N/A |
| Full UPA | 0.9982069 | 1.0015563 | 1.0012743 | 1.0022216 | 1.0029484 |

Shared-task averages:

- WiFo-like baseline: `1.0014852`
- Full UPA: `1.0016841`

Full UPA was clearly better on D4 validation, slightly worse on the D17
shared-task average, and the margin was far too small for a decision.

## 512-Sample Result

| Model | Best D4 validation NMSE | Random | Temporal | Frequency | Spatial |
|---|---:|---:|---:|---:|---:|
| WiFo-like baseline | 0.9971673 | 1.0008496 | 1.0008772 | 1.0007424 | N/A |
| Full UPA | 0.9965379 | 1.0006615 | 1.0007602 | 1.0011936 | 1.0043856 |

Shared-task averages:

- WiFo-like baseline: `1.0008230`
- Full UPA: `1.0008718`

Full UPA was better on the D4 validation split by about `6.3e-4` NMSE. On
D17, it was better on random and temporal masking, worse on frequency
masking, and slightly worse overall by about `4.9e-5` NMSE on the shared-task
average. The absolute difference is far below any reasonable decision
threshold for a one-seed, two-epoch pilot.

## Interpretation

This is **not** the paper's zero-shot protocol and must not be reported as a
zero-shot performance result. It trains a newly initialized model on only one
formal dataset for two epochs, uses one seed, and evaluates only the first 64
D17 test samples on CPU.

The result is useful for checking that the official train/validation loader,
model selection, and held-out D17 evaluation path work end to end. A decision
about whether UPA-aware modeling improves WiFo requires mixed D1-D16
pretraining, repeated seeds, D17-D18 evaluation, and the separately obtained
original D19 split.
