# Execution Runbook

This runbook separates locally reproducible checks from work that requires an
external CUDA machine and the official pretraining archive.

## 1. CUDA Forward, Backward, and AMP

Install the repository and run from the repository root on a CUDA machine:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/verify_cuda.py --config configs/little.yaml \
  --dataset D5 --samples 2 --batch-size 1 --amp
```

The script fails if CUDA is unavailable, if a forward/backward pass fails, if
AMP produces non-finite metrics, or if the CUDA peak-memory query fails. Run it
without `--amp` for a plain CUDA control. Use `configs/full.yaml` only after
the Little configuration passes.

## 2. Public MATLAB Test Data

The public Hugging Face listing contains `X_test.mat` for D1-D18 only:

```bash
.venv/bin/python scripts/download_data.py --datasets D17 D18 --output data
```

The script now skips existing nonempty files and explicitly rejects D19 rather
than silently requesting a missing URL. D19 must come from the official
pretraining archive; synthetic D19 tensors are shape-contract tests, not
zero-shot performance results.

The current PKU cloud pretraining link in the official README is expired.
Follow `https://github.com/PKU-PCNI/WiFo/issues/10` for an updated link. The
WiFo-2 D17-D19 `.pt` files are not substitutes for the original WiFo D17-D19
datasets.

### Public-Split Pilot

Because the public repository contains test files only, you can run a
deterministic train/validation/test split on D17 as a same-distribution pilot:

```bash
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

Repeat with `--config configs/pilot_wifo.yaml --variant wifo` for the baseline.
See `docs/public-split-pilot.md` for the current result and its limitations.
This is not a zero-shot evaluation.

## 3. Train a Pilot and Save a Checkpoint

A small real-data pilot can be trained and saved with:

```bash
.venv/bin/python scripts/run_ablation.py \
  --variant full --source mat --mat-path data/D5/X_test.mat \
  --dataset D5 --samples 8 --batch-size 2 --epochs 1 \
  --task-schedule sequential --device cuda --amp \
  --save-checkpoint checkpoints/full_d5_pilot.pt
```

The checkpoint contains the model, optimizer, optional scheduler, task
schedule, ratios, and model type. Checkpoints are ignored by git.

## 4. Zero-Shot Task Evaluation

Evaluate a saved checkpoint on each available task:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --config configs/full.yaml \
  --checkpoint checkpoints/full_d5_pilot.pt \
  --dataset D17 --source mat --mat-path data/D17/X_test.mat \
  --samples 8 --batch-size 2 --device cuda --amp
```

Repeat for D18. The output is JSON with separate random, temporal, frequency,
and, for the Full model, spatial metrics. Use the same task ratios and
schedule recorded in the checkpoint for strict comparisons.

## 5. Formal Pretraining

The formal experiment uses D1-D16 training splits from the official PKU cloud
archive, then D17-D19 held-out test splits. The official archive was released,
but its current public link is expired; see `docs/references.md` for the
access issue.
After extracting the archive, run mixed-dataset training with:

```bash
.venv/bin/python scripts/pretrain.py \
  --config configs/small.yaml \
  --data D1=data/D1/X_train.mat \
  --data D2=data/D2/X_train.mat \
  --data D3=data/D3/X_train.mat \
  --data D4=data/D4/X_train.mat \
  --data D5=data/D5/X_train.mat \
  --data D6=data/D6/X_train.mat \
  --data D7=data/D7/X_train.mat \
  --data D8=data/D8/X_train.mat \
  --data D9=data/D9/X_train.mat \
  --data D10=data/D10/X_train.mat \
  --data D11=data/D11/X_train.mat \
  --data D12=data/D12/X_train.mat \
  --data D13=data/D13/X_train.mat \
  --data D14=data/D14/X_train.mat \
  --data D15=data/D15/X_train.mat \
  --data D16=data/D16/X_train.mat \
  --batch-size 8 --epochs 200 --warmup-epochs 5 \
  --task-schedule sequential --device cuda --amp \
  --checkpoint checkpoints/small_d1_d16.pt
```

Resume an interrupted run by adding `--resume`; the script refuses to overwrite
an existing checkpoint accidentally. `LazyMatCSIDataset` reads MATLAB v7.3
samples lazily, and `ShapeBucketBatchSampler` keeps each batch to one CSI shape
while shuffling shape buckets across the epoch.

Before the full 200-epoch campaign:

1. Verify CUDA with Little.
2. Measure one batch of the largest `4x8`, `T=24`, `K=128` configuration.
3. Start with `configs/small.yaml`, batch size 8 or 16, and sequential tasks.
4. Save a checkpoint at least every epoch and evaluate D17-D18 before D19 data
   is available.

The public Hugging Face repository contains test files only. The full training
archive must be downloaded manually once an updated official link is available.
