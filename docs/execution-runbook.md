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
archive linked in `docs/references.md`, then D17-D19 held-out test splits.
Before the full 200-epoch campaign:

1. Verify CUDA with Little.
2. Measure one batch of the largest `4x8`, `T=24`, `K=128` configuration.
3. Start with `configs/small.yaml`, batch size 8 or 16, and sequential tasks.
4. Save a checkpoint at least every epoch and evaluate D17-D18 before D19 data
   is available.

The public Hugging Face repository contains test files only. The full training
archive must be downloaded manually from the official PKU cloud link because
that service is an interactive web application rather than a stable direct
download endpoint.
