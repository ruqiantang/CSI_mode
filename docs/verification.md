# Verification Record

Date: 2026-09-20

## Automated Verification

- Unit tests: `64 passed`
- Package smoke test: passed
- Python compilation: passed
- CPU AMP training and evaluation: passed

## Real MATLAB CSI Subset

The data files are intentionally not committed. Reproduce the local downloads
with:

```bash
python scripts/download_data.py --datasets D1 D3 D4 D5 D8 --output data
```

Then reproduce a small real-data run with, for example:

```bash
python scripts/run_ablation.py \
  --variant full \
  --source mat \
  --mat-path data/D1/X_test.mat \
  --dataset D1 \
  --samples 4 \
  --batch-size 2 \
  --epochs 1 \
  --task-schedule sample
```

Each run used the Full V1 configuration, four samples, batch size two, one
epoch, and `task_schedule="sample"` on CPU. These are engineering smoke runs,
not performance experiments; the losses and NMSE values are not convergence
results.

| Dataset | UPA | Training loss | Masked NMSE | Full NMSE | Estimated forward FLOPs |
|---|---:|---:|---:|---:|---:|
| D1 | `1 x 4` | 3.417928 | 2.207769 | 2.301784 | 31,305,216,000 |
| D3 | `1 x 8` | 3.119844 | 3.574919 | 3.417501 | 18,288,717,824 |
| D5 | `2 x 2` | 2.484904 | 2.389826 | 2.542227 | 12,750,077,952 |
| D8 | `4 x 4` | 2.877052 | 2.281345 | 2.608614 | 46,923,694,080 |
| D4 | `4 x 8` | 3.038674 | 2.199173 | 2.360514 | 46,923,694,080 |

The selected datasets cover a single-row array, a long column, a small square
array, a conventional square array, and the largest training UPA.

The flattened WiFo baseline was also verified with real `complex128` MATLAB
data using `D1`, two samples, batch size two, and one CPU epoch. This verifies
the loader-to-baseline dtype conversion path. The UPA and baseline use separate
FLOPs estimators, and sequential schedules accumulate FLOPs per task with that
task's visible-token ratio.

## D17-D18 Public Data Smoke Verification

The public D17 and D18 `X_test.mat` files were downloaded from Hugging Face and
not committed. D17 is a MATLAB v5 file and D18 is a MATLAB v7.3/HDF5 file.
Both loaded through `load_mat_csi()` as `[1000,T,K,Nh,Nv]` complex tensors with
finite real and imaginary components.

The following two-sample, one-epoch, CPU runs verify the data-to-model path and
backward pass. They are not zero-shot performance results because the models are
newly initialized and not pretrained.

| Dataset | Model | Training loss | Masked NMSE | Full NMSE | Training FLOPs | Evaluation FLOPs |
|---|---|---:|---:|---:|---:|---:|
| D17 | Full UPA | 2.004032 | 6.639937 | 6.245722 | 108,529,631,232 | 19,744,604,160 |
| D18 | Full UPA | 1.977585 | 6.288396 | 6.386303 | 185,140,690,944 | 36,259,676,160 |
| D17 | WiFo-like baseline | 0.718339 | 1.505134 | 1.511122 | 12,575,490,048 | 3,314,466,816 |
| D18 | WiFo-like baseline | 0.869146 | 1.426712 | 1.440792 | 20,400,488,448 | 5,401,657,344 |

## Formal D4-to-D17 Pilot

The official D4 `X_train.mat` and `X_val.mat` files and the public D17
`X_test.mat` file were used for a controlled pilot. The data files were not
committed. Both models used 512 training samples, 32 validation samples, 64
held-out D17 test samples, two epochs, batch size two, and seed 17.

| Model | Best D4 validation NMSE | D17 random | D17 temporal | D17 frequency | D17 spatial | Shared average |
|---|---:|---:|---:|---:|---:|---:|
| WiFo-like baseline | 0.9971673 | 1.0008496 | 1.0008772 | 1.0007424 | N/A | 1.0008230 |
| Full UPA | 0.9965379 | 1.0006615 | 1.0007602 | 1.0011936 | 1.0043856 | 1.0008718 |

Full UPA improved D4 validation but was slightly worse on the D17 shared-task
average. This is not a zero-shot result and is not decision-grade; see
`docs/formal-pilot.md`.

## Not Yet Verified

- CUDA forward/backward and CUDA AMP.
- Formal D1-D16 pretraining and D17-D19 zero-shot performance.
- Public D19 MATLAB data: the public Hugging Face listing currently provides
  D1-D18 test files only.
- The official PKU cloud D1-D16 archive is available, but all 16 formal
  training datasets have not yet been downloaded and a mixed-dataset pretraining
  run has not yet been executed.

## Reproducibility Entry Points

`scripts/verify_cuda.py` is the required entry point on a CUDA machine.
`scripts/run_ablation.py --save-checkpoint` writes a resumable checkpoint, and
`scripts/evaluate_checkpoint.py` reports task-specific zero-shot metrics.
- The FLOPs estimate is analytic and excludes I/O, Python overhead, loss
  reduction, and checkpointing.
