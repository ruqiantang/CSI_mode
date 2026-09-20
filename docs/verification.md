# Verification Record

Date: 2026-09-20

## Automated Verification

- Unit tests: `58 passed`
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

## Not Yet Verified

- CUDA forward/backward and CUDA AMP.
- Full training-scale convergence and D17-D19 zero-shot evaluation.
- The FLOPs estimate is analytic and excludes I/O, Python overhead, loss
  reduction, and checkpointing.
