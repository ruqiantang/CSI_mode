# Verification Record

Date: 2026-09-20

## Automated Verification

- Unit tests: `55 passed`
- Package smoke test: passed
- Python compilation: passed
- CPU AMP training and evaluation: passed

## Real MATLAB CSI Subset

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

## Not Yet Verified

- CUDA forward/backward and CUDA AMP.
- Full training-scale convergence and D17-D19 zero-shot evaluation.
- The FLOPs estimate is analytic and excludes I/O, Python overhead, loss
  reduction, and checkpointing.
