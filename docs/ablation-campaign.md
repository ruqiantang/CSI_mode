# Reduced Ablation Campaign

This campaign is the first six-variant, three-seed ablation on official formal
CSI data. It is a reduced CPU pilot, not the full D1-D16 pretraining protocol.

## Protocol

- Variants: WiFo-like baseline, A, B, C, D, and Full.
- Seeds: `17`, `18`, and `19`.
- Training data: official `D1/X_train.mat` and `D4/X_train.mat`, 64 samples
  from each file.
- Validation data: official `D4/X_val.mat`, 16 samples.
- Test data: public `D17/X_test.mat` and `D18/X_test.mat`, 32 samples each.
- Epochs: `1`.
- Batch size: `2`.
- Task schedule: `sequential`.
- Shared selection metric: average masked NMSE over `random`, `temporal`, and
  `frequency`.
- Device: CPU on Apple M3 Pro.
- Commit at run time: `df4aa3b500b7111dcfe0f00cdd0c33269eb9d7b1`.

Reproduce with:

```bash
.venv/bin/python scripts/run_campaign.py \
  --variants wifo a b c d full --seeds 17 18 19 \
  --train-data D1=data/D1/X_train.mat \
  --train-data D4=data/D4/X_train.mat \
  --val-data D4=data/D4/X_val.mat \
  --test-data D17=data/D17/X_test.mat \
  --test-data D18=data/D18/X_test.mat \
  --train-samples-per-dataset 64 \
  --val-samples-per-dataset 16 \
  --test-samples-per-dataset 32 \
  --epochs 1 --batch-size 2 --task-schedule sequential \
  --output experiments/ablation_d1_d4_to_d17_d18_64_16_32_e1_3seeds.json
```

The experiment JSON is intentionally ignored because experiment outputs are
local artifacts.

## Results

Values are the mean over three seeds. The shared test metric is the average
masked NMSE over random, temporal, and frequency reconstruction.

| Variant | Validation | D17 shared | D18 shared | D17 spatial | D18 spatial |
|---|---:|---:|---:|---:|---:|
| WiFo | 1.0061059 | 1.0062662 | 1.0081864 | N/A | N/A |
| A | 1.0100349 | 1.0108119 | 1.0158257 | N/A | N/A |
| B | 1.0074772 | 1.0081189 | 1.0128658 | N/A | N/A |
| C | 1.0074712 | 1.0081129 | 1.0128605 | N/A | N/A |
| D | 1.0075813 | 1.0078653 | 1.0123407 | 1.0084137 | 1.0101330 |
| Full | 1.0073910 | 1.0080195 | 1.0127561 | 1.0087398 | 1.0101919 |

### D17 and D18 Averages

The WiFo-like baseline has the lowest shared-task NMSE in this reduced run.
Full UPA is worse than the baseline by:

- D17: `0.0017533` NMSE.
- D18: `0.0045697` NMSE.

The seed standard deviation for Full is approximately:

- D17 shared: `0.0001648`.
- D18 shared: `0.0014003`.

The D17 gap is larger than the Full seed standard deviation, but the D18 gap is
only about 3.3 times that standard deviation. This is not a decision-grade
statistical test, but it does not provide evidence of an improvement in this
small-scale setup.

### Ablation Reading

- Moving from WiFo to A makes performance worse in this run. A changes the TF
  embedding while keeping the control PE and no UPA bias.
- Replacing the control PE with the 4D PE (A to B) improves the shared average
  substantially in this run.
- Adding the UPA relative bias (B to C) changes shared-task NMSE by less than
  `6e-6` and does not provide a meaningful accuracy gain.
- Adding spatial pretraining (C to D, or C to Full) gives only small changes
  that are mostly within seed variation.
- Full does not improve over D consistently in this run.

### Cost

For the three shared tasks, averaged over the three seeds:

| Variant | D17 forward FLOPs | D18 forward FLOPs | D17 inference time | D18 inference time |
|---|---:|---:|---:|---:|
| WiFo | 19,743,768,576 | 33,698,217,984 | 0.212 s | 0.286 s |
| A | 142,410,383,360 | 278,591,045,632 | 1.794 s | 4.112 s |
| B | 142,410,383,360 | 278,591,045,632 | 1.674 s | 4.101 s |
| C | 142,410,383,360 | 278,591,045,632 | 2.557 s | 6.008 s |
| D | 142,410,383,360 | 278,591,045,632 | 2.410 s | 5.600 s |
| Full | 142,410,383,360 | 278,591,045,632 | 2.855 s | 6.980 s |

The UPA-aware variants use:

- D17: 1,024 tokens per sample versus 256 for the flattened baseline.
- D18: 1,536 tokens per sample versus 384 for the flattened baseline.

The analytic forward-FLOPs estimate is therefore about 7.2 times higher on D17
and 8.3 times higher on D18 in this run. CPU inference time is 8-24 times
higher depending on dataset and variant. CUDA peak memory remains unverified
because this machine has no CUDA device.

## Interpretation

This run does not support a claim that the UPA-aware model improves WiFo on
the original CSI data under this training budget. The best-supported positive
finding is that the 4D PE recovers much of the loss introduced by the
antenna-independent TF embedding. The UPA relative bias and spatial pretraining
do not show a reliable NMSE improvement at this scale.

The result is still useful because it completes the six-variant ablation
workflow with three seeds and records token counts, visible-token counts,
FLOPs, and timing. A full D1-D16 mixed pretraining run on CUDA is still
required before making a research conclusion.
