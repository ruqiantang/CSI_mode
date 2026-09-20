# Experiment Plan

## 1. Research Questions

1. Does removing cross-antenna convolution from the embedding improve CSI
   prediction, harm it, or leave it unchanged?
2. Does explicit `(r,c)` UPA geometry improve representation over flattened
   antenna indices?
3. Does relative UPA bias help antennas exchange information efficiently?
4. Can a model reconstruct missing antennas from partial-array observations?
5. Does the geometry-aware model generalize to unseen UPA shapes?

## 2. Datasets

Use the official D1-D16 datasets for pre-training and D17-D19 for zero-shot
evaluation.

The public D1-D18 configurations include:

- `T=16` or `24`
- `K=32`, `64`, or `128`
- UPAs from `1x4` to `4x8`
- seven carrier frequencies and multiple 3GPP scenarios

The first implementation should require exact divisibility by `pt=4` and
`pf=4`. No silent padding.

## 3. Ablation Matrix

| Run | Conv patch | Position | UPA bias | Spatial mask | Purpose |
|---|---|---|---|---|---|
| WiFo | `(4,4,4)` | `(t,f,s)` | No | No | Official baseline |
| A | `(4,4,1)` | `(t,f,s)` | No | No | Remove cross-antenna convolution |
| B | `(4,4,1)` | `(t,f,r,c)` | No | No | Add 2D UPA coordinates |
| C | `(4,4,1)` | `(t,f,r,c)` | Yes | No | Add relative geometry |
| D | `(4,4,1)` | `(t,f,r,c)` | Yes | Antenna | Add partial-array reconstruction |
| Full | `(4,4,1)` | `(t,f,r,c)` | Yes | Ant/Row/Col/Block | Complete first model |

Interpretation:

- If `WiFo -> A` improves, flattened-antenna convolution is a possible
  inductive-bias problem.
- If A does not improve but B or C does, explicit geometry matters more than
  the convolutional change.
- If D improves spatial reconstruction without degrading temporal and frequency
  prediction, the partial-array objective is useful.

## 4. Training Configuration

Start with WiFo-Small:

- `D_enc = D_dec = 256`
- encoder depth `6`
- decoder depth `4`
- heads `8`
- MLP ratio `2`
- patch `(4,4,1)`
- batch size `8` or `16`
- optimizer AdamW
- base learning rate `5e-4`
- weight decay `0.05`
- cosine decay
- five-epoch warmup
- 200 epochs target, with early smoke tests at 5-10 epochs

If WiFo-Small fits and converges, move to WiFo-Base only after recording peak
memory and runtime for a representative `4x8` sample.

Do not warm-start any run in the ablation table. Warm starting would conflate
the official pretraining distribution with the architectural change.

### Task Schedule

Implement both modes before running the ablation:

- `task_schedule="sample"`: uniformly sample one of random, temporal,
  frequency, and spatial per batch. Use this for early structural validation.
- `task_schedule="sequential"`: run all four tasks on each batch and average
  their losses. Use this for the final strict comparison with WiFo.

The sampled schedule substantially reduces training cost. It is not equivalent
to sequential optimization, so the schedule must be recorded with every
result.

## 5. Task Ratios

Use the official ratios first:

- random: `0.85`
- temporal: `0.50`
- frequency: `0.50`

For spatial masking, run two settings:

- antenna mask ratio `0.25`
- antenna mask ratio `0.50`

Define spatial visibility over UPA elements:

```text
R_visible_space = |visible antennas| / (Nh*Nv)
```

The spatial mask generator must enforce `R_visible_space >= 0.5`. For row,
column, and block strategies, sample masks until the invariant holds; do not
arbitrarily truncate a generated mask because that changes the intended
geometry.

## 6. Metrics

Primary:

- NMSE over masked positions for each task and dataset.

Secondary:

- peak CUDA memory
- average training step time
- average inference time
- encoder and decoder sequence length
- estimated FLOPs
- parameter count
- per-UPA-shape breakdown

For zero-shot geometry, hold out one or more UPA shapes from training and
report absolute 4D PE extrapolation and relative-bias behavior within the
distance range covered by the trained tables. Delta clamping is only an
out-of-range fallback and must not be presented as evidence of larger-array
geometric extrapolation.

## 7. Success Criteria

The first prototype is successful if:

1. It trains without shape or mask-order errors across all 16 configurations.
2. It preserves temporal and frequency prediction within a small tolerance of
   the strongest baseline.
3. It improves partial-antenna reconstruction.
4. Its compute overhead is quantified rather than hidden.

The study is not a success merely because one aggregate NMSE number improves.
The ablation must identify which architectural change causes the effect.

## 8. Implementation Milestones

1. **Shape harness:** synthetic complex tensors for every supported UPA shape;
   assert convolution, token, mask, decoder, and unpatchify round trips.
2. **Ablation A:** antenna-independent convolution with original flattened
   spatial coordinate.
3. **Ablation B:** exact 4D positional encoding.
4. **Ablation C:** UPA relative bias.
5. **Ablation D:** antenna masking.
6. **Full:** all spatial strategies.
7. **Training loop:** task sampling, optimizer, schedule, checkpointing, and
   evaluation.
8. **Small-scale experiments:** short runs to validate stability.
9. **Full experiments:** 200-epoch runs after smoke tests.

## 9. Compute Budget

Before full training, measure one batch of each largest configuration:

- `T=24`, `K=128`, `Nh=4`, `Nv=8`
- `T'=6`, `K'=32`
- tokens `L = 6*32*32 = 6144`

Use this measurement to choose batch size and gradient accumulation. Stop
scaling if the attention implementation cannot hold the activation memory.

If the full model is too expensive, do not immediately redesign the model.
First finish WiFo-Small and Little experiments so the architectural signal is
known before optimizing sequence length.

When using `task_schedule="sequential"`, multiply the per-batch reconstruction
cost by four. Estimate both schedules separately:

```text
sample: one task forward/backward pass per batch
sequential: four task forward/backward passes per batch
```
