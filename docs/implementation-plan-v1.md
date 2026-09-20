# V1 Implementation Plan

## 1. Goal

V1 does not target SOTA performance. Its goal is to prove that the complete
UPA-aware modeling chain is correct:

```text
complex CSI
  -> antenna-independent embedding
  -> (t,k,r,c) token
  -> 4D PE
  -> geometry-aware attention
  -> structured masking
  -> MAE reconstruction
  -> complex CSI
```

The chain must have the correct shape, the correct coordinate semantics, the
correct mask semantics, working backpropagation, deterministic restoration,
and a trainable loop. Performance comparison begins only after this loop is
stable.

## 2. Frozen Ablation Variants

| Variant | TF embedding | PE | UPA bias | Spatial pretraining |
|---|---|---|---|---|
| WiFo | `(4,4,4)` | STF `(t,f,s)` | No | No |
| A | `(4,4,1)` | Compatible control `(t,f,s)` | No | No |
| B | `(4,4,1)` | `(t,f,r,c)` | No | No |
| C | `(4,4,1)` | `(t,f,r,c)` | Yes | No |
| D | `(4,4,1)` | `(t,f,r,c)` | Yes | Antenna |
| Full | `(4,4,1)` | `(t,f,r,c)` | Yes | Antenna/row/column/block |

For variant A, the compatible control PE keeps the original three-coordinate
form `(t,f,s)`, where:

```text
s = r*Nv + c
```

It must not use the full 4D `(t,f,r,c)` encoding. This ensures that
`WiFo -> A` isolates the effect of changing the embedding from `(4,4,4)` to
`(4,4,1)`.

The causal chain is:

1. `A > WiFo`: removing cross-antenna convolution is useful.
2. `B > A`: two-dimensional absolute UPA coordinates are useful.
3. `C > B`: relative UPA geometry is useful.
4. `D > C`: spatial reconstruction pretraining is useful.

The configuration mapping is explicit:

```text
allow_spatial_mask=false       -> A, B, and C
allow_spatial_mask=true
spatial_types=("antenna",)     -> D
spatial_types=(all four)       -> Full, Base, Small, and Little
```

## 3. Frozen Repository Layout

```text
src/wifo_upa/
  __init__.py
  geometry.py
  embed.py
  pe.py
  masks.py
  attention.py
  model.py
  baseline.py
  data.py
  train.py
  evaluate.py
  config.py
  smoke_test.py

configs/
  base.yaml
  small.yaml
  little.yaml
  ablation_a.yaml
  ablation_b.yaml
  ablation_c.yaml
  ablation_d.yaml
  full.yaml

tests/
  test_geometry.py
  test_embed.py
  test_masks.py
  test_pe.py
  test_attention.py
  test_model.py
  test_baseline.py
  test_evaluate.py
  test_train_loop.py
  test_data.py

scripts/
  run_ablation.py
  download_data.py
```

`smoke_test.py` belongs in the package, not in `scripts/`. The smoke-test
entry point is:

```bash
python -m wifo_upa.smoke_test
```

`scripts/` contains only experiment and download entry points.

## 4. Geometry API

`geometry.py` performs reversible data arrangement only. It must not contain a
learnable module or projection.

Frozen public functions:

```python
token_index(t, k, r, c, Kp, Nh, Nv)
inverse_token_index(l, Kp, Nh, Nv)
build_coords(Tp, Kp, Nh, Nv)
reshape_upa(x, shape_before, shape_after)
patchify_tf(x, pt, pf)
flatten_token_patches(patches)
unflatten_token_patches(x, T, K, Nh, Nv, pt, pf)
unpatchify_tf(x, T, K, Nh, Nv, pt, pf)
```

`patchify_tf()` returns the structured grid
`[B,Tp,Kp,Nh,Nv,2,pt,pf]`. `flatten_token_patches()` is the explicit model
representation used for `[B,L,2*pt*pf]`.

Token order is always:

```text
(t,k,r,c)
```

with `c` fastest-changing:

```text
l = ((t*Kp + k)*Nh + r)*Nv + c
```

Every consumer uses the same ordering:

- patching and unpatching
- positional encoding
- masks
- relative bias
- decoder restoration
- visualization

The geometry integrity invariant is:

```text
unpatchify_tf(patchify_tf(x), ...) == x
```

## 5. Embedding API

`embed.py` contains:

```python
AntennaIndependentPatchEmbed
```

The frozen tensor contract is:

```text
[B,2,T,K,Nh,Nv]
  -> [B*Nh*Nv,2,T,K,1]
  -> Conv3d(kernel=(pt,pf,1), stride=(pt,pf,1))
  -> [B*Nh*Nv,D,Tp,Kp,1]
  -> [B,Tp,Kp,Nh,Nv,D]
```

The convolution weights are shared by all antennas. `geometry.py` owns the
reversible reshape operations; `embed.py` owns the learnable projection.

## 6. Positional Encoding API

`pe.py` implements the fixed 4D SinCos encoding:

```text
D_t = D_f = D_r = D_c = D/4
```

`D/4` must be even. It also implements the compatible control encoding used by
variant A:

```text
(t,k,s), s = r*Nv + c
```

with widths:

```text
D_t = D_k = floor(D/3)
D_s = D - 2*floor(D/3)
```

The 4D encoding is:

```text
P_i = concat(
  PE_t(t_i),
  PE_k(k_i),
  PE_r(r_i),
  PE_c(c_i),
)
```

## 7. Mask API

All four mask generators return the same type and shape:

```python
mask: torch.BoolTensor  # [B,L]
```

The convention is:

```text
False / 0 = visible
True  / 1 = masked
```

Supported mask types:

- `random`
- `temporal`
- `frequency`
- `spatial`

The spatial generator may internally sample:

- `antenna`
- `row`
- `column`
- `block`

Spatial masks may initially be generated as `[B,Nh,Nv]`, but must be broadcast
through the frozen token ordering before returning:

```text
[B,Nh,Nv] -> [B,Tp,Kp,Nh,Nv] -> [B,L]
```

The encoder is therefore mask-type agnostic:

```python
visible_tokens = tokens[~mask]
```

The batch-shared indices are represented by `MaskLayout`, which exposes
`mask`, `visible_mask`, `visible_ids`, `masked_ids`, `coords`, and token counts
in one validated object.

Spatial visibility is defined over antenna elements:

```text
R_visible_space = |visible antennas| / (Nh*Nv)
```

The invariant is:

```text
R_visible_space >= 0.5
```

For `Nh=1`, row masking is disabled. No strategy may mask every antenna.

## 8. Model Forward API

The model forward signature is:

```python
output = model(
    H,
    mask_type="spatial",
    spatial_type="antenna",
)
```

It returns a dictionary, not only the prediction:

```python
{
    "prediction": H_hat,
    "loss": loss,
    "mask": mask,
    "visible_mask": visible_mask,
    "coords": coords,
    "num_tokens": L,
    "num_visible_tokens": L_vis,
}
```

Definitions:

- `prediction`: complex tensor `[B,T,K,Nh,Nv]`
- `loss`: scalar masked complex MSE
- `mask`: `[B,L]`, true means masked
- `visible_mask`: `[B,L]`, true means visible
- `coords`: `[L,4]` integer coordinates in `(t,k,r,c)` order
- `L = Tp*Kp*Nh*Nv`
- `L_vis = L - mask.sum(dim=1)[0]`

For v1, every mask generator returns the same number of masked tokens within a
batch. Therefore `L_vis` is a single integer and no padded visible-token layout
is needed. If per-sample variable mask counts are introduced later, the packed
encoder layout and `num_visible_tokens` must be redesigned explicitly.

Training uses `output["loss"]`; debugging can inspect `coords`, `mask`, and
`visible_mask`.

## 9. Relative Bias API

`attention.py` contains:

```python
UPARelativePositionBias
GeometryAwareAttention
```

The bias is:

```text
B_h(delta_r,delta_c)
  = b_r^h[delta_r] + b_c^h[delta_c]
```

Each head has independent row and column tables.

Do not permanently cache the complete `[num_heads,L,L]` bias matrix. Because
`L` is already four times the original WiFo token count, storing this matrix
would waste substantial memory across heterogeneous configurations.

Cache only coordinate-derived index tensors:

```text
coords
dr_index
dc_index
```

At forward time compute:

```python
bias = row_bias[:, dr_index] + col_bias[:, dc_index]
```

The result is transient:

```text
[num_heads,L,L]
```

For the official paper's D1-D19 configurations, the training range is:

```text
delta_r in [-3,3]
delta_c in [-7,7]
```

Clamping is only an out-of-range guard. The primary zero-shot experiments must
remain within the covered relative-distance range.

## 10. Training API

The trainer supports:

```text
task_schedule="sample"
task_schedule="sequential"
```

Sample mode selects one task uniformly per batch. Sequential mode runs random,
temporal, frequency, and spatial tasks and averages their four losses.

Both modes must be implemented before ablation experiments. Every result
records the selected schedule.

Training also implements:

- AdamW
- cosine decay
- five-epoch warmup
- checkpoint saving and resuming
- mixed precision when available
- gradient clipping
- finite-loss checks
- NMSE and efficiency metrics

## 11. Development Order

1. `geometry.py`
2. `embed.py`
3. `pe.py`
4. `masks.py`
5. `attention.py` and relative bias
6. `model.py` forward path
7. `baseline.py`
8. synthetic training
9. real-data loader
10. evaluation and ablation

This order validates the core embedding contract before masking or attention.

## 12. V1 Acceptance

The implementation is complete when:

1. `pytest` passes.
2. `python -m wifo_upa.smoke_test` passes.
3. Token counts are correct for all D1-D19 UPA/T/K configurations.
4. For every token:

   ```text
   l = ((t*Kp+k)*Nh+r)*Nv+c
   ```

   and the inverse recovers exactly `(t,k,r,c)`.
5. `unpatchify_tf(patchify_tf(x)) == x`.
6. All masks return `[B,L]`.
7. Spatial masks preserve `R_visible_space >= 0.5`.
8. The model forward and backward pass succeeds on CPU and, when available, GPU.
9. One synthetic training epoch completes with finite loss.
10. One real-data subset epoch completes with finite loss.
11. Checkpoint save and resume restore model and optimizer state.
12. Evaluation reports NMSE, token count, visible-token count, peak memory,
    training time, inference time, and parameter count.
13. The repository contains no datasets, weights, TensorBoard logs, or private
    local paths.

## 13. Minimum MVP

The first MVP includes:

- `geometry.py`
- `embed.py`
- `pe.py`
- `masks.py`
- `model.py`
- `train.py`
- `configs/small.yaml`
- `tests/test_geometry.py`
- `tests/test_embed.py`
- `tests/test_masks.py`
- `tests/test_model.py`

The MVP must specifically assert that:

```python
tokens[b, t, k, r, c]
```

is located at:

```text
l=((t*Kp+k)*Nh+r)*Nv+c
```

This test is mandatory because a coordinate mismatch can still train and reduce
loss while making PE, masks, and relative bias semantically wrong.
