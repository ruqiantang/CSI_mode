# Model Specification

## 1. Objective

The model is defined as:

```text
Antenna-independent TF embedding
  + geometry-aware UPA Transformer
```

It separates time-frequency feature extraction from inter-antenna spatial
modeling:

```text
CSI
  -> antenna-independent time-frequency embedding
  -> 4D positional encoding
  -> UPA relative position bias
  -> MAE Transformer encoder
  -> decoder
  -> CSI reconstruction
```

In WiFo, the `Conv3d` embedding also performs part of the spatial modeling over
the flattened antenna axis. Here, the convolution only extracts features within
one antenna. The Transformer is the only module that models relationships
between antennas. This shift is the main research narrative, not a minor
positional-encoding adjustment.

## 2. Input Tensor

Let the complex CSI be:

```text
H in C^(B x T x K x Nh x Nv)
```

where `T` is time, `K` is frequency, and the base station has an `Nh x Nv`
UPA. The model consumes real and imaginary components as two input channels:

```text
X in R^(B x 2 x T x K x Nh x Nv)
X[:, 0] = Re(H)
X[:, 1] = Im(H)
```

Each antenna is temporarily treated as an independent sample for the shared
time-frequency embedding:

```text
X_a in R^((B*Nh*Nv) x 2 x T x K x 1)
```

The final size-one dimension is retained so that the shared convolution is a
`Conv3d` with an antenna kernel and stride of one.

## 3. Antenna-Independent Embedding

The first model uses:

```text
Conv3d(
  in_channels=2,
  out_channels=D,
  kernel_size=(pt, pf, 1),
  stride=(pt, pf, 1),
)
```

The default WiFo patch sizes are `pt=4` and `pf=4`. Convolution weights are
shared across all antennas. One token therefore represents:

```text
4 time positions x 4 frequency positions x 1 antenna
```

Before the Transformer, the convolution output is restored to the original UPA
coordinate system:

```text
Y in R^(B x T' x K' x Nh x Nv x D)
T' = T / pt
K' = K / pf
```

`T` and `K` must be divisible by their patch sizes. If padding is introduced
later, it should be explicit and recorded in the sample metadata.

## 4. Token Order

Tokens are flattened in this order:

```text
(t, k, r, c)
```

`c` is the fastest-changing coordinate. The token index is:

```text
l = ((t*K' + k)*Nh + r)*Nv + c
```

The antenna coordinate is exactly:

```text
n = r*Nv + c
```

The implementation must preserve this order in patching, mask generation,
positional encoding, decoder restoration, and unpatching.

## 5. Four-Dimensional SinCos Positional Encoding

The first version uses fixed SinCos encodings, not learnable positional
embeddings. The feature dimension is split equally among time, frequency,
antenna row, and antenna column:

```text
D_t = D_f = D_r = D_c = D / 4
```

`D/4` must be even because SinCos uses paired sine and cosine features. The
current WiFo widths satisfy this requirement:

| Model width `D` | Per-coordinate dimensions |
|---:|---:|
| 64 | 16 |
| 128 | 32 |
| 256 | 64 |
| 512 | 128 |
| 768 | 192 |

For each token:

```text
P[t,k,r,c] = concat(
  PE_t(t),
  PE_k(k),
  PE_r(r),
  PE_c(c),
)
```

and the Transformer input is:

```text
Z[t,k,r,c] = Y[t,k,r,c] + P[t,k,r,c]
```

The convolution output itself is not semantically partitioned by coordinate.
Only the positional encoding has explicit feature subspaces.

## 6. UPA Relative Position Bias

Each attention head has independent row and column bias tables. For token
`i` attending to token `j`:

```text
delta_r = r_i - r_j
delta_c = c_i - c_j
B_ij^h = b_r^h[delta_r] + b_c^h[delta_c]
```

Attention becomes:

```text
S_ij^h = Q_i^h (K_j^h)^T / sqrt(d_h) + B_ij^h
```

The bias is applied in both the encoder and decoder. This is required because
masked antennas are absent from the encoder and are reconstructed only by the
decoder.

For the public D1-D18 datasets, the largest UPA is `4 x 8`, so the training
range is `delta_r in [-3,3]` and `delta_c in [-7,7]`.

The main zero-shot geometry experiments must stay within the relative-distance
range covered by the trained tables. For example, if `|delta_c| <= 7` during
training, the primary evaluation should not treat a larger UPA with
`|delta_c| > 7` as evidence of geometric extrapolation.

Delta clamping may still be implemented as an out-of-range guard so that code
does not crash, but it is an engineering fallback. It maps all larger distances
to the boundary and therefore cannot demonstrate that the model distinguishes
distances such as 8, 10, and 15.

If larger-UPA extrapolation becomes a research objective, replace the lookup
table with a continuous relative-distance function, such as:

```text
B_h(delta_r, delta_c) = MLP_h(delta_r, delta_c)
```

or use a relative-position bucket design. That decision should be evaluated as
a separate ablation rather than mixed into the first model.

## 7. Masking

The model keeps the three official WiFo tasks:

- `random`
- `temporal`
- `frequency`

It adds one aggregate `spatial` task. During training, one task is sampled per
batch. The spatial task internally samples one strategy:

- single antenna
- full row
- full column
- local rectangular sub-array block

All masks are represented over token coordinates:

```text
M in {0,1}^(T' x K' x Nh x Nv)
```

`M=1` denotes a masked token, and `M=0` denotes a visible token.

Spatial visibility is defined over UPA elements, not over time-frequency
tokens:

```text
R_visible_space = |visible antennas| / (Nh*Nv)
```

Spatial masks must enforce:

```text
R_visible_space >= 0.5
```

For a `4 x 8` UPA, at most 16 of 32 antennas may be masked. Masking one row
leaves a spatial visible ratio of `24/32 = 0.75`.

For `Nh=1`, row masking is unavailable and the spatial sampler selects among
antenna, column, and block. No strategy may mask every antenna. The first
single-antenna spatial ratio is 25%; the main comparison should also run 50%.

## 8. Encoder, Decoder, and Reconstruction

The encoder consumes only visible tokens:

```text
E in R^(B x L_vis x D_enc)
```

The decoder maps these to `D_dec`, inserts a learnable mask token at every
masked position, restores the original token order, adds the same form of 4D
positional encoding, and applies UPA relative bias in every attention layer.

The final linear layer predicts:

```text
2 * pt * pf
```

real values per token. With the default `4 x 4` time-frequency patch, this is
32 values. The output is unpatchified to:

```text
X_hat in R^(B x 2 x T x K x Nh x Nv)
```

and then converted to complex:

```text
H_hat = X_hat[:,0] + 1j*X_hat[:,1]
```

## 9. Loss

The first version uses only complex MSE on masked tokens:

```text
L_m = sum_m |H[m] - H_hat[m]|^2 / |Omega_m|
```

The training loop must expose two schedules:

```text
task_schedule="sample"
```

Each batch samples one of the four tasks uniformly. This is the default for
cheap structural-validation experiments because it avoids running four
reconstructions on every batch.

```text
task_schedule="sequential"
```

Every batch runs all four tasks, and the loss is:

```text
L = (L_random + L_temporal + L_frequency + L_spatial) / 4
```

The sequential mode extends the official WiFo training logic and should be used
for the final strict comparison. The sample mode is not assumed to be identical
to sequential; the experiment report must state which mode was used.

The spatial loss weight is initially `lambda_s = 1`. Do not add magnitude,
phase, spectral, or contrastive losses until the geometry-aware architecture is
validated.

## 10. Intentional First-Version Restrictions

- No warm start from official WiFo weights.
- No learnable positional embeddings.
- No combined row and column relative-bias interaction table.
- No token aggregation before the first experimental result.
- No additional loss terms.

These restrictions are intended to make the first experiment interpretable.
