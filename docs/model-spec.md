# Model Specification

## 1. Objective

The model separates time-frequency feature extraction from inter-antenna
spatial modeling:

```text
CSI
  -> antenna-independent time-frequency embedding
  -> 4D positional encoding
  -> UPA relative position bias
  -> MAE Transformer encoder
  -> decoder
  -> CSI reconstruction
```

The central hypothesis is that a UPA should not be treated as an arbitrary
one-dimensional antenna list. The antenna has a two-dimensional physical
coordinate, and spatial prediction should exploit that geometry.

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

| Model width `D` | Per-coordinate dimensions |
|---:|---:|
| 512 | 128 |
| 256 | 64 |
| 128 | 32 |
| 64 | 16 |
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
range is `delta_r in [-3,3]` and `delta_c in [-7,7]`. For an unseen larger
UPA, clamp the delta before lookup. The clamped behavior is deterministic but
not a guarantee of extrapolation; unseen-geometry tests must report it
explicitly.

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

Spatial masks must enforce:

```text
number of visible tokens >= ceil(0.5 * T' * K' * Nh * Nv)
```

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

Training samples one of four tasks per batch and averages gradients over the
task distribution. An equivalent deterministic implementation is:

```text
L = (L_random + L_temporal + L_frequency + L_spatial) / 4
```

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
