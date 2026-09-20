# Tensor Contract

This document freezes the runtime contracts that tests and experiment code may
rely on. Geometry functions perform reversible arrangement only; learnable
projections live in `embed.py`, `model.py`, and `baseline.py`.

## Complex CSI

The public model input is:

```text
H: complex tensor [B,T,K,Nh,Nv]
```

The model converts it internally to:

```text
X: real tensor [B,2,T,K,Nh,Nv]
X[:,0] = Re(H)
X[:,1] = Im(H)
```

MATLAB v7.3 files store the public WiFo data as:

```text
complex structured array [K,N,T,B]
```

The loader converts it to:

```text
H: [B,T,K,Nh,Nv]
```

with `N=Nh*Nv`.

## Token Geometry

Patch counts are:

```text
Tp = T / pt
Kp = K / pf
L  = Tp*Kp*Nh*Nv
```

V1 freezes `pt=4` and `pf=4`. The token order is `(t,k,r,c)`, with `c` the
fastest-changing axis:

```text
l = ((t*Kp+k)*Nh+r)*Nv+c
```

The inverse mapping must recover exactly:

```text
inverse_token_index(l) = (t,k,r,c)
```

## Structured Patching

`patchify_tf()` has the frozen output shape:

```text
[B,Tp,Kp,Nh,Nv,2,pt,pf]
```

The last three axes are ordered as `[channel,time,frequency]`. The inverse is:

```text
unpatchify_tf(patchify_tf(X)) == X
```

Model internals may flatten token axes only through:

```text
flatten_token_patches:
  [B,Tp,Kp,Nh,Nv,2,pt,pf] -> [B,L,2*pt*pf]

unflatten_token_patches:
  [B,L,2*pt*pf] -> [B,Tp,Kp,Nh,Nv,2,pt,pf]
```

These two helpers are exact inverses for known `(T,K,Nh,Nv,pt,pf)`.

## Antenna-Independent Embedding

The shared TF embedding uses:

```text
X: [B,2,T,K,Nh,Nv]
  -> [B*Nh*Nv,2,T,K,1]
  -> Conv3d(kernel=(pt,pf,1), stride=(pt,pf,1))
  -> [B*Nh*Nv,D,Tp,Kp,1]
  -> [B,L,D]
```

No convolution crosses antennas. The convolution weights are shared by all
antennas.

## Mask Layout

All generators return:

```text
mask: BoolTensor [B,L]
False = visible
True  = masked
```

V1 masks are batch-shared. `MaskLayout` derives and validates:

```text
shape
coords
mask
visible_mask
visible_ids
masked_ids
num_tokens
num_visible_tokens
```

The encoder consumes only `visible_ids`; the decoder restores the complete
token sequence using `coords`.

Batch sharing is an intentional V1 simplification, not an oversight. It gives
every sample in a batch the same visible coordinate set, which keeps the
encoder layout dense and avoids padded or packed variable-length sequences.
Per-sample masks would require explicit packed layout metadata and per-sample
attention masks; that is deferred to a later version.

## Positional Encoding

The 4D PE uses:

```text
D_t = D_f = D_r = D_c = D/4
```

The compatible control PE uses:

```text
D_t = D_k = floor(D/3)
D_s = D - 2*floor(D/3)
s   = r*Nv+c
```

For odd widths, the 1D encoder emits `floor(d/2)` sine features and
`ceil(d/2)` cosine features. For example, `D=256` uses widths:

```text
[85,85,86]
```

## Experiment Metrics

Every training and evaluation report records:

- masked NMSE
- full-tensor NMSE
- trainable parameter count
- approximate forward FLOPs
- training or inference wall time
- peak CUDA allocated memory when CUDA is available

The FLOPs estimate doubles multiply-accumulate operations and excludes I/O,
Python overhead, loss reduction, and checkpointing. CPU peak memory is reported
as zero rather than using a platform-dependent process RSS estimate.

The UPA estimator uses `L=Tp*Kp*Nh*Nv`. The flattened-antenna WiFo baseline
uses its own estimator and token count `L=Tp*Kp*(Nh*Nv/4)`. In sequential
training, FLOPs are accumulated separately for each task using that task's
visible-token ratio.
