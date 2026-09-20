# Official WiFo Source Audit

Audited source:

- Repository: `https://github.com/liuboxun/WiFo`
- Commit: `a0889e124aeb9dcc423fc37e6796f77c7c4f7f24`
- Commit date: 2025-11-28
- Paper: `WiFo: Wireless Foundation Model for Channel Prediction`,
  Science China Information Sciences, 2025.

The audit below compares the public implementation with the proposed UPA-aware
extension.

## 1. Data Shape and Channels

The public data loader reads complex MATLAB data, unsqueezes a batch dimension,
and concatenates real and imaginary tensors along channel dimension one. This is
compatible with the proposed `[B,2,T,K,Nh,Nv]` interface, provided the UPA
shape is supplied separately and the flattened antenna dimension is reshaped.

The public code does not itself carry `(Nh,Nv)` metadata. The extension must
pass the UPA shape through the dataset configuration and model forward call.

## 2. Convolutional Embedding

In `src/Embed.py`, `TokenEmbedding.__init__` constructs:

```python
kernel_size = [t_patch_size, patch_size, patch_size]
```

The public model therefore applies one `Conv3d` directly to
`[B,2,T,K,N]`. The frequency dimension and flattened antenna dimension are
both locally patched.

The proposal changes this to:

```python
kernel_size = (pt, pf, 1)
stride = (pt, pf, 1)
```

after moving antennas temporarily into the batch dimension. This removes
cross-antenna convolution from the embedding while preserving shared
time-frequency feature extraction.

## 3. Patch and Token Count

The public implementation patches each axis by four under the default
settings. Its token count is:

```text
L_WiFo = (T/4) * (K/4) * (N/4)
```

The proposal uses:

```text
L_UPA = (T/4) * (K/4) * N
```

Therefore:

```text
L_UPA = 4 * L_WiFo
```

Attention matrix area is proportional to `L^2`, so the theoretical attention
memory and computation grow by approximately sixteen times.

This is the largest feasibility risk. The first implementation should use
WiFo-Small or WiFo-Little, batch size 8 or 16, mixed precision, and gradient
accumulation when necessary.

## 4. Positional Encoding

The public repository supports `SinCos`, `SinCos_3D`, and `None` positional
encoding modes. The README inference commands use `SinCos_3D`. The
implementation generates three coordinate encodings for time, frequency, and
one spatial dimension, then concatenates feature dimensions. For `D=512`, it
uses approximately `170 + 170 + 172` feature dimensions.

The proposal replaces the one-dimensional antenna coordinate with row and
column coordinates and uses an exact equal split:

```text
D=512: 128 + 128 + 128 + 128
```

This is a controlled change. The equal split is preferred for the first
version because it avoids tuning coordinate-specific feature budgets before
the architecture is validated.

## 5. Attention Bias

The public `Attention` module computes:

```text
QK^T / sqrt(d_h)
```

and does not include relative-position bias.

The proposal adds a separable UPA bias to encoder and decoder attention. The
first implementation should keep the bias additive, per-head, and separable:

```text
B(delta_r, delta_c) = b_r(delta_r) + b_c(delta_c)
```

This gives a clean ablation and avoids the quadratic memory of an explicit
`delta_r x delta_c` table.

## 6. Mask Strategies

The public repository implements:

- `random_masking`
- `causal_masking` for temporal masking
- `fre_masking` for frequency masking

The paper uses random ratio 85%, temporal ratio 50%, and frequency ratio 50%.
The proposal preserves these three tasks and adds an aggregate spatial task.

The extension cannot reuse the public mask functions unchanged because their
reshape logic assumes token order `(T',K',N')` with `N'` the patched antenna
count. The new token order is `(T',K',Nh,Nv)` with no antenna patching.

## 7. Decoder and Restoration

The public decoder maps encoder width to decoder width, inserts mask tokens,
restores the original sequence, adds decoder positional encoding, and predicts
the original patch. This structure is reusable conceptually.

Restoration code must be rewritten for the new token order and for the spatial
mask, because spatial masks cannot be represented by the public temporal or
frequency restoration helpers.

## 8. Training Code

The public repository's `TrainLoop.run_loop()` currently calls
`Evaluation()` and returns. It does not iterate epochs, sample batches, compute
gradients, step the optimizer, or apply the paper's cosine schedule.

The paper reports AdamW, weight decay 0.05, batch size 128, 200 epochs, a
five-epoch warmup, and base learning rate `5e-4`. A new training loop must
implement those settings, plus the four-task sampling rule and checkpointing.

This is a material gap: the extension cannot be trained by simply adding
modules to the current inference-only entry point.

## 9. Evaluation

The public evaluation computes NMSE only over masked positions. This matches
the proposed reconstruction metric. The extension should keep that behavior and
add:

- token count
- encoder and decoder sequence lengths
- peak CUDA memory
- FLOPs where practical
- wall-clock inference time

## 10. Feasibility Verdict

The proposal is structurally feasible and well motivated. The following work
items are required before a credible result:

1. Pass `(Nh,Nv)` metadata through loading and model calls.
2. Implement antenna-independent convolution and exact token reshaping.
3. Implement 4D positional encoding and UPA relative bias.
4. Rewrite random, temporal, and frequency masks for the new token order.
5. Add spatial masks with a visible-token invariant.
6. Add a real training loop.
7. Run the staged ablation from small models upward.

No issue in the public implementation invalidates the proposed research
hypothesis. The primary risk is computational cost rather than mathematical
inconsistency.
