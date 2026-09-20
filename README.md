# WiFo UPA Spatial Extension

This private research project evaluates and documents an extension of WiFo for
UPA-aware CSI prediction and partial-array reconstruction.

## Current Finding

The proposed model is feasible as a first research prototype:

- The official WiFo implementation embeds the flattened antenna dimension with a
  `Conv3d` kernel of `(t_patch_size, patch_size, patch_size)`, so both the
  frequency and antenna dimensions are convolved locally.
- Replacing that with a shared single-antenna time-frequency `Conv3d`,
  followed by a UPA-aware Transformer, is a coherent structural change.
- The main cost is sequence length. WiFo's antenna patching divides the number
  of antenna positions by four, while the proposed model keeps one token per
  antenna position. Attention matrix size therefore grows by approximately
  sixteen times for the same CSI size.
- The public WiFo repository mainly provides inference and evaluation code;
  its `TrainLoop.run_loop()` method invokes evaluation and does not implement
  the full pre-training loop described in the paper. A training entry point
  must be added before running the proposed experiments.

## Project Contents

- `docs/model-spec.md`: complete tensor, token, positional-encoding, bias,
  masking, decoder, and loss specification.
- `docs/source-audit.md`: comparison of the proposed design against the official
  WiFo paper and repository.
- `docs/experiment-plan.md`: ablations, training schedule, evaluation metrics,
  and compute budget.
- `docs/references.md`: canonical links for the paper, source commit, datasets,
  and pretrained weights.

## Scope

This repository currently contains documentation and an implementation plan. It
does not contain modified WiFo code, simulation datasets, TensorBoard logs, or
model checkpoints. The first implementation should be based on the official
repository at commit `a0889e124aeb9dcc423fc37e6796f77c7c4f7f24`.

The GitHub repository is intended to remain private until the research result
and publication plan are settled. No open-source license is added for now.
