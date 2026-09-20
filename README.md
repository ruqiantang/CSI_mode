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
- The central narrative is **antenna-independent time-frequency embedding plus
  a geometry-aware UPA Transformer**. The change is not merely a new positional
  encoding: spatial modeling moves from the convolutional embedding to the
  Transformer.
- The main cost is sequence length. WiFo's antenna patching divides the number
  of antenna positions by four, while the proposed model keeps one token per
  antenna position. Attention matrix size therefore grows by approximately
  sixteen times for the same CSI size.
- The public WiFo repository mainly provides inference and evaluation code;
  its `TrainLoop.run_loop()` method invokes evaluation and does not implement
  the full pre-training loop described in the paper. A training entry point
  must be added before running the proposed experiments.
- Relative-position lookup clamping is an engineering fallback, not the claimed
  generalization mechanism. The main zero-shot experiments must stay within the
  relative-distance range covered by the trained bias tables.
- Spatial visibility is defined over antenna elements, not over time-frequency
  tokens. At least half of the UPA elements must remain visible.
- Training must support both `task_schedule="sample"` for efficient structural
  validation and `task_schedule="sequential"` for a strict WiFo-style
  comparison in which all four tasks are run on each batch.

## Project Contents

- `src/wifo_upa/`: V1 implementation of antenna-independent TF embedding,
  4D/control positional encodings, structured masks, UPA relative bias,
  training, evaluation, and the WiFo-like baseline.
- `configs/`: base, Small, Little, and frozen A/B/C/D/Full ablation variants.
- `tests/`: geometry, embedding, PE, mask, attention, model, training-loop,
  and D1-D19 dataset-shape contracts.
- `scripts/`: ablation and dataset download entry points.
- `docs/model-spec.md`: complete tensor, token, positional-encoding, bias,
  masking, decoder, and loss specification.
- `docs/source-audit.md`: comparison of the proposed design against the official
  WiFo paper and repository.
- `docs/experiment-plan.md`: ablations, training schedule, evaluation metrics,
  and compute budget.
- `docs/references.md`: canonical links for the paper, source commit, datasets,
  and pretrained weights.
- `docs/implementation-plan-v1.md`: frozen v1 repository layout, APIs,
  ablation variants, development order, and acceptance criteria.

## Quick Verification

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python -m wifo_upa.smoke_test
```

## Scope

This repository contains documentation and an independent V1 reference
implementation. It does not copy modified WiFo source code and does not contain
simulation datasets, TensorBoard logs, or model checkpoints. The official source
audit is based on repository commit
`a0889e124aeb9dcc423fc37e6796f77c7c4f7f24`.

The GitHub repository is intended to remain private until the research result
and publication plan are settled. No open-source license is added for now.
