# PilotWiMAE UPA Spatial-Geometry Audit

## Audit Scope

This is a source-level audit of:

- **Paper:** *PilotWiMAE: Pilot-Native Representation Learning for Wireless Channels*, arXiv `2605.22856v1`
- **PilotWiMAE repository:** `BerkIGuler/PilotWiMAE`
- **PilotWiMAE commit:** `0829fc2cf5d845a01df0df49da929142b55048b2`
- **CSIGen repository:** `BerkIGuler/CSIGen`
- **CSIGen commit:** `527fd18621fc1afdac2db56d792b0a91e7afdfc2`

The audit compares PilotWiMAE with the target model:

```text
H ∈ C^[B,T,K,Nh,Nv]
```

where:

```text
T  = time
K  = frequency/subcarrier
Nh = UPA row/horizontal dimension
Nv = UPA column/vertical dimension
```

The target model uses antenna-independent time-frequency patch embedding, token coordinates `(t,k,r,c)`, 4D positional encoding over `(t,k,r,c)`, UPA relative bias

```text
B_h(Δr,Δc) = b_r^h(Δr) + b_c^h(Δc)
```

and masks:

```text
random
temporal
frequency
antenna
row
column
2D block
```

The central question is whether PilotWiMAE already covers explicit 2D UPA row-column topology.

## Executive Conclusion

**PilotWiMAE does not make the explicit 2D UPA geometry thesis obsolete.**

PilotWiMAE is built on physical UPA data, but it linearizes the antenna array before model input:

```text
(Nh,Nv) -> S = Nh*Nv
```

Its model geometry is:

```text
H ∈ C^(T × S × F)
```

not:

```text
H ∈ C^(T × F × Nh × Nv)
```

Its "space" axis is a flattened antenna axis, and patching fuses four antenna elements into each spatial patch. There is no model-level row axis, no column axis, no `(t,k,r,c)` token coordinate, no `Δrow`, no `Δcolumn`, and no UPA relative-position bias.

Therefore:

```text
Final assessment: partial overlap.
The impact is limited to time × flattened space-frequency modeling,
pilot-native input, high-ratio masking, and channel representation learning.
It does not cover explicit 2D UPA row-column geometry.
```

## Source-Level Findings

### 1. Array and Raw Tensor Shape

#### Physical array

The paper and CSIGen configuration confirm that the TX array is a physical UPA:

```text
TX array: 4 × 8
RX array: 1 × 1
Spacing: 0.5λ
```

Evidence:

- Paper, Sionna-based dataset generation parameters, `main.tex:270-298`.
- CSIGen `config/pilotwimae_dataset_configs/pretrain/boston_config.yaml`:

  ```yaml
  tx_num_rows: 4
  tx_num_cols: 8
  rx_num_rows: 1
  rx_num_cols: 1
  tx_vertical_spacing: 0.5
  tx_horizontal_spacing: 0.5
  ```

- CSIGen `src/base_station.py::set_tx_antenna_array` constructs a Sionna `PlanarArray`.
- CSIGen `src/user_equipment.py::set_rx_antenna_array` constructs the RX `PlanarArray`.

The paper's beam-selection section defines a UPA with:

```text
N_h = horizontal/column count
N_v = vertical/row count
(N_h,N_v) = (8,4)
```

Evidence: `main.tex:355-394`.

#### Raw CFR tensor

CSIGen's CFR output has the following shape:

```text
[num_selected_users, 1, num_rx_ant, num_tx_ant, num_subcarriers, num_ofdm_symbols]
```

For PilotWiMAE's configuration:

```text
num_rx_ant = 1
num_tx_ant = 4*8 = 32
num_subcarriers = 32
num_ofdm_symbols = 14
```

Evidence:

- CSIGen `src/channel.py::compute_cfr_for_paths`.
- CSIGen `src/channel.py::save_channel_data`.
- Paper, `main.tex:157`.

Thus, the raw saved tensor has a single flattened TX-antenna axis. The `4 × 8` geometry exists in the generation config and metadata, but not as two tensor axes.

#### PilotWiMAE dataloader

PilotWiMAE loads:

```text
[batch, 1, 1, N, T, K]
```

and converts it to:

```text
[batch, T, N, K]
```

The code:

```python
h_data[batch_start:batch_end, 0, 0, :, :, :].transpose(0, 2, 1, 3)
```

Evidence:

- `pilotwimae/data/dataset.py::OptimizedPreloadedDataset`.
- Paper, `main.tex:157`.

There is no `(4,8)` reshape and no model-level reconstruction of `(Nh,Nv)`.

### 2. Patching and Tokenization

PilotWiMAE's model input is:

```text
(B,T,S,F)
```

where:

```text
T = 14
S = Nh*Nv = 32
F = 32
```

Official configuration:

```yaml
input_shape: [14, 32, 32]
patch_size: [1, 4, 4]
```

Therefore:

```text
pt = 1
ps = 4
pf = 4

nt = T/pt = 14
ns = S/ps = 8
nf = F/pf = 8

P = nt*ns*nf = 14*8*8 = 896
Dp = 2*pt*ps*pf = 32
```

Each token contains:

```text
1 time position × 4 antenna elements × 4 frequency positions
```

Evidence:

- `pilotwimae/models/modules/patching.py::Patcher3D`.
- `pilotwimae/models/encoder_backbone.py::tokenizer_geometry`.
- `configs/train/fst/tk2_sm09_dec2_snr_scale.yaml`.

#### Consequence

PilotWiMAE is **not antenna-independent** at patch embedding.

It fuses four flattened antenna elements into each token before attention. Its `ns` axis is a flattened antenna-patch axis, not a UPA row axis and not a UPA column axis.

The token coordinate is:

```text
(t_patch, s_patch, f_patch)
```

not:

```text
(t,k,r,c)
```

### 3. Positional Encoding

PilotWiMAE uses `SinusoidalConcat3D`.

It builds separate 1D sinusoidal tables for:

```text
nt
ns
nf
```

and concatenates them into one model-dimension vector.

The resulting PE is:

```text
PE(t_patch, spatial_patch, frequency_patch)
```

Evidence:

- `pilotwimae/models/modules/pos_encodings.py::SinusoidalConcat3D`.
- Paper, `main.tex:161-166`.

There is no `PE(t,k,r,c)`, no row coordinate, no column coordinate, no RoPE, no relative positional encoding, and no antenna-distance embedding. The only learned PE parameter is a scalar scale applied to the fixed sinusoidal PE.

### 4. Factorized Attention

PilotWiMAE's factorized encoder reshapes the visible token sequence as:

```text
(B, Tk*Sk, D)
-> (B, Tk, Sk, D)
```

where:

```text
Tk = retained temporal patch count
Sk = retained flattened space-frequency patch count
```

Temporal attention:

```text
(B,Tk,Sk,D)
-> permute/reshape
-> (B*Sk,Tk,D)
```

Space-frequency attention:

```text
(B,Tk,Sk,D)
-> reshape
-> (B*Tk,Sk,D)
```

Evidence:

- `pilotwimae/models/modules/encoder.py::FactorizedEncoder.forward`.
- Paper, `main.tex:177-182`.

Here:

```text
Sk = ns*nf
```

Therefore PilotWiMAE's "space-frequency attention" is:

```text
flattened antenna-patch × frequency-patch
```

not:

```text
row × column × frequency
```

There is no axis separation for UPA rows and columns.

### 5. Decoder

The decoder is a joint transformer over the full flattened token sequence.

It:

1. Inserts learnable mask tokens at masked positions.
2. Places encoded visible tokens at their flat indices.
3. Adds decoder-side `SinusoidalConcat3D`.
4. Runs a standard `nn.TransformerEncoder`.
5. Projects tokens back to patch dimension.

Evidence:

- `pilotwimae/models/modules/decoder.py::Decoder`.

The decoder also has no row/column attention, no 2D UPA reshape, and no UPA relative bias.

### 6. Relative Geometry

Searches across the paper, official model code, configs, training scripts, evaluation scripts, and tests found no:

```text
Δrow
Δcolumn
Δantenna
B(Δr,Δc)
relative-position bias
row distance table
column distance table
antenna-distance embedding
```

The only explicit horizontal/vertical geometry appears in downstream beam-label generation and DFT codebook construction.

Relevant files:

- `pilotwimae/data/beam/beam_codebook.py`
- `pilotwimae/data/beam/beam_labels.py`
- `scripts/beam_prediction/evaluate_knn_beam.sh`

This is downstream supervision/evaluation geometry, not encoder geometry.

Attention may learn implicit spatial relationships from the flattened antenna ordering, but that is not explicit UPA relative geometry.

### 7. Masking

The official implementation supports:

```text
random
temporal
tube
factorized structured random
```

Evidence:

- `pilotwimae/models/modules/masking.py::MaskGenerator`.
- `pilotwimae/models/modules/masking.py::FactorizedMaskGenerator`.

The full PilotWiMAE model uses the factorized mask:

```text
keep Tk temporal patch indices
keep Sk flattened space-frequency positions
share the same Sk positions across retained temporal indices
```

The visible fraction is:

```text
(Tk/nt) * (Sk/(ns*nf))
```

#### Phase 1

Official config:

```yaml
num_time_keep: 2
spatial_mask_ratio: 0.9
```

With:

```text
nt = 14
ns*nf = 64
```

the code keeps:

```text
Sk = round(64 * (1 - 0.9)) = 6
visible tokens = 2 * 6 = 12
total tokens = 896
mask ratio = 884/896 = 98.66%
```

The paper calls this approximately 99%.

#### Phase 2 discrepancy

The paper states:

```text
n_k = 4
rho_k = 0.75
approximately 79% overall mask
```

In the paper, `rho_k` is defined as the kept fraction of spectro-spatial positions. Under that definition:

```text
visible fraction = (4/14) * 0.75
mask ratio = 1 - (4/14)*0.75
           = 78.57%
```

However, the official released config uses:

```yaml
num_time_keep: 4
spatial_mask_ratio: 0.75
```

The code interprets `spatial_mask_ratio` as a mask ratio:

```python
num_spatial_keep = round(ns_nf * (1 - spatial_mask_ratio))
```

Therefore it keeps only 25%:

```text
Sk = 16
visible fraction = (4/14)*0.25
mask ratio = 92.86%
```

This is a paper/config semantic inconsistency, not a rounding issue.

#### Missing mask types

PilotWiMAE does not implement standalone:

```text
frequency mask
antenna mask
row mask
column mask
2D UPA block mask
```

Its high mask ratio is not a 2D UPA structured mask. It is a temporal/flattened-space-frequency structured random mask.

## Pilot-Native Design

PilotWiMAE is pilot-native because its encoder directly consumes sparse, noisy pilot observations instead of requiring a full clean CSI tensor.

The fixed pilot pattern is:

```text
time symbols: {2,11}
subcarriers: {0,1,2,3,
              8,9,10,11,
              16,17,18,19,
              24,25,26,27}
```

Thus:

```text
pilot REs = 2*16 = 32
total time-frequency REs = 14*32 = 448
```

At each pilot resource element, all antenna entries are observed.

Evidence:

- Paper, `main.tex:134-147`.
- Paper, `main.tex:298`.
- `pilotwimae/downstream/beam_prediction/pilot_pattern.py`.

The encoder receives noisy visible patches under structured masking and AWGN. Reconstruction targets come from the clean full-grid channel.

PilotWiMAE's main technical identity is therefore:

1. Pilot-native input interface.
2. Noise curriculum.
3. High-ratio structured masking.
4. Patch-normalized reconstruction loss.
5. Auxiliary scale loss.
6. Decoder-centric second pretraining stage.

Its main novelty is not explicit UPA geometry.

## Dataset, Training, and Scale

### Dataset

Ray-tracing data generated with CSIGen/Sionna.

Training cities:

```text
Boston
New York City
San Francisco
Chicago
```

OOD test city:

```text
Los Angeles
```

Sample counts:

| City | Train | Test |
|---|---:|---:|
| Boston | 152,675 | 18,423 |
| New York City | 92,153 | 8,130 |
| San Francisco | 94,262 | 17,755 |
| Chicago | 99,344 | 10,158 |
| Los Angeles | n/a | 17,466 |
| **Total** | **438,434** | **71,932** |

Validation split:

```text
10% of training
```

Evidence:

- Paper, Appendix *Dataset details*, `main.tex:546-570`.

### Model and training

```text
Input shape: (14,32,32)
Patch shape: (1,4,4)
Encoder: FST
Encoder dimension: 128
Encoder blocks: 3
Attention heads: 8
Decoder: JST
Phase 1 decoder: 2 layers, 4 heads
Phase 2 decoder: 8 heads
Trainable parameters reported: 1,594,658
FLOPs/epoch reported: 2.017 P
Time/epoch reported: 208.01 s
GPU: single NVIDIA RTX A4000
```

Phase 1:

```text
epochs: 500
batch size: 512
optimizer: AdamW
learning rate: 5e-4
weight decay: 0.005
scheduler: cosine
precision: mixed precision
```

Phase 2:

```text
paper epochs: 200
released decoder-only run epochs: 300
learning rate: 1e-4
encoder frozen
```

Evidence:

- Paper, `main.tex:300-333`.
- Paper, `main.tex:494-534`.
- `configs/train/fst/tk2_sm09_dec2_snr_scale.yaml`.
- `configs/train/decoder_only/dec_only_fst_tk2_sm09_tk4_sm075_dec12_normpatch_mse.yaml`.
- `runs/decoder_ablations/pilotwimae_dec_only_fst_tk4_sm075_dec12_from_tk2_sm09_fst_scaleaux_noiserobust_snr40/config.yaml`.

The paper/released-run epoch discrepancy is unresolved.

## Performance Summary

The paper reports performance through figures rather than numeric result tables.

### Tasks

1. Beam selection
   - Metric: top-3 accuracy.
   - ID and OOD.
   - 3.5 GHz to 28 GHz transfer.

2. Channel characterization
   - Metric: LoS/NLoS classification accuracy.
   - ID and OOD.
   - 3.5 GHz to 28 GHz transfer.

3. Dense channel estimation
   - Metric: NMSE.
   - Los Angeles OOD at 3.5 GHz.

### Qualitative reported findings

- The FST family is more robust than the JST baseline, especially at low SNR.
- Noise-robust pretraining is the dominant low-SNR enabler.
- Auxiliary scale loss especially helps LoS/NLoS classification.
- Pilot-only inference remains competitive with full-channel inference.
- Increasing decoder depth improves channel-estimation NMSE.
- Decoder depth 12 closes the high-SNR gap in the reported decoder-depth sweep.

Evidence:

- Paper, `main.tex:423-492`.

No numeric task values are invented here.

## Feature Matrix

| Feature | PilotWiMAE | Target UPA model |
|---|---|---|
| Explicit time axis | Yes | Yes |
| Explicit frequency axis | Yes | Yes |
| Explicit antenna axis | Yes, flattened `S` | Yes |
| Explicit `Nh` | Data/config only; not a model axis | Yes |
| Explicit `Nv` | Data/config only; not a model axis | Yes |
| Antenna-independent embedding | No | Yes |
| Patch fuses multiple antennas | Yes, `ps=4` | No |
| Token coordinate contains `t` | Yes | Yes |
| Token coordinate contains `k` | Yes | Yes |
| Token coordinate contains antenna | Flattened spatial-patch index | Yes |
| Token coordinate contains row | No | Yes |
| Token coordinate contains column | No | Yes |
| Temporal attention | Yes | Yes |
| Spatial attention | Flattened space-frequency attention | UPA-aware attention |
| Factorized attention | Yes | No |
| `Δantenna` | No | No |
| `Δrow` | No | Yes |
| `Δcolumn` | No | Yes |
| UPA relative bias | No | Yes |
| Random mask | Yes | Yes |
| Temporal mask | Yes | Yes |
| Frequency mask | No standalone frequency mask | Yes |
| Antenna mask | No standalone antenna mask | Yes |
| Row mask | No | Yes |
| Column mask | No | Yes |
| 2D block mask | No | Yes |
| UPA shape generalization | Unknown/not shown | Planned |

## Seven-Axis Comparison

### 1. Tensor axes

PilotWiMAE:

```text
(B,T,S,F)
```

Target:

```text
(B,T,K,Nh,Nv)
```

PilotWiMAE knows the physical array was `4×8` only through generation metadata. Inside the model, `4×8` and any other arrangement with `S=32` are structurally equivalent.

### 2. Patch

PilotWiMAE:

```text
(pt,ps,pf) = (1,4,4)
```

A token mixes four flattened antenna elements.

Target:

```text
(pt,pf) = (4,4)
```

Each antenna is patched independently over time-frequency, then antenna coordinates are retained.

### 3. Token

PilotWiMAE:

```text
(t_patch, s_patch, f_patch)
```

Target:

```text
(t_patch, k_patch, r, c)
```

### 4. Attention

PilotWiMAE:

```text
temporal attention
+
flattened space-frequency attention
```

Target:

```text
UPA-aware attention with explicit row/column geometry
```

### 5. Positional encoding

PilotWiMAE:

```text
PE(t_patch, s_patch, f_patch)
```

Target:

```text
PE(t_patch, k_patch, row, column)
```

### 6. Relative geometry

PilotWiMAE:

```text
None explicit
```

Target:

```text
B_h(Δr,Δc) = b_r^h(Δr) + b_c^h(Δc)
```

### 7. Masking

PilotWiMAE:

```text
temporal × flattened space-frequency structured random mask
```

Target:

```text
random
temporal
frequency
antenna
row
column
2D block
```

## Overlap Assessment

### Covered by PilotWiMAE

- Explicit time axis.
- Explicit frequency axis.
- Flattened antenna axis.
- Temporal attention.
- Joint space-frequency attention over a flattened spatial index.
- Pilot-native input.
- High-ratio structured masking.
- Channel foundation representation learning.

### Partially Impacted

PilotWiMAE demonstrates that temporal/flattened-space-frequency factorization and pilot-native pretraining can work on physical UPA data. Therefore, the target project must show measurable benefit from preserving explicit row-column geometry over this flattened baseline.

### Not Covered

- Antenna-independent time-frequency embedding.
- One token per antenna or antenna-patch without cross-antenna fusion at embedding.
- `(t,k,r,c)` token coordinates.
- `Δrow`.
- `Δcolumn`.
- UPA relative bias.
- Row masks.
- Column masks.
- 2D UPA block masks.
- UPA shape generalization.

## Fact Grades

### Confirmed Facts

1. The physical TX array is a `4×8` UPA.
2. The RX array is `1×1`.
3. CSIGen constructs Sionna `PlanarArray` objects.
4. The saved CFR tensor has a single TX-antenna axis.
5. PilotWiMAE uses `(B,T,S,F)`, with `S=Nh*Nv`.
6. PilotWiMAE does not preserve `(Nh,Nv)` as model axes.
7. Patch size is `(1,4,4)`.
8. Each token contains four flattened antenna elements.
9. Token coordinates are `(t_patch,s_patch,f_patch)`.
10. PE is axial 3D sinusoidal over `(nt,ns,nf)`.
11. Factorized attention is temporal plus flattened space-frequency.
12. There is no row axis or column axis in encoder or decoder.
13. There is no explicit UPA relative-position bias.
14. Full-model masking is temporal/flattened-space-frequency structured masking.
15. There are no row/column/2D-UPA block masks.
16. Pilot-native inference directly uses sparse pilot observations.
17. The fixed pilot pattern uses time `{2,11}` and 16 subcarrier positions.

### Reasonable Inference

The learned representation may indirectly encode horizontal/vertical directional structure because:

1. The data are generated from a physical UPA.
2. The flattened antenna ordering is deterministic.
3. Downstream beam codebooks and labels are UPA-aware.
4. The paper observes different behavior for horizontal/vertical codebook tilings.

However, this is implicit learned structure. It does not make PilotWiMAE an explicit UPA row-column geometry model.

### Unconfirmed or Inconsistent

1. Phase 2 mask ratio is inconsistent between the paper and released config.
2. Paper says Phase 2 uses 200 epochs; released decoder-only run config says 300 epochs.
3. UPA shape generalization is not shown.
4. The exact flatten order from Sionna's antenna axis is not used or declared in PilotWiMAE's model geometry.
5. The downloaded binary checkpoint was incomplete and could not be safely parsed in the audit environment.
6. No numeric performance table is available; the paper reports results through figures.

## Final Judgment

### Has PilotWiMAE already done the UPA thesis?

**No. It creates partial impact, not a replacement.**

PilotWiMAE proves the value of:

```text
pilot-native input
temporal × flattened space-frequency factorization
high-ratio masking
channel representation learning
```

It does not prove:

```text
explicit row-column UPA geometry is unnecessary
```

because it never tests that geometry.

### Is it a bigger threat than Adaptive 3D-RoPE?

For the broad Channel Foundation Model framing, **yes**: PilotWiMAE is more complete because it has pilot-native input, large-scale pretraining, official code, factorized attention, and downstream results.

For the specific explicit UPA row-column geometry thesis, **no**: PilotWiMAE discards `(Nh,Nv)` before the model and therefore does not directly cover:

```text
(t,k,r,c)
Δrow
Δcolumn
B_h(Δr,Δc)
row/column/block masking
UPA shape generalization
```

Adaptive 3D-RoPE may be more directly relevant to explicit spatial coordinate encoding, while PilotWiMAE is the stronger representation-learning and pilot-native baseline.
