# References and External Artifacts

## Paper

- WiFo: Wireless Foundation Model for Channel Prediction
  - arXiv: `https://arxiv.org/abs/2412.08908`
  - DOI: `https://doi.org/10.1007/s11432-025-4349-0`

## Source Code

- Official GitHub repository:
  `https://github.com/liuboxun/WiFo`
- Audited commit:
  `a0889e124aeb9dcc423fc37e6796f77c7c4f7f24`
- Commit message:
  `Update README.md`
- Commit date:
  `2025-11-28`

## Pretrained Weights

- Hugging Face model repository:
  `https://huggingface.co/liuboxun/WiFo`
- Files include:
  - `wifo_base.pkl`
  - `wifo_large.pkl`
  - `wifo_little.pkl`
  - `wifo_small.pkl`
  - `wifo_tiny.pkl`

Weights are referenced for provenance only. The first ablation does not load
them.

## Public Test Datasets

- Hugging Face dataset:
  `https://huggingface.co/datasets/pku-pcni-lab/RF-only_channel_dataset_for_WiFo`
- Public files include `dataset/D1/X_test.mat` through
  `dataset/D18/X_test.mat`.
- The paper defines D1-D19. The current public Hugging Face listing shown above
  includes D1-D18 only, so D19 must be covered by the synthetic shape harness
  until its data file is separately obtained.

The official WiFo repository states that all pre-training datasets were
released. The current official link is:

- `https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E`

As of 2026-09-20, this PKU cloud link returns "This address has expired".
Access issue:
`https://github.com/PKU-PCNI/WiFo/issues/10`

The public repository does not include full training splits. The WiFo-2
dataset contains directories named D17-D19, but its README maps them to QC17,
QC18, and QC19 from the WiFo-2 paper; they are not the original WiFo D17-D19
datasets and use `.pt` rather than the original `.mat` format.

## Related Concepts

- Masked autoencoder pretraining:
  He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
- VideoMAE:
  Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for
  Self-Supervised Video Pre-Training", NeurIPS 2022.
- Relative-position attention:
  Shaw et al., "Self-Attention with Relative Position Representations", NAACL
  2018.
