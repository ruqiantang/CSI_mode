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
  includes D1-D18, so D19 must be covered by the synthetic shape harness until
  its data file is separately obtained.

The public repository does not include full training splits. The pre-training
dataset is linked from the official WiFo README at:

- `https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E`

## Related Concepts

- Masked autoencoder pretraining:
  He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
- VideoMAE:
  Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for
  Self-Supervised Video Pre-Training", NeurIPS 2022.
- Relative-position attention:
  Shaw et al., "Self-Attention with Relative Position Representations", NAACL
  2018.
