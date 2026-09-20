"""End-to-end package smoke test."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import SyntheticCSIDataset
from .model import UPAMAE
from .train import Trainer


def main() -> None:
    config = ModelConfig(
        embed_dim=32,
        decoder_embed_dim=32,
        depth=1,
        decoder_depth=1,
        num_heads=4,
        decoder_num_heads=4,
        pt=4,
        pf=4,
        pe_mode="4d",
        use_upa_bias=True,
        allow_spatial_mask=True,
    )
    model = UPAMAE(config)
    dataset = SyntheticCSIDataset("D5", num_samples=2, seed=7)
    loader = DataLoader(dataset, batch_size=2)
    trainer = Trainer(model, task_schedule="sample", lr=1e-3)
    metrics = trainer.train_epoch(loader, epoch=0)
    print(f"smoke test passed: loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
