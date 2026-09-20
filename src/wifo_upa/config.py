"""Model and experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    embed_dim: int = 256
    decoder_embed_dim: int = 256
    depth: int = 6
    decoder_depth: int = 4
    num_heads: int = 8
    decoder_num_heads: int = 8
    mlp_ratio: float = 2.0
    pt: int = 4
    pf: int = 4
    pe_mode: str = "4d"  # "4d" or compatible-control "control"
    use_upa_bias: bool = True
    max_delta_r: int = 3
    max_delta_c: int = 7
    allow_spatial_mask: bool = True
    spatial_types: tuple[str, ...] = (
        "antenna",
        "row",
        "column",
        "block",
    )
    dropout: float = 0.0

    def validate(self) -> None:
        self.spatial_types = tuple(self.spatial_types)
        positive = [
            self.embed_dim,
            self.decoder_embed_dim,
            self.depth,
            self.decoder_depth,
            self.num_heads,
            self.decoder_num_heads,
            self.pt,
            self.pf,
            self.max_delta_r,
            self.max_delta_c,
        ]
        if any(value <= 0 for value in positive):
            raise ValueError("all positive model fields must be > 0")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.decoder_embed_dim % self.decoder_num_heads:
            raise ValueError("decoder_embed_dim must be divisible by decoder_num_heads")
        if self.pe_mode not in ("4d", "control"):
            raise ValueError("pe_mode must be '4d' or 'control'")
        if self.pt != 4 or self.pf != 4:
            # V1 intentionally freezes the paper-compatible TF patch size.
            raise ValueError("V1 requires pt=4 and pf=4")
        valid_spatial_types = {"antenna", "row", "column", "block"}
        spatial_types = tuple(self.spatial_types)
        if (
            not spatial_types
            or any(value not in valid_spatial_types for value in spatial_types)
            or len(set(spatial_types)) != len(spatial_types)
        ):
            raise ValueError(
                "spatial_types must be a nonempty subset of "
                "antenna, row, column, block without duplicates"
            )
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError("config YAML root must be a mapping")
        return cls.from_dict(values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
