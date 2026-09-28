"""Shared model initialization helpers."""

from __future__ import annotations

import torch
from torch import nn


def _init_module_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def initialize_model_weights(model: nn.Module, mask_token: nn.Parameter) -> None:
    """Apply the common transformer initialization used by model variants."""
    model.apply(_init_module_weights)
    nn.init.normal_(mask_token, std=0.02)
