"""UPA-aware masked autoencoder for complex CSI."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .attention import GeometryAwareBlock, UPARelativePositionBias
from .config import ModelConfig
from .embed import AntennaIndependentPatchEmbed
from .geometry import build_coords, patchify_tf, unpatchify_tf
from .masks import make_mask
from .pe import build_positional_encoding


class UPAMAE(nn.Module):
    """Antenna-independent TF embedding plus geometry-aware UPA Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed = AntennaIndependentPatchEmbed(
            embed_dim=config.embed_dim,
            pt=config.pt,
            pf=config.pf,
        )

        encoder_bias = (
            UPARelativePositionBias(
                num_heads=config.num_heads,
                max_delta_r=config.max_delta_r,
                max_delta_c=config.max_delta_c,
            )
            if config.use_upa_bias
            else None
        )
        decoder_bias = (
            UPARelativePositionBias(
                num_heads=config.decoder_num_heads,
                max_delta_r=config.max_delta_r,
                max_delta_c=config.max_delta_c,
            )
            if config.use_upa_bias
            else None
        )

        self.encoder = nn.ModuleList(
            GeometryAwareBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop=config.dropout,
                relative_bias=encoder_bias if index == 0 else None,
            )
            for index in range(config.depth)
        )
        # The official WiFo model has no relative bias. V1 adds it to every
        # encoder and decoder attention layer only when explicitly enabled.
        if config.use_upa_bias:
            for block in self.encoder[1:]:
                block.attn.relative_bias = encoder_bias

        self.norm = nn.LayerNorm(config.embed_dim)
        self.decoder_embed = nn.Linear(
            config.embed_dim, config.decoder_embed_dim
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder = nn.ModuleList(
            GeometryAwareBlock(
                dim=config.decoder_embed_dim,
                num_heads=config.decoder_num_heads,
                mlp_ratio=config.mlp_ratio,
                drop=config.dropout,
                relative_bias=decoder_bias if index == 0 else None,
            )
            for index in range(config.decoder_depth)
        )
        if config.use_upa_bias:
            for block in self.decoder[1:]:
                block.attn.relative_bias = decoder_bias
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        self.pred = nn.Linear(
            config.decoder_embed_dim, 2 * config.pt * config.pf
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _pe(self, coords: torch.Tensor, Nv: int) -> torch.Tensor:
        return build_positional_encoding(
            coords=coords,
            embed_dim=self.config.embed_dim,
            mode=self.config.pe_mode,
            Nv=Nv,
        )

    def _decoder_pe(self, coords: torch.Tensor, Nv: int) -> torch.Tensor:
        return build_positional_encoding(
            coords=coords,
            embed_dim=self.config.decoder_embed_dim,
            mode=self.config.pe_mode,
            Nv=Nv,
        )

    def forward(
        self,
        H: torch.Tensor,
        mask_type: str = "random",
        ratio: float = 0.85,
        spatial_type: str = "antenna",
    ) -> Dict[str, Any]:
        if H.ndim != 5:
            raise ValueError(
                f"expected complex H [B,T,K,Nh,Nv], got {tuple(H.shape)}"
            )
        if not torch.is_complex(H):
            raise ValueError("H must be a complex tensor")
        B, T, K, Nh, Nv = H.shape
        if T % self.config.pt or K % self.config.pf:
            raise ValueError(
                f"T={T} and K={K} must be divisible by "
                f"pt={self.config.pt}, pf={self.config.pf}"
            )
        if mask_type == "spatial" and not self.config.allow_spatial_mask:
            raise ValueError("this model variant does not allow spatial masks")

        Tp = T // self.config.pt
        Kp = K // self.config.pf
        L = Tp * Kp * Nh * Nv
        coords = build_coords(Tp, Kp, Nh, Nv).to(H.device)
        x = torch.stack((H.real, H.imag), dim=1).to(
            self.embed.proj.weight.dtype
        )
        tokens = self.embed(x)
        mask = make_mask(
            (B, Tp, Kp, Nh, Nv), mask_type, ratio, spatial_type
        ).to(H.device)
        if not mask.any():
            raise ValueError("mask must contain at least one masked token")

        tokens = tokens + self._pe(coords, Nv)
        visible_mask = ~mask
        visible_ids = visible_mask[0].nonzero(as_tuple=False).squeeze(1)
        masked_ids = mask[0].nonzero(as_tuple=False).squeeze(1)
        visible_tokens = tokens[:, visible_ids, :]
        visible_coords = coords[visible_ids]

        encoded = visible_tokens
        for block in self.encoder:
            encoded = block(encoded, visible_coords)
        encoded = self.norm(encoded)

        decoded_width = self.config.decoder_embed_dim
        decoded = torch.zeros(
            B, L, decoded_width, dtype=encoded.dtype, device=encoded.device
        )
        decoded[:, visible_ids, :] = self.decoder_embed(encoded)
        decoded[:, masked_ids, :] = self.mask_token.to(decoded.dtype)
        decoded = decoded + self._decoder_pe(coords, Nv)

        for block in self.decoder:
            decoded = block(decoded, coords)
        decoded = self.decoder_norm(decoded)
        prediction = self.pred(decoded)

        target = patchify_tf(x, self.config.pt, self.config.pf)
        loss = (
            (prediction[:, masked_ids, :] - target[:, masked_ids, :]) ** 2
        ).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite masked reconstruction loss")

        reconstructed = unpatchify_tf(
            prediction,
            T,
            K,
            Nh,
            Nv,
            self.config.pt,
            self.config.pf,
        )
        prediction_complex = reconstructed[:, 0] + 1j * reconstructed[:, 1]

        return {
            "prediction": prediction_complex,
            "loss": loss,
            "mask": mask,
            "visible_mask": visible_mask,
            "coords": coords,
            "num_tokens": L,
            "num_visible_tokens": int(visible_mask[0].sum().item()),
        }
