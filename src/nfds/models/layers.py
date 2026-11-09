"""Shared building blocks."""

from __future__ import annotations

import torch
from torch import nn


def periodic_conv(c_in: int, c_out: int, kernel: int = 3, stride: int = 1) -> nn.Conv2d:
    """Convolution with circular padding, so the network respects the periodic domain."""
    return nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2,
                     padding_mode="circular")


def zero_init(module: nn.Module) -> nn.Module:
    """Zero the weights so a residual model starts out as the identity (persistence)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class NuEmbedding(nn.Module):
    """Small MLP mapping the normalised log-viscosity to a conditioning vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, nu_feat: torch.Tensor) -> torch.Tensor:
        return self.net(nu_feat.view(-1, 1))


class FiLM(nn.Module):
    """Feature-wise affine modulation: h * (1 + scale(e)) + shift(e)."""

    def __init__(self, emb_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = zero_init(nn.Linear(emb_dim, 2 * channels))

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
