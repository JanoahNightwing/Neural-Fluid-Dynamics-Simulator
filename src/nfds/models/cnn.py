"""Residual CNN: the simplest learned surrogate.

omega_{t+1} = omega_t + f(omega_{t-n_in+1..t}, nu). Local 3x3 periodic
convolutions give exact translation equivariance; the receptive field grows by
2 pixels per convolution, which limits how well it can represent the global
coupling created by the Poisson solve. That limitation is part of what the
comparison with the U-Net and FNO is meant to show.
"""

from __future__ import annotations

import torch
from torch import nn

from nfds.models.layers import periodic_conv, zero_init


class ResBlock(nn.Module):
    def __init__(self, width: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv1 = periodic_conv(width, width, kernel)
        self.conv2 = periodic_conv(width, width, kernel)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.conv2(self.act(self.conv1(self.act(h))))


class ResidualCNN(nn.Module):
    """Parameters: n_in input frames, width channels, depth residual blocks.

    The normalised log-viscosity is appended as a constant input channel
    when ``condition_on_nu`` is true.
    """

    def __init__(self, n_in: int = 1, width: int = 64, depth: int = 8, kernel: int = 3,
                 condition_on_nu: bool = True) -> None:
        super().__init__()
        self.n_in, self.condition_on_nu = n_in, condition_on_nu
        self.lift = periodic_conv(n_in + int(condition_on_nu), width, kernel)
        self.blocks = nn.Sequential(*[ResBlock(width, kernel) for _ in range(depth)])
        self.proj = zero_init(periodic_conv(width, 1, kernel))
        self.receptive_field = 1 + (kernel - 1) * (2 + 2 * depth)

    def forward(self, x: torch.Tensor, nu_feat: torch.Tensor) -> torch.Tensor:
        """x: (B, n_in, H, W); nu_feat: (B,). Returns (B, 1, H, W)."""
        h = x
        if self.condition_on_nu:
            h = torch.cat([x, nu_feat.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[-2:])], dim=1)
        h = self.blocks(self.lift(h))
        return x[:, -1:] + self.proj(nn.functional.gelu(h))
