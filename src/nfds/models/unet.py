"""U-Net for PDE surrogates, following the modern design used in PDEArena.

Residual blocks with group normalisation, circular padding everywhere, strided
convolutions for downsampling, nearest-neighbour upsampling, and FiLM
conditioning on the log-viscosity at every block. The multi-scale structure
gives a receptive field covering the whole 64² domain in a few layers.
"""

from __future__ import annotations

import torch
from torch import nn

from nfds.models.layers import FiLM, NuEmbedding, periodic_conv, zero_init


def _groups(ch: int) -> int:
    return min(8, ch)


class FiLMResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c_in), c_in)
        self.conv1 = periodic_conv(c_in, c_out)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.film = FiLM(emb_dim, c_out)
        self.conv2 = periodic_conv(c_out, c_out)
        self.act = nn.GELU()
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act(self.norm1(h)))
        out = self.conv2(self.act(self.film(self.norm2(out), emb)))
        return self.skip(h) + out


class UNet(nn.Module):
    """widths: channels per resolution level (len(widths) - 1 downsamplings)."""

    def __init__(self, n_in: int = 1, widths: tuple[int, ...] | list[int] = (32, 64, 128),
                 blocks_per_level: int = 2, emb_dim: int = 64,
                 condition_on_nu: bool = True) -> None:
        super().__init__()
        widths = list(widths)
        self.n_in, self.condition_on_nu = n_in, condition_on_nu
        self.downsample_factor = 2 ** (len(widths) - 1)
        self.embed = NuEmbedding(emb_dim)
        self.lift = periodic_conv(n_in, widths[0])

        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, w in enumerate(widths):
            c_in = widths[max(i - 1, 0)] if i > 0 else widths[0]
            level = nn.ModuleList()
            for b in range(blocks_per_level):
                level.append(FiLMResBlock(c_in if b == 0 else w, w, emb_dim))
            self.down_blocks.append(level)
            if i < len(widths) - 1:
                self.downs.append(periodic_conv(w, w, kernel=3, stride=2))

        self.mid = nn.ModuleList([FiLMResBlock(widths[-1], widths[-1], emb_dim)
                                  for _ in range(2)])

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(len(widths) - 2, -1, -1):
            self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                          periodic_conv(widths[i + 1], widths[i])))
            level = nn.ModuleList()
            for b in range(blocks_per_level):
                level.append(FiLMResBlock(2 * widths[i] if b == 0 else widths[i], widths[i],
                                          emb_dim))
            self.up_blocks.append(level)

        self.out_norm = nn.GroupNorm(_groups(widths[0]), widths[0])
        self.proj = zero_init(periodic_conv(widths[0], 1))

    def forward(self, x: torch.Tensor, nu_feat: torch.Tensor) -> torch.Tensor:
        """x: (B, n_in, H, W) with H, W divisible by the downsample factor."""
        if x.shape[-1] % self.downsample_factor or x.shape[-2] % self.downsample_factor:
            raise ValueError(f"grid size must be divisible by {self.downsample_factor}")
        if not self.condition_on_nu:
            nu_feat = torch.zeros_like(nu_feat)
        emb = self.embed(nu_feat)
        h = self.lift(x)
        skips = []
        for i, level in enumerate(self.down_blocks):
            for block in level:
                h = block(h, emb)
            if i < len(self.downs):
                skips.append(h)
                h = self.downs[i](h)
        for block in self.mid:
            h = block(h, emb)
        for up, level in zip(self.ups, self.up_blocks, strict=True):
            h = torch.cat([up(h), skips.pop()], dim=1)
            for block in level:
                h = block(h, emb)
        return x[:, -1:] + self.proj(nn.functional.gelu(self.out_norm(h)))
