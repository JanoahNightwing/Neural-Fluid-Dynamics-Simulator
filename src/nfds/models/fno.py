"""Fourier Neural Operator (Li et al., 2021), written from scratch.

Each Fourier layer computes

    v  ->  act( W v + F^-1[ R . F(v) ] )

where F is the 2D real FFT, R is a learned complex weight applied to the
lowest Fourier modes only (all higher modes are dropped), and W is a
pointwise (1x1) linear map. Because R acts on Fourier coefficients rather
than on pixels, the same weights define an operator at any grid resolution:
the model can be evaluated on grids it was not trained on.

Mode convention (as in the original paper and code): ``modes = m`` keeps
wavenumbers with 0 <= kx < m and -m <= ky < m, i.e. two blocks of rows in the
rfft2 layout ([0, m) and [n - m, n)). Note that the ``neuraloperator``
library's ``n_modes`` counts total modes per axis, so its n_modes = 2m keeps
roughly the same set.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nfds.models.layers import zero_init


class SpectralConv2d(nn.Module):
    """Learned Fourier multiplier on the lowest ``modes`` wavenumbers."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes, modes)
        # Two blocks: non-negative ky rows [0, m) and negative ky rows [n - m, n)
        self.weight_pos = nn.Parameter(scale * torch.rand(shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.rand(shape, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m = self.modes
        if 2 * m > h or m > w // 2 + 1:
            raise ValueError(f"{m} modes need a grid of at least {2 * m}² (got {h}x{w})")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfft2(x.float())
            out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat,
                                 device=x.device)
            out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m, :m],
                                                self.weight_pos)
            out_ft[:, :, h - m :, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, h - m :, :m],
                                                     self.weight_neg)
            # The default ("backward") FFT normalisation makes the output values independent
            # of the grid size, which is what gives resolution invariance.
            return torch.fft.irfft2(out_ft, s=(h, w))


def periodic_positional_encoding(h: int, w: int, length: float = 2 * math.pi,
                                 device=None, dtype=torch.float32) -> torch.Tensor:
    """(4, H, W) channels sin x, cos x, sin y, cos y: smooth, periodic and defined at any
    resolution (unlike raw coordinates, which jump at the boundary)."""
    x = torch.arange(w, device=device, dtype=dtype) * (length / w)
    y = torch.arange(h, device=device, dtype=dtype) * (length / h)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    k = 2 * math.pi / length
    return torch.stack([torch.sin(k * xx), torch.cos(k * xx), torch.sin(k * yy),
                        torch.cos(k * yy)])


class FNO(nn.Module):
    """FNO surrogate predicting the next frame as a residual on the last input frame.

    Parameters
    ----------
    n_in : input frames
    width : channels in the Fourier layers
    modes : retained Fourier modes per direction (see module docstring)
    n_layers : Fourier layers
    positional_encoding : append sin/cos coordinate channels. The flow itself is
        translation-invariant, but Kolmogorov forcing (sin 4y) is not, so knowing
        y can help. Turning it off makes the model exactly translation-equivariant.
    condition_on_nu : append the normalised log-viscosity as a constant channel
    channel_mlp : add a small per-pixel MLP (width -> width/2 -> width, residual) after
        each Fourier layer, as the ``neuraloperator`` library does by default. Off gives
        the original paper's architecture.
    """

    def __init__(self, n_in: int = 1, width: int = 32, modes: int = 12, n_layers: int = 4,
                 positional_encoding: bool = True, condition_on_nu: bool = True,
                 proj_width: int = 128, channel_mlp: bool = False) -> None:
        super().__init__()
        self.n_in, self.modes = n_in, modes
        self.positional_encoding, self.condition_on_nu = positional_encoding, condition_on_nu
        c_in = n_in + 4 * positional_encoding + int(condition_on_nu)
        self.lift = nn.Conv2d(c_in, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes)
                                       for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width // 2, 1), nn.GELU(),
                          nn.Conv2d(width // 2, width, 1))
            for _ in range(n_layers)]) if channel_mlp else None
        self.act = nn.GELU()
        self.proj1 = nn.Conv2d(width, proj_width, 1)
        self.proj2 = zero_init(nn.Conv2d(proj_width, 1, 1))

    def forward(self, x: torch.Tensor, nu_feat: torch.Tensor) -> torch.Tensor:
        """x: (B, n_in, H, W) at any resolution with H, W >= 2 * modes."""
        b, _, h, w = x.shape
        feats = [x]
        if self.positional_encoding:
            pe = periodic_positional_encoding(h, w, device=x.device, dtype=x.dtype)
            feats.append(pe.expand(b, -1, -1, -1))
        if self.condition_on_nu:
            feats.append(nu_feat.view(-1, 1, 1, 1).to(x.dtype).expand(b, 1, h, w))
        v = self.lift(torch.cat(feats, dim=1))
        for i, (spec, pw) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
            v = spec(v) + pw(v)
            if i < len(self.spectral) - 1:
                v = self.act(v)
            if self.mlps is not None:
                v = v + self.mlps[i](v)
        return x[:, -1:] + self.proj2(self.act(self.proj1(v)))
