"""Resolution changes between spectral grids."""

from __future__ import annotations

import torch


def spectral_downsample(w_hat: torch.Tensor, n_in: int, n_out: int) -> torch.Tensor:
    """Ideal low-pass restriction of rfft2 coefficients from an n_in to an n_out grid.

    Keeps modes with |kx|, |ky| < n_out/2 (the n_out Nyquist modes are
    dropped) and returns the *physical* field on the n_out grid, shape
    (..., n_out, n_out). For a field already band-limited below n_out/2 this
    is exact: the result equals the field sampled on the coarse grid.
    """
    if n_out > n_in or n_out % 2:
        raise ValueError("n_out must be even and <= n_in")
    if n_out == n_in:
        return torch.fft.irfft2(w_hat, s=(n_in, n_in))
    h = n_out // 2
    out = torch.zeros((*w_hat.shape[:-2], n_out, h + 1), dtype=w_hat.dtype,
                      device=w_hat.device)
    out[..., :h, :h] = w_hat[..., :h, :h]
    out[..., n_out - h + 1 :, :h] = w_hat[..., n_in - h + 1 :, :h]
    out = out * (n_out / n_in) ** 2
    return torch.fft.irfft2(out, s=(n_out, n_out))


def spectral_resample(w: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample physical periodic fields (..., n, n) to (..., n_out, n_out) in Fourier space.

    Downsampling truncates (ideal low-pass); upsampling zero-pads (band-limited
    interpolation). Modes with |k| >= min(n, n_out)/2 are dropped, so a field that is
    band-limited below both Nyquist limits is reproduced exactly on the new grid.
    """
    n_in = w.shape[-1]
    if n_out % 2 or n_in % 2:
        raise ValueError("grid sizes must be even")
    if n_out == n_in:
        return w.clone()
    w_hat = torch.fft.rfft2(w)
    h = min(n_in, n_out) // 2
    out = torch.zeros((*w.shape[:-2], n_out, n_out // 2 + 1), dtype=w_hat.dtype,
                      device=w.device)
    out[..., :h, :h] = w_hat[..., :h, :h]
    out[..., n_out - h + 1 :, :h] = w_hat[..., n_in - h + 1 :, :h]
    return torch.fft.irfft2(out * (n_out / n_in) ** 2, s=(n_out, n_out))
