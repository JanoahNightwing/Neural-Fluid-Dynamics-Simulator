"""Evaluation metrics (proposal section 7). All take tensors shaped (N, T, H, W)."""

from __future__ import annotations

import math

import torch

from nfds.solver.spectral_torch import TorchSpectralGrid


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """||pred - true|| / ||true|| per sample and time, shape (N, T)."""
    num = (pred - true).flatten(2).norm(dim=-1)
    return num / true.flatten(2).norm(dim=-1).clamp_min(1e-12)


def correlation(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between fields per sample and time, shape (N, T)."""
    p = pred.flatten(2) - pred.flatten(2).mean(-1, keepdim=True)
    t = true.flatten(2) - true.flatten(2).mean(-1, keepdim=True)
    return (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-12)


def high_correlation_time(corr: torch.Tensor, dt: float, threshold: float = 0.8
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Time until correlation first drops below ``threshold`` (Kochkov et al., 2021).

    Returns (time, censored): ``censored`` marks samples that never dropped
    below the threshold within the horizon; their time is the full horizon.
    NaN correlations count as below the threshold.
    """
    below = ~(corr >= threshold)  # NaN -> True
    n_t = corr.shape[1]
    first = torch.where(below.any(dim=1), below.float().argmax(dim=1),
                        torch.full((corr.shape[0],), n_t, device=corr.device))
    return first.double() * dt, ~below.any(dim=1)


def failure_step(pred: torch.Tensor, true_energy_mean: torch.Tensor, grid: TorchSpectralGrid,
                 factor: float = 10.0) -> torch.Tensor:
    """First step where the prediction is non-finite or its energy exceeds ``factor`` times
    the true mean energy; T (the horizon) if it never fails. Shape (N,)."""
    e = field_energy(pred, grid)
    bad = ~torch.isfinite(e) | (e > factor * true_energy_mean[:, None])
    n_t = pred.shape[1]
    return torch.where(bad.any(dim=1), bad.float().argmax(dim=1),
                       torch.full((pred.shape[0],), n_t, device=pred.device))


def field_energy(w: torch.Tensor, grid: TorchSpectralGrid) -> torch.Tensor:
    """Kinetic energy of each frame, (N, T)."""
    return grid.energy(grid.fft(w.to(grid.dtype)))


def field_enstrophy(w: torch.Tensor) -> torch.Tensor:
    return 0.5 * (w.double() ** 2).mean(dim=(-2, -1))


def mean_spectrum(w: torch.Tensor, grid: TorchSpectralGrid) -> torch.Tensor:
    """Energy spectrum averaged over samples, for each time: (T, n/2 + 1)."""
    spec = grid.energy_spectrum(grid.fft(w.to(grid.dtype)))  # (N, T, K)
    finite = torch.isfinite(spec).all(dim=-1, keepdim=True)
    spec = torch.where(finite, spec, torch.zeros_like(spec))
    return spec.sum(0) / finite.sum(0).clamp_min(1)


def spectral_error(pred_spec: torch.Tensor, true_spec: torch.Tensor,
                   bands: dict[str, tuple[int, int]]) -> dict[str, float]:
    """Mean absolute log10 error of E(k) within wavenumber bands."""
    out = {}
    for name, (lo, hi) in bands.items():
        p, t = pred_spec[..., lo:hi], true_spec[..., lo:hi]
        out[name] = float((torch.log10(p.clamp_min(1e-30)) - torch.log10(t.clamp_min(1e-30)))
                          .abs().mean())
    return out


def default_bands(n: int) -> dict[str, tuple[int, int]]:
    """Large, intermediate and small scales on an n² grid (integer shells)."""
    k_cut = n // 3
    return {"large (k<5)": (1, 5), "inertial (5<=k<12)": (5, 12),
            f"small (12<=k<{k_cut})": (12, k_cut)}


def nanmean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    mask = torch.isfinite(x)
    return torch.where(mask, x, torch.zeros_like(x)).sum(dim) / mask.sum(dim).clamp_min(1)


def summarise_curve(x: torch.Tensor) -> dict[str, list[float]]:
    """Mean and std over samples of an (N, T) curve, ignoring non-finite values."""
    mean = nanmean(x, 0)
    var = nanmean((x - mean) ** 2, 0)
    return {"mean": mean.tolist(), "std": var.sqrt().tolist()}


def finite_or_none(v: float) -> float | None:
    return v if math.isfinite(v) else None
