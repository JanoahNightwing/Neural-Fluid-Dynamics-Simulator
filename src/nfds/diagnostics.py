"""Physical diagnostics for 2D incompressible flow on a periodic box.

All averages are domain means, so for velocity u and vorticity w:

    E = 1/2 <|u|^2>,   Z = 1/2 <w^2>,   P = 1/2 <|grad w|^2>

Budgets (periodic box, viscosity nu, linear drag alpha, forcing f):

    dE/dt = -2 nu Z - 2 alpha E + <u . f>
    dZ/dt = -2 nu P - 2 alpha Z + <w (curl f)_z>
"""

from __future__ import annotations

import numpy as np

from nfds.grid import SpectralGrid


def energy(grid: SpectralGrid, w_hat: np.ndarray) -> float:
    u, v = grid.velocity(w_hat)
    return float(0.5 * np.mean(u**2 + v**2))


def enstrophy(grid: SpectralGrid, w_hat: np.ndarray) -> float:
    w = grid.ifft(w_hat)
    return float(0.5 * np.mean(w**2))


def palinstrophy(grid: SpectralGrid, w_hat: np.ndarray) -> float:
    wx = grid.ifft(grid.ddx(w_hat))
    wy = grid.ifft(grid.ddy(w_hat))
    return float(0.5 * np.mean(wx**2 + wy**2))


def max_velocity(grid: SpectralGrid, w_hat: np.ndarray) -> float:
    u, v = grid.velocity(w_hat)
    return float(np.sqrt(np.max(u**2 + v**2)))


def max_divergence(grid: SpectralGrid, w_hat: np.ndarray) -> float:
    u, v = grid.velocity(w_hat)
    return float(np.max(np.abs(grid.divergence(u, v))))


def _rfft_weights(grid: SpectralGrid) -> np.ndarray:
    """Multiplicity of each rfft2 coefficient in the full spectrum."""
    w = np.full(grid.spectral_shape, 2.0)
    w[:, 0] = 1.0
    if grid.n % 2 == 0:
        w[:, -1] = 1.0
    return w


def energy_spectrum(grid: SpectralGrid, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shell-summed energy spectrum E(k), with sum_k E(k) == energy().

    Returns (k, E_k) for integer shells k = 0 .. n/2 (in units of 2*pi/L).
    """
    u_hat, v_hat = grid.velocity_hat(w_hat)
    norm = grid.n**2
    power = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / norm**2
    power *= _rfft_weights(grid)
    k_int = np.sqrt(grid.k2) * grid.length / (2 * np.pi)
    shells = np.rint(k_int).astype(int)
    kmax = grid.n // 2
    keep = shells <= kmax
    e_k = np.bincount(shells[keep], weights=power[keep], minlength=kmax + 1)
    # Modes in the corners (|k| > n/2) are dropped; they are dealiased to
    # zero in any resolved simulation.
    return np.arange(kmax + 1), e_k


def resolution_ratio(grid: SpectralGrid, w_hat: np.ndarray) -> float:
    """Energy near the dealiasing cutoff relative to the spectral peak.

    max E(k) over the last 10% of resolved shells, divided by max E(k).
    A well-resolved run keeps this below ~1e-6 at all times; values near
    1e-3 or above mean energy is piling up at the grid scale.
    """
    _, e_k = energy_spectrum(grid, w_hat)
    k_cut = int(grid.dealias * grid.n // 2)
    tail = e_k[int(0.9 * k_cut) : k_cut]
    return float(tail.max() / e_k.max())


def summary(grid: SpectralGrid, w_hat: np.ndarray) -> dict[str, float]:
    return {
        "energy": energy(grid, w_hat),
        "enstrophy": enstrophy(grid, w_hat),
        "palinstrophy": palinstrophy(grid, w_hat),
        "max_velocity": max_velocity(grid, w_hat),
        "mean_vorticity": float(np.real(w_hat[0, 0]) / grid.n**2),
    }
