"""Initial vorticity fields. Every function returns a real (n, n) array."""

from __future__ import annotations

import numpy as np

from nfds.grid import SpectralGrid


def taylor_green(grid: SpectralGrid, k: int = 1, amplitude: float = 1.0) -> np.ndarray:
    """Taylor-Green vortex: u = A sin(kx) cos(ky), v = -A cos(kx) sin(ky).

    omega = 2 A k sin(kx) sin(ky). It is an exact Navier-Stokes solution
    (the non-linear term vanishes) and decays as exp(-2 nu k^2 t).
    """
    x, y = grid.coords
    return 2 * amplitude * k * np.sin(k * x) * np.sin(k * y)


def taylor_green_exact(
    grid: SpectralGrid, t: float, nu: float, drag: float = 0.0, k: int = 1, amplitude: float = 1.0
) -> np.ndarray:
    return taylor_green(grid, k, amplitude) * np.exp(-(2 * nu * k**2 + drag) * t)


def random_field(
    grid: SpectralGrid,
    k_peak: float = 4.0,
    energy: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Gaussian random vorticity with energy spectrum E(k) ~ k^4 exp(-(k/k_peak)^2).

    Built by filtering white noise in Fourier space, so the result is real and
    exactly periodic. The mean is removed, the field is truncated to the
    resolved modes, and it is rescaled to the requested kinetic energy.
    """
    from nfds.diagnostics import energy as energy_of

    rng = np.random.default_rng(seed)
    noise_hat = grid.fft(rng.standard_normal((grid.n, grid.n)))
    k = np.sqrt(grid.k2) * grid.length / (2 * np.pi)
    # Per-mode vorticity variance ~ k^2 E(k) / k  (shell has ~2 pi k modes)
    amp = np.zeros_like(k)
    nz = k > 0
    amp[nz] = k[nz] ** 2.5 * np.exp(-0.5 * (k[nz] / k_peak) ** 2)
    w_hat = grid.truncate(noise_hat * amp)
    w_hat[0, 0] = 0.0
    e0 = energy_of(grid, w_hat)
    w_hat *= np.sqrt(energy / e0)
    return grid.ifft(w_hat)


def shear_layer(
    grid: SpectralGrid, thickness: float = 0.03, perturbation: float = 0.05, mode: int = 2
) -> np.ndarray:
    """Double shear layer (two Kelvin-Helmholtz-unstable tanh jets), periodic in y.

    u = tanh((y - L/4)/d) for y < L/2 and tanh((3L/4 - y)/d) otherwise,
    v = perturbation * sin(mode * x), with d = thickness * L, computed on the
    domain scaled to [0, 2pi). Needs n >= 128 to resolve the default thickness.
    """
    x, y = grid.coords
    s = 2 * np.pi / grid.length
    xs, ys = x * s, y * s
    d = thickness * 2 * np.pi
    u = np.where(ys <= np.pi, np.tanh((ys - np.pi / 2) / d), np.tanh((3 * np.pi / 2 - ys) / d))
    v = perturbation * np.sin(mode * xs)
    w = grid.vorticity_from_velocity(u, v)
    w_hat = grid.truncate(grid.fft(w))
    w_hat[0, 0] = 0.0
    return grid.ifft(w_hat)


def vortex_dipole(
    grid: SpectralGrid, separation: float = 0.6, radius: float = 0.25, strength: float = 10.0
) -> np.ndarray:
    """Two opposite-signed Gaussian vortices centred in the domain."""
    x, y = grid.coords
    c = grid.length / 2

    def gaussian(x0: float, y0: float) -> np.ndarray:
        # periodic distance
        dx = (x - x0 + c) % grid.length - c
        dy = (y - y0 + c) % grid.length - c
        return np.exp(-(dx**2 + dy**2) / (2 * radius**2))

    w = strength * (gaussian(c - separation / 2, c) - gaussian(c + separation / 2, c))
    w_hat = grid.truncate(grid.fft(w))
    w_hat[0, 0] = 0.0
    return grid.ifft(w_hat)
