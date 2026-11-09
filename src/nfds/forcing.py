"""Body forcings. Each provides its velocity form f and vorticity form (curl f)_z."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfds.grid import SpectralGrid


@dataclass(frozen=True)
class KolmogorovForcing:
    """f = (A sin(k_f y), 0), so (curl f)_z = -A k_f cos(k_f y).

    With linear drag this drives statistically stationary 2D turbulence
    (k_f = 4, A = 1, drag = 0.1 is the setup of Kochkov et al., 2021).
    """

    wavenumber: int = 4
    amplitude: float = 1.0

    def velocity(self, grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray]:
        _, y = grid.coords
        fx = self.amplitude * np.sin(self.wavenumber * y)
        return fx, np.zeros_like(fx)

    def vorticity(self, grid: SpectralGrid) -> np.ndarray:
        _, y = grid.coords
        return -self.amplitude * self.wavenumber * np.cos(self.wavenumber * y)

    def laminar_vorticity(self, grid: SpectralGrid, nu: float, drag: float) -> np.ndarray:
        """Steady laminar solution: omega = curl f / (nu k_f^2 + drag)."""
        return self.vorticity(grid) / (nu * self.wavenumber**2 + drag)
