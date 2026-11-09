"""Periodic grid and Fourier operators on [0, L)^2.

Conventions
-----------
* Fields are real arrays of shape (N, N) indexed as ``field[iy, ix]``:
  axis 0 is y, axis 1 is x.
* Spectral fields use ``numpy.fft.rfft2`` and have shape (N, N//2 + 1);
  the halved axis is x.
* Wavenumbers are angular: k = 2*pi*n / L, so for L = 2*pi they are integers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpectralGrid:
    """Uniform periodic grid with precomputed wavenumbers and masks.

    Parameters
    ----------
    n : int
        Grid points per side (even).
    length : float
        Domain side length L (default 2*pi).
    dealias : float
        Fraction of the maximum wavenumber to keep. 2/3 is Orszag's rule,
        which removes all aliasing errors from quadratic products.
    """

    n: int
    length: float = 2 * np.pi
    dealias: float = 2.0 / 3.0

    kx: np.ndarray = field(init=False, repr=False)
    ky: np.ndarray = field(init=False, repr=False)
    k2: np.ndarray = field(init=False, repr=False)
    inv_k2: np.ndarray = field(init=False, repr=False)
    mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n % 2:
            raise ValueError("n must be even")
        n, length = self.n, self.length
        scale = 2 * np.pi / length
        kx_1d = np.fft.rfftfreq(n, d=1.0 / n) * scale  # 0..n/2
        ky_1d = np.fft.fftfreq(n, d=1.0 / n) * scale  # 0..n/2-1, -n/2..-1
        self.kx, self.ky = np.meshgrid(kx_1d, ky_1d)  # shape (n, n//2+1)
        self.k2 = self.kx**2 + self.ky**2
        self.inv_k2 = np.zeros_like(self.k2)
        nz = self.k2 > 0
        self.inv_k2[nz] = 1.0 / self.k2[nz]

        # Rectangular 2/3 mask in integer wavenumber units. The Nyquist
        # row/column is always removed (it has no well-defined derivative).
        kmax_int = n // 2
        cut = self.dealias * kmax_int
        nx = np.abs(self.kx / scale)
        ny = np.abs(self.ky / scale)
        self.mask = (nx < cut) & (ny < cut)

    # ------------------------------------------------------------------ grid
    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical coordinates (X, Y), each of shape (n, n)."""
        x = np.arange(self.n) * self.dx
        return np.meshgrid(x, x)  # X varies along axis 1, Y along axis 0

    @property
    def spectral_shape(self) -> tuple[int, int]:
        return (self.n, self.n // 2 + 1)

    # ------------------------------------------------------------ transforms
    def fft(self, f: np.ndarray) -> np.ndarray:
        return np.fft.rfft2(f)

    def ifft(self, f_hat: np.ndarray) -> np.ndarray:
        return np.fft.irfft2(f_hat, s=(self.n, self.n))

    # ------------------------------------------------------------- operators
    def ddx(self, f_hat: np.ndarray) -> np.ndarray:
        return 1j * self.kx * f_hat

    def ddy(self, f_hat: np.ndarray) -> np.ndarray:
        return 1j * self.ky * f_hat

    def laplacian(self, f_hat: np.ndarray) -> np.ndarray:
        return -self.k2 * f_hat

    def streamfunction(self, w_hat: np.ndarray) -> np.ndarray:
        """Solve laplacian(psi) = -omega; the k = 0 mode is set to zero."""
        return w_hat * self.inv_k2

    def velocity_hat(self, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u = d(psi)/dy, v = -d(psi)/dx, returned in spectral space."""
        psi_hat = self.streamfunction(w_hat)
        return self.ddy(psi_hat), -self.ddx(psi_hat)

    def velocity(self, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_hat, v_hat = self.velocity_hat(w_hat)
        return self.ifft(u_hat), self.ifft(v_hat)

    def vorticity_from_velocity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """omega = dv/dx - du/dy (physical space in, physical space out)."""
        return self.ifft(self.ddx(self.fft(v)) - self.ddy(self.fft(u)))

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.ifft(self.ddx(self.fft(u)) + self.ddy(self.fft(v)))

    def truncate(self, f_hat: np.ndarray) -> np.ndarray:
        """Project onto the resolved (dealiased) modes."""
        return f_hat * self.mask
