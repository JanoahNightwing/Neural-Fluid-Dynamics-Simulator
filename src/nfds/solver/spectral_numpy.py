"""Pseudo-spectral solver for 2D incompressible Navier-Stokes in vorticity form.

    d(omega)/dt + u . grad(omega) = nu lap(omega) - alpha omega + (curl f)_z
    lap(psi) = -omega,   u = (d psi/dy, -d psi/dx)

The linear terms L = -nu k^2 - alpha are integrated exactly with an
integrating factor; the non-linear advection term and the forcing are
integrated with an explicit Runge-Kutta method (Heun = RK2, or classic RK4).
The advection term is dealiased with the 2/3 rule.

This NumPy implementation is the readable reference. The batched PyTorch
solver in milestone M1 is tested against it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nfds.grid import SpectralGrid

Scheme = Literal["ifrk2", "ifrk4"]


@dataclass
class SimulationResult:
    times: np.ndarray  # (T,)
    vorticity: np.ndarray  # (T, n, n) physical space
    n_steps: int
    meta: dict = field(default_factory=dict)


class SpectralVorticitySolver:
    """Integrating-factor Runge-Kutta solver on a periodic box.

    Parameters
    ----------
    grid : SpectralGrid
    nu : float
        Kinematic viscosity.
    drag : float
        Linear (Rayleigh) drag coefficient alpha.
    forcing_vorticity : array (n, n), optional
        Time-independent (curl f)_z in physical space.
    scheme : "ifrk2" | "ifrk4"
        Formal temporal order 2 or 4.
    """

    def __init__(
        self,
        grid: SpectralGrid,
        nu: float,
        drag: float = 0.0,
        forcing_vorticity: np.ndarray | None = None,
        scheme: Scheme = "ifrk4",
    ) -> None:
        if scheme not in ("ifrk2", "ifrk4"):
            raise ValueError(f"unknown scheme {scheme!r}")
        self.grid = grid
        self.nu = float(nu)
        self.drag = float(drag)
        self.scheme = scheme
        self.linear = -self.nu * grid.k2 - self.drag  # spectral multiplier L
        if forcing_vorticity is None:
            self.forcing_hat = np.zeros(grid.spectral_shape, dtype=complex)
        else:
            self.forcing_hat = grid.truncate(grid.fft(forcing_vorticity))
        self._cached_dt: float | None = None
        self._e_half: np.ndarray | None = None
        self._e_full: np.ndarray | None = None

    # ------------------------------------------------------------ physics
    def advection(self, w_hat: np.ndarray) -> np.ndarray:
        """-u . grad(omega), dealiased, in spectral space."""
        g = self.grid
        u_hat, v_hat = g.velocity_hat(w_hat)
        u, v = g.ifft(u_hat), g.ifft(v_hat)
        wx, wy = g.ifft(g.ddx(w_hat)), g.ifft(g.ddy(w_hat))
        return g.truncate(g.fft(-(u * wx + v * wy)))

    def nonlinear(self, w_hat: np.ndarray) -> np.ndarray:
        """Everything that is not integrated exactly: advection + forcing."""
        return self.advection(w_hat) + self.forcing_hat

    # ------------------------------------------------------- time stepping
    def _factors(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if dt != self._cached_dt:
            self._e_half = np.exp(self.linear * dt / 2)
            self._e_full = self._e_half**2
            self._cached_dt = dt
        return self._e_half, self._e_full  # type: ignore[return-value]

    def step(self, w_hat: np.ndarray, dt: float) -> np.ndarray:
        """Advance the spectral vorticity by one step of size dt."""
        e_half, e_full = self._factors(dt)
        n = self.nonlinear
        if self.scheme == "ifrk2":
            a = dt * n(w_hat)
            b = dt * n(e_full * (w_hat + a))
            out = e_full * (w_hat + 0.5 * a) + 0.5 * b
        else:
            a = dt * n(w_hat)
            b = dt * n(e_half * (w_hat + 0.5 * a))
            c = dt * n(e_half * w_hat + 0.5 * b)
            d = dt * n(e_full * w_hat + e_half * c)
            out = e_full * w_hat + (e_full * a + 2 * e_half * (b + c) + d) / 6
        out[0, 0] = 0.0  # mean vorticity is identically zero on a periodic box
        return out

    def cfl_dt(self, w_hat: np.ndarray, cfl: float) -> float:
        u, v = self.grid.velocity(w_hat)
        umax = float(np.sqrt(np.max(u**2 + v**2)))
        return math.inf if umax == 0 else cfl * self.grid.dx / umax

    # ----------------------------------------------------------- driver
    def simulate(
        self,
        w0: np.ndarray,
        t_final: float,
        save_dt: float,
        *,
        dt: float | None = None,
        cfl: float = 0.5,
        max_dt: float | None = None,
        callback: Callable[[float, np.ndarray], None] | None = None,
    ) -> SimulationResult:
        """Integrate from w0 (physical space) and save every save_dt.

        With ``dt`` given, steps are fixed (dt must divide save_dt). Otherwise
        dt is chosen from the CFL condition at the start of each save interval
        and adjusted so that save times are hit exactly.

        The initial condition is projected onto the resolved modes first, so
        the discrete system conserves energy and enstrophy exactly when
        nu = drag = forcing = 0 (up to time-stepping error).
        """
        g = self.grid
        w_hat = g.truncate(g.fft(w0))
        w_hat[0, 0] = 0.0
        n_saves = int(round(t_final / save_dt))
        if not math.isclose(n_saves * save_dt, t_final, rel_tol=1e-9):
            raise ValueError("t_final must be a multiple of save_dt")
        if max_dt is None:
            max_dt = save_dt

        frames = [g.ifft(w_hat)]
        times = [0.0]
        t, total_steps = 0.0, 0
        for i in range(n_saves):
            if dt is not None:
                n_sub = int(round(save_dt / dt))
                if not math.isclose(n_sub * dt, save_dt, rel_tol=1e-9):
                    raise ValueError("dt must divide save_dt")
            else:
                dt_cfl = min(self.cfl_dt(w_hat, cfl), max_dt)
                n_sub = max(1, math.ceil(save_dt / dt_cfl - 1e-12))
            h = save_dt / n_sub
            for _ in range(n_sub):
                w_hat = self.step(w_hat, h)
            total_steps += n_sub
            t = (i + 1) * save_dt
            if not np.all(np.isfinite(w_hat)):
                raise FloatingPointError(f"solution blew up at t = {t:.4g}")
            w = g.ifft(w_hat)
            frames.append(w)
            times.append(t)
            if callback is not None:
                callback(t, w_hat)

        return SimulationResult(
            times=np.asarray(times),
            vorticity=np.stack(frames),
            n_steps=total_steps,
            meta={"nu": self.nu, "drag": self.drag, "scheme": self.scheme, "n": g.n},
        )
