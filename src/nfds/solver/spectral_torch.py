"""Batched, differentiable PyTorch version of the spectral vorticity solver.

Same equations, conventions and time-stepping schemes as
``nfds.solver.spectral_numpy`` (the reference implementation it is tested
against), with three additions:

* **Batching.** The state has shape (B, n, n//2 + 1); every trajectory in a
  batch may have its own viscosity, so one call generates a whole slice of a
  dataset.
* **Devices and precision.** Runs on CPU or GPU, in float64 (verification)
  or float32 (data generation).
* **Differentiability.** ``step`` uses only out-of-place operations, so
  gradients flow through any number of solver steps. This is what the hybrid
  "solver-in-the-loop" model in milestone M7 relies on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

Scheme = Literal["ifrk2", "ifrk4"]
ArrayLike = np.ndarray | torch.Tensor


class TorchSpectralGrid:
    """Periodic grid on [0, L)^2 with wavenumbers stored as tensors.

    Same layout as ``nfds.grid.SpectralGrid``: fields are indexed
    ``[..., iy, ix]`` and spectral fields use ``torch.fft.rfft2``.
    """

    def __init__(
        self,
        n: int,
        length: float = 2 * math.pi,
        dealias: float = 2.0 / 3.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n % 2:
            raise ValueError("n must be even")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        self.n, self.length, self.dealias = n, float(length), float(dealias)
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        scale = 2 * math.pi / length
        kw = {"device": self.device, "dtype": dtype}
        kx_1d = torch.fft.rfftfreq(n, d=1.0 / n, **kw) * scale
        ky_1d = torch.fft.fftfreq(n, d=1.0 / n, **kw) * scale
        self.ky, self.kx = torch.meshgrid(ky_1d, kx_1d, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.inv_k2 = torch.where(self.k2 > 0, 1.0 / self.k2.clamp_min(1e-30),
                                  torch.zeros_like(self.k2))
        cut = self.dealias * (n // 2)
        self.mask = ((self.kx.abs() / scale < cut) & (self.ky.abs() / scale < cut)).to(dtype)
        zero_mean = torch.ones_like(self.k2)
        zero_mean[0, 0] = 0.0
        self.zero_mean = zero_mean

    # ---------------------------------------------------------------- basics
    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def spectral_shape(self) -> tuple[int, int]:
        return (self.n, self.n // 2 + 1)

    def coords(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.arange(self.n, device=self.device, dtype=self.dtype) * self.dx
        y_grid, x_grid = torch.meshgrid(x, x, indexing="ij")
        return x_grid, y_grid

    def as_tensor(self, f: ArrayLike) -> torch.Tensor:
        return torch.as_tensor(f, device=self.device, dtype=self.dtype)

    def fft(self, f: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(f)

    def ifft(self, f_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(f_hat, s=(self.n, self.n))

    # ------------------------------------------------------------- operators
    def ddx(self, f_hat: torch.Tensor) -> torch.Tensor:
        return 1j * self.kx * f_hat

    def ddy(self, f_hat: torch.Tensor) -> torch.Tensor:
        return 1j * self.ky * f_hat

    def velocity_hat(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi_hat = w_hat * self.inv_k2
        return self.ddy(psi_hat), -self.ddx(psi_hat)

    def velocity(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat, v_hat = self.velocity_hat(w_hat)
        return self.ifft(u_hat), self.ifft(v_hat)

    def truncate(self, f_hat: torch.Tensor) -> torch.Tensor:
        return f_hat * self.mask

    # ----------------------------------------------------- batched diagnostics
    def energy(self, w_hat: torch.Tensor) -> torch.Tensor:
        """Kinetic energy 1/2 <|u|^2> per batch element."""
        u, v = self.velocity(w_hat)
        return 0.5 * (u**2 + v**2).mean(dim=(-2, -1))

    def enstrophy(self, w_hat: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.ifft(w_hat) ** 2).mean(dim=(-2, -1))

    def max_velocity(self, w_hat: torch.Tensor) -> torch.Tensor:
        u, v = self.velocity(w_hat)
        return torch.sqrt((u**2 + v**2).amax(dim=(-2, -1)))

    def energy_spectrum(self, w_hat: torch.Tensor) -> torch.Tensor:
        """Shell-summed E(k) for k = 0..n/2, shape (..., n/2 + 1)."""
        u_hat, v_hat = self.velocity_hat(w_hat)
        power = 0.5 * (u_hat.abs() ** 2 + v_hat.abs() ** 2) / self.n**4
        weights = torch.full(self.spectral_shape, 2.0, device=self.device, dtype=self.dtype)
        weights[:, 0] = 1.0
        weights[:, -1] = 1.0
        power = power * weights
        shells = torch.round(torch.sqrt(self.k2) * self.length / (2 * math.pi)).long()
        kmax = self.n // 2
        keep = shells <= kmax
        flat = power[..., keep]  # (..., M)
        idx = shells[keep]
        out = torch.zeros((*flat.shape[:-1], kmax + 1), device=self.device, dtype=self.dtype)
        return out.index_add_(-1, idx, flat)

    def resolution_ratio(self, w_hat: torch.Tensor) -> torch.Tensor:
        """Max E(k) over the last 10% of resolved shells / max E(k) (see diagnostics.py)."""
        e_k = self.energy_spectrum(w_hat)
        k_cut = int(self.dealias * self.n // 2)
        tail = e_k[..., int(0.9 * k_cut) : k_cut]
        return tail.amax(dim=-1) / e_k.amax(dim=-1)


@dataclass
class TorchSimulationResult:
    times: torch.Tensor  # (T,)
    vorticity: torch.Tensor  # (B, T, n, n)
    n_steps: int
    meta: dict = field(default_factory=dict)


class TorchVorticitySolver:
    """Batched integrating-factor Runge-Kutta solver.

    Parameters
    ----------
    grid : TorchSpectralGrid
    nu : float or sequence/tensor of shape (B,)
        Viscosity, either shared or one per batch element.
    drag : float
        Linear drag coefficient alpha.
    forcing_vorticity : array (n, n), optional
        Time-independent (curl f)_z, shared by the whole batch.
    scheme : "ifrk2" | "ifrk4"
    """

    def __init__(
        self,
        grid: TorchSpectralGrid,
        nu: float | ArrayLike,
        drag: float = 0.0,
        forcing_vorticity: ArrayLike | None = None,
        scheme: Scheme = "ifrk4",
    ) -> None:
        if scheme not in ("ifrk2", "ifrk4"):
            raise ValueError(f"unknown scheme {scheme!r}")
        self.grid = grid
        self.scheme = scheme
        self.drag = float(drag)
        nu_t = grid.as_tensor(nu).reshape(-1)
        self.nu = nu_t
        self.batch_size = None if nu_t.numel() == 1 else nu_t.numel()
        self.linear = -nu_t.view(-1, 1, 1) * grid.k2 - self.drag  # (1 or B, n, m)
        if forcing_vorticity is None:
            self.forcing_hat = torch.zeros(grid.spectral_shape, device=grid.device,
                                           dtype=grid.cdtype)
        else:
            self.forcing_hat = grid.truncate(grid.fft(grid.as_tensor(forcing_vorticity)))
        self._cache: tuple[float, torch.Tensor, torch.Tensor] | None = None

    # -------------------------------------------------------------- physics
    def advection(self, w_hat: torch.Tensor) -> torch.Tensor:
        g = self.grid
        u_hat, v_hat = g.velocity_hat(w_hat)
        u, v = g.ifft(u_hat), g.ifft(v_hat)
        wx, wy = g.ifft(g.ddx(w_hat)), g.ifft(g.ddy(w_hat))
        return g.truncate(g.fft(-(u * wx + v * wy)))

    def nonlinear(self, w_hat: torch.Tensor) -> torch.Tensor:
        return self.advection(w_hat) + self.forcing_hat

    # --------------------------------------------------------- time stepping
    def _factors(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is None or self._cache[0] != dt:
            e_half = torch.exp(self.linear * (dt / 2))
            self._cache = (dt, e_half, e_half**2)
        return self._cache[1], self._cache[2]

    def step(self, w_hat: torch.Tensor, dt: float) -> torch.Tensor:
        """One differentiable step for a (B, n, n//2+1) or (n, n//2+1) state."""
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
        return out * self.grid.zero_mean

    def cfl_dt(self, w_hat: torch.Tensor, cfl: float) -> float:
        """Largest stable step for the whole batch (one dt shared by all samples)."""
        umax = float(self.grid.max_velocity(w_hat).max())
        return math.inf if umax == 0 else cfl * self.grid.dx / umax

    # ---------------------------------------------------------------- driver
    def initial_state(self, w0: ArrayLike) -> torch.Tensor:
        """Physical vorticity (n, n) or (B, n, n) -> truncated, zero-mean spectrum."""
        g = self.grid
        w_hat = g.truncate(g.fft(g.as_tensor(w0))) * g.zero_mean
        if w_hat.dim() == 2:
            w_hat = w_hat.unsqueeze(0)
        if self.batch_size is not None and w_hat.shape[0] != self.batch_size:
            if w_hat.shape[0] != 1:
                raise ValueError(f"batch of {w_hat.shape[0]} initial conditions but "
                                 f"{self.batch_size} viscosities")
            w_hat = w_hat.expand(self.batch_size, -1, -1)
        return w_hat

    def simulate(
        self,
        w0: ArrayLike,
        t_final: float,
        save_dt: float,
        *,
        dt: float | None = None,
        cfl: float = 0.5,
        max_dt: float | None = None,
        requires_grad: bool = False,
        store_device: str | torch.device | None = "cpu",
        store_dtype: torch.dtype | None = None,
        callback: Callable[[float, torch.Tensor], None] | None = None,
    ) -> TorchSimulationResult:
        """Integrate a batch and save every save_dt (same semantics as the NumPy solver).

        Returns vorticity of shape (B, T, n, n). Frames are moved to
        ``store_device`` (default CPU, so long GPU runs don't exhaust GPU
        memory) and optionally cast to ``store_dtype``. With
        ``requires_grad=True`` frames stay on the compute device and keep
        their autograd history.
        """
        g = self.grid
        n_saves = int(round(t_final / save_dt))
        if not math.isclose(n_saves * save_dt, t_final, rel_tol=1e-9):
            raise ValueError("t_final must be a multiple of save_dt")
        if dt is not None:
            n_fixed = int(round(save_dt / dt))
            if not math.isclose(n_fixed * dt, save_dt, rel_tol=1e-9):
                raise ValueError("dt must divide save_dt")
        max_dt = save_dt if max_dt is None else max_dt
        if requires_grad:
            store_device, store_dtype = None, None

        def store(w_hat: torch.Tensor) -> torch.Tensor:
            w = g.ifft(w_hat)
            if store_dtype is not None:
                w = w.to(store_dtype)
            return w.to(store_device) if store_device is not None else w

        with torch.set_grad_enabled(requires_grad):
            w_hat = self.initial_state(w0)
            frames = [store(w_hat)]
            total = 0
            for i in range(n_saves):
                if dt is not None:
                    n_sub = n_fixed
                else:
                    n_sub = max(1, math.ceil(save_dt / min(self.cfl_dt(w_hat, cfl), max_dt)
                                             - 1e-12))
                h = save_dt / n_sub
                for _ in range(n_sub):
                    w_hat = self.step(w_hat, h)
                total += n_sub
                if not bool(torch.isfinite(torch.view_as_real(w_hat)).all()):
                    raise FloatingPointError(f"solution blew up at t = {(i + 1) * save_dt:.4g}")
                frames.append(store(w_hat))
                if callback is not None:
                    callback((i + 1) * save_dt, w_hat)

        times = torch.arange(n_saves + 1, dtype=torch.float64) * save_dt
        return TorchSimulationResult(
            times=times,
            vorticity=torch.stack(frames, dim=1),
            n_steps=total,
            meta={"nu": self.nu.detach().cpu().numpy(), "drag": self.drag,
                  "scheme": self.scheme, "n": g.n, "dtype": str(g.dtype),
                  "device": str(g.device)},
        )
