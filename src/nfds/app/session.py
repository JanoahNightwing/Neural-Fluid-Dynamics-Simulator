"""One user's simulation: the reference solver and a predictor advancing in lockstep.

The reference runs on a finer grid (``ref_n``, default 128) with adaptive time steps; the
predictor runs on its training grid. Both start from the same field, receive the same brush
edits, and are compared on the predictor's grid after every saved interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from nfds.app.registry import Entry
from nfds.data.transforms import spectral_resample
from nfds.evaluation.steppers import CoarseSolverStepper, PersistenceStepper, Stepper
from nfds.forcing import KolmogorovForcing
from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field, shear_layer, taylor_green, vortex_dipole
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver

PRESETS = {
    "random": "Random turbulence",
    "shear_layer": "Shear layer",
    "dipole": "Vortex pair",
    "taylor_green": "Taylor–Green cells",
    "calm": "Still fluid",
}


def make_initial_field(n: int, kind: str, seed: int = 0, k_peak: float = 4.0,
                       energy: float = 0.5, length: float = 2 * math.pi) -> np.ndarray:
    g = SpectralGrid(n, length=length)
    if kind == "random":
        return random_field(g, k_peak=k_peak, energy=energy, seed=seed)
    if kind == "calm":
        return np.zeros((n, n))
    if kind == "shear_layer":
        w = shear_layer(g, thickness=0.03, perturbation=0.05, mode=2)
    elif kind == "dipole":
        w = vortex_dipole(g, separation=0.8, radius=0.3, strength=10.0)
    elif kind == "taylor_green":
        rng = np.random.default_rng(seed)
        w = taylor_green(g, k=2) + 0.05 * random_field(g, k_peak=5, energy=1.0, seed=int(
            rng.integers(2**31)))
    else:
        raise ValueError(f"unknown initial condition {kind!r}")
    w_hat = g.truncate(g.fft(w))
    w_hat[0, 0] = 0.0
    from nfds.diagnostics import energy as energy_of

    e = energy_of(g, w_hat)
    return g.ifft(w_hat * math.sqrt(energy / e)) if e > 0 else g.ifft(w_hat)


def gaussian_vortex(n: int, x: float, y: float, radius: float, strength: float,
                    length: float = 2 * math.pi) -> np.ndarray:
    """Periodic Gaussian blob of vorticity centred at (x, y) given as fractions of the box."""
    s = np.arange(n) * (length / n)
    xx, yy = np.meshgrid(s, s)
    dx = (xx - x * length + length / 2) % length - length / 2
    dy = (yy - y * length + length / 2) % length - length / 2
    return strength * np.exp(-(dx**2 + dy**2) / (2 * radius**2))


@dataclass
class Metrics:
    t: float
    rel_l2: float | None
    correlation: float | None
    energy_ref: float
    energy_model: float | None
    diverged: bool
    ref_resolution_ratio: float  # > ~1e-4: the reference itself is under-resolved

    def as_dict(self) -> dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Session:
    entry: Entry
    stepper: Stepper
    ref_n: int = 128
    device: str = "cpu"
    nu: float = 3e-3
    forcing: bool = True
    t: float = 0.0
    steps: int = 0
    diverged: bool = False
    _ref_state: torch.Tensor | None = field(default=None, repr=False)
    _window: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        p = self.entry.physics
        self.n = self.entry.resolution
        if self.ref_n < self.n:
            raise ValueError(f"reference grid {self.ref_n} is coarser than the model's {self.n}")
        self.length, self.save_dt = p["length"], p["save_dt"]
        self.ref_grid = TorchSpectralGrid(self.ref_n, length=self.length, device=self.device)
        self.grid = TorchSpectralGrid(self.n, length=self.length)
        self._forcing_field = None
        if p.get("forcing_wavenumber"):
            self._forcing_field = KolmogorovForcing(
                p["forcing_wavenumber"], p.get("forcing_amplitude", 1.0)).vorticity(
                SpectralGrid(self.ref_n, length=self.length))
        self._build_solver()

    # ---------------------------------------------------------------- set-up
    def _build_solver(self) -> None:
        self.ref_solver = TorchVorticitySolver(
            self.ref_grid, self.nu, drag=self.entry.physics["drag"],
            forcing_vorticity=self._forcing_field if self.forcing else None)

    def reset(self, kind: str = "random", seed: int = 0, k_peak: float = 4.0) -> None:
        w0 = make_initial_field(self.ref_n, kind, seed, k_peak, length=self.length)
        self._ref_state = self.ref_solver.initial_state(w0)
        self.t, self.steps, self.diverged = 0.0, 0, False
        # predictors with a history window get the true history: advance the reference
        history = [self.ref_display()]
        for _ in range(self.stepper.n_in - 1):
            self._advance_ref()
            history.append(self.ref_display())
        self._window = torch.stack(history)[None]  # (1, n_in, n, n)

    def set_viscosity(self, nu: float) -> None:
        self.nu = float(nu)
        self._build_solver()

    def set_forcing(self, on: bool) -> None:
        self.forcing = bool(on)
        self._build_solver()
        self._sync_baseline_forcing()

    def _sync_baseline_forcing(self) -> None:
        # The coarse-solver baseline is physics, so it follows the toggle; networks cannot.
        if isinstance(self.stepper, CoarseSolverStepper):
            if not hasattr(self.stepper, "_forcing_on"):
                self.stepper._forcing_on = self.stepper.forcing
            self.stepper.forcing = self.stepper._forcing_on if self.forcing else None

    def set_stepper(self, entry: Entry, stepper: Stepper) -> None:
        if entry.resolution != self.n:
            raise ValueError("predictor uses a different grid; start a new session")
        self.entry, self.stepper = entry, stepper
        self._sync_baseline_forcing()
        self.resync()

    def resync(self) -> None:
        """Restart the predictor from the reference's current state."""
        frame = self.ref_display()
        self._window = frame[None, None].expand(1, self.stepper.n_in, -1, -1).clone()
        self.diverged = False

    # --------------------------------------------------------------- stepping
    def _advance_ref(self) -> None:
        dt_cfl = self.ref_solver.cfl_dt(self._ref_state, 0.5)
        n_sub = max(1, math.ceil(self.save_dt / min(dt_cfl, self.save_dt) - 1e-12))
        h = self.save_dt / n_sub
        with torch.no_grad():
            for _ in range(n_sub):
                self._ref_state = self.ref_solver.step(self._ref_state, h)
        self.t += self.save_dt

    def step(self) -> Metrics:
        self._advance_ref()
        nxt = self.stepper(self._window, torch.tensor([self.nu], dtype=torch.float64))
        nxt = nxt.to(torch.float64)
        if not bool(torch.isfinite(nxt).all()) or float(nxt.abs().max()) > 1e4:
            self.diverged = True
            nxt = torch.nan_to_num(nxt, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        self._window = torch.cat([self._window[:, 1:], nxt[:, None]], dim=1)
        self.steps += 1
        return self.metrics()

    def brush(self, x: float, y: float, radius: float, strength: float) -> None:
        """Add a Gaussian vortex at (x, y) in [0, 1)^2 to both the reference and the predictor."""
        r = radius * self.length
        blob_ref = torch.as_tensor(gaussian_vortex(self.ref_n, x, y, r, strength, self.length))
        w = self.ref_grid.ifft(self._ref_state[0]) + blob_ref.to(self.ref_grid.device)
        self._ref_state = self.ref_solver.initial_state(w)
        blob = torch.as_tensor(gaussian_vortex(self.n, x, y, r, strength, self.length))
        blob = blob - blob.mean()
        self._window = self._window.clone()
        self._window[0, -1] += blob

    # ------------------------------------------------------------------ output
    def ref_display(self) -> torch.Tensor:
        """Reference vorticity on the predictor's grid, shape (n, n)."""
        w = self.ref_grid.ifft(self._ref_state[0]).cpu().double()
        return spectral_resample(w, self.n) if self.ref_n != self.n else w

    def model_display(self) -> torch.Tensor:
        return self._window[0, -1].double()

    def metrics(self) -> Metrics:
        ref, mod = self.ref_display(), self.model_display()
        diff = (mod - ref).norm() / ref.norm().clamp_min(1e-12)
        a, b = ref - ref.mean(), mod - mod.mean()
        corr = (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)
        e_ref = float(self.grid.energy(self.grid.fft(ref)))
        e_mod = float(self.grid.energy(self.grid.fft(mod)))
        rr = float(self.ref_grid.resolution_ratio(self._ref_state)[0])
        return Metrics(self.t, float(diff), float(corr), e_ref, e_mod, self.diverged, rr)

    def spectra(self) -> dict[str, list[float]]:
        e_ref = self.grid.energy_spectrum(self.grid.fft(self.ref_display()))
        e_mod = self.grid.energy_spectrum(self.grid.fft(self.model_display()))
        return {"ref": e_ref.tolist(), "model": e_mod.tolist()}


def build_stepper(entry: Entry, device: str = "cpu") -> Stepper:
    if entry.kind == "model":
        from nfds.training.checkpoint import load_stepper

        return load_stepper(entry.path, device=device, name=entry.id)
    if entry.id == "persistence":
        return PersistenceStepper()
    if entry.id == "coarse_solver":
        p = entry.physics
        return CoarseSolverStepper(entry.resolution, p["save_dt"], p["drag"],
                                   p.get("forcing_wavenumber") or None,
                                   p.get("forcing_amplitude", 1.0), p["length"],
                                   device=device, dtype=torch.float32)
    raise KeyError(entry.id)
