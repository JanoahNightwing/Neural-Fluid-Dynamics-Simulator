"""Steppers: anything that advances physical vorticity by one saved frame.

A stepper takes the last ``n_in`` frames, shape (B, n_in, H, W), in physical
units, plus the viscosity of each sample, and returns the next frame (B, H, W).
Neural models, the persistence baseline and the coarse solver all share this
interface, so the same rollout and metrics code evaluates all of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from nfds.forcing import KolmogorovForcing
from nfds.grid import SpectralGrid
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver


@dataclass
class NuNormalizer:
    """Maps viscosity to the network's conditioning feature (standardised log nu)."""

    log_mean: float
    log_std: float

    @classmethod
    def from_values(cls, nu: np.ndarray) -> NuNormalizer:
        logs = np.log(np.asarray(nu, dtype=np.float64))
        return cls(float(logs.mean()), float(max(logs.std(), 1e-8)))

    def __call__(self, nu: torch.Tensor) -> torch.Tensor:
        return (torch.log(nu) - self.log_mean) / self.log_std


class Stepper:
    name = "stepper"
    n_in = 1

    def __call__(self, frames: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PersistenceStepper(Stepper):
    """Predicts that nothing changes: the floor every model must beat."""

    name = "persistence"

    def __call__(self, frames: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        return frames[:, -1].clone()


class NeuralStepper(Stepper):
    """Wraps a trained model with its normalisation of vorticity and viscosity."""

    def __init__(self, model: torch.nn.Module, mean: float, std: float,
                 nu_norm: NuNormalizer, n_in: int, name: str = "model",
                 device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.mean, self.std, self.nu_norm = mean, std, nu_norm
        self.n_in, self.name, self.device = n_in, name, torch.device(device)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        x = ((frames.to(self.device, torch.float32) - self.mean) / self.std)
        out = self.model(x, self.nu_norm(nu.to(self.device, torch.float32)))
        return (out[:, 0] * self.std + self.mean).to(frames.device, frames.dtype)


class CoarseSolverStepper(Stepper):
    """The reference solver run directly on the coarse (training) grid.

    This is the honest baseline: exact physics at the same resolution and a
    comparable cost to the network. A surrogate is only interesting if it is
    more accurate than this, or as accurate and much faster.
    """

    name = "coarse_solver"

    def __init__(self, n: int, save_dt: float, drag: float, forcing_wavenumber: int | None,
                 forcing_amplitude: float = 1.0, length: float = 2 * math.pi,
                 cfl: float = 0.5, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64) -> None:
        self.n, self.save_dt, self.drag, self.cfl = n, save_dt, drag, cfl
        self.grid = TorchSpectralGrid(n, length=length, device=device, dtype=dtype)
        self.forcing = None
        if forcing_wavenumber:
            self.forcing = KolmogorovForcing(forcing_wavenumber, forcing_amplitude).vorticity(
                SpectralGrid(n, length=length))
        self.device, self.dtype = torch.device(device), dtype

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        solver = TorchVorticitySolver(self.grid, nu.to(self.device, self.dtype),
                                      drag=self.drag, forcing_vorticity=self.forcing)
        w_hat = solver.initial_state(frames[:, -1].to(self.device, self.dtype))
        n_sub = max(1, math.ceil(self.save_dt / solver.cfl_dt(w_hat, self.cfl) - 1e-12))
        h = self.save_dt / n_sub
        for _ in range(n_sub):
            w_hat = solver.step(w_hat, h)
        return self.grid.ifft(w_hat).to(frames.device, frames.dtype)

    @classmethod
    def from_dataset_attrs(cls, attrs: dict, n: int, **kwargs) -> CoarseSolverStepper:
        import json

        cfg = json.loads(attrs["config_json"])
        f = cfg["forcing"]
        return cls(n=n, save_dt=float(attrs["save_dt"]), drag=float(attrs["drag"]),
                   forcing_wavenumber=f["wavenumber"] if f["type"] == "kolmogorov" else None,
                   forcing_amplitude=f.get("amplitude", 1.0),
                   length=cfg["domain_length"], **kwargs)
