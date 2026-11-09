"""Hybrid "solver-in-the-loop" model (Um et al., 2020; Kochkov et al., 2021).

One step:

    w_solver = CoarseSolver(w_t; nu)          the reference physics on the coarse grid
    w_{t+1}  = w_solver + Corrector(w_solver, w_t-n_in+1..t, nu)

The coarse solver is the differentiable PyTorch solver, so the corrector is trained end to
end *through* the physics, and with unrolled training it learns to correct the solver's
accumulated error rather than just one step. The corrector's output layer is zero-initialised,
so an untrained hybrid is exactly the coarse-solver baseline: training can only start from
physics and improve on it.

The model works in the same normalised units as every other model. It needs the physics
(drag, forcing, save interval) and the normalisation constants to convert to physical units
for the solver; the training loop fills these in from the dataset and stores them in the
checkpoint, so they are never set by hand.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nfds.models.cnn import ResidualCNN
from nfds.models.unet import UNet
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver

PHYSICS_KEYS = ("drag", "forcing_wavenumber", "forcing_amplitude", "save_dt", "length",
                "vorticity_mean", "vorticity_std", "nu_log_mean", "nu_log_std")


class HybridSolver(nn.Module):
    """Parameters
    ----------
    n_in : input frames (history fed to the corrector)
    corrector : "cnn" or "unet"
    corrector_kwargs : passed to the corrector's constructor
    solver_substeps : fixed solver steps per save interval (fixed so that gradients and
        cost are deterministic; choose so the CFL number stays well below 1)
    drag, forcing_wavenumber, forcing_amplitude, save_dt, length : physics of the dataset
    vorticity_mean, vorticity_std, nu_log_mean, nu_log_std : normalisation used in training
    """

    def __init__(self, n_in: int = 1, corrector: str = "cnn",
                 corrector_kwargs: dict | None = None, solver_substeps: int = 25,
                 drag: float = 0.1, forcing_wavenumber: int = 4, forcing_amplitude: float = 1.0,
                 save_dt: float = 0.25, length: float = 2 * math.pi,
                 vorticity_mean: float = 0.0, vorticity_std: float = 1.0,
                 nu_log_mean: float = math.log(3e-3), nu_log_std: float = 0.3) -> None:
        super().__init__()
        self.n_in = n_in
        self.solver_substeps = int(solver_substeps)
        self.drag, self.save_dt, self.length = float(drag), float(save_dt), float(length)
        self.forcing_wavenumber, self.forcing_amplitude = int(forcing_wavenumber), forcing_amplitude
        self.mean, self.std = float(vorticity_mean), float(vorticity_std)
        self.nu_log_mean, self.nu_log_std = float(nu_log_mean), float(nu_log_std)
        kw = dict(corrector_kwargs or {})
        # The corrector sees the input history plus the solver's prediction (last channel),
        # and predicts a residual on that last channel: zero output = pure solver.
        if corrector == "cnn":
            self.corrector = ResidualCNN(n_in=n_in + 1, **kw)
        elif corrector == "unet":
            self.corrector = UNet(n_in=n_in + 1, **kw)
        else:
            raise ValueError(f"unknown corrector {corrector!r}")
        self._grids: dict[tuple, tuple[TorchSpectralGrid, torch.Tensor | None]] = {}

    def _grid(self, n: int, device: torch.device):
        key = (n, str(device))
        if key not in self._grids:
            grid = TorchSpectralGrid(n, length=self.length, device=device, dtype=torch.float32)
            forcing = None
            if self.forcing_wavenumber:
                y = torch.arange(n, dtype=torch.float64) * (self.length / n)
                f = -self.forcing_amplitude * self.forcing_wavenumber * torch.cos(
                    self.forcing_wavenumber * y)
                forcing = f[:, None].expand(n, n).numpy()
            self._grids[key] = (grid, forcing)
        return self._grids[key]

    def solver_step(self, w: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Advance physical vorticity (B, n, n) by one save interval with the coarse solver."""
        grid, forcing = self._grid(w.shape[-1], w.device)
        with torch.autocast(device_type=w.device.type, enabled=False):
            solver = TorchVorticitySolver(grid, nu.float(), drag=self.drag,
                                          forcing_vorticity=forcing)
            w_hat = solver.initial_state(w.float())
            h = self.save_dt / self.solver_substeps
            for _ in range(self.solver_substeps):
                w_hat = solver.step(w_hat, h)
            return grid.ifft(w_hat)

    def forward(self, x: torch.Tensor, nu_feat: torch.Tensor) -> torch.Tensor:
        """x: (B, n_in, H, W) normalised; nu_feat: (B,) normalised log-viscosity."""
        nu = torch.exp(nu_feat.float() * self.nu_log_std + self.nu_log_mean)
        w = x[:, -1].float() * self.std + self.mean
        z_solver = (self.solver_step(w, nu) - self.mean) / self.std
        return self.corrector(torch.cat([x, z_solver[:, None].to(x.dtype)], dim=1), nu_feat)


def hybrid_physics_kwargs(data_root: str, split: str, stats: dict, nu_log_mean: float,
                          nu_log_std: float) -> dict:
    """Physics and normalisation for a HybridSolver, read from a generated dataset."""
    import json
    from pathlib import Path

    import h5py

    with h5py.File(Path(data_root) / f"{split}.h5") as f:
        attrs = dict(f.attrs)
    cfg = json.loads(attrs["config_json"])
    forcing = cfg["forcing"]
    return {
        "drag": float(attrs["drag"]), "save_dt": float(attrs["save_dt"]),
        "forcing_wavenumber": int(forcing["wavenumber"]) if forcing["type"] == "kolmogorov" else 0,
        "forcing_amplitude": float(forcing.get("amplitude", 1.0)),
        "length": float(cfg["domain_length"]),
        "vorticity_mean": float(stats["vorticity_mean"]),
        "vorticity_std": float(stats["vorticity_std"]),
        "nu_log_mean": float(nu_log_mean), "nu_log_std": float(nu_log_std),
    }


__all__ = ["PHYSICS_KEYS", "HybridSolver", "hybrid_physics_kwargs"]
