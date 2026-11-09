"""Forced (Kolmogorov) 2D turbulence for several viscosities in one batched run.

Shows spin-up to a statistically stationary state and checks resolution.

Usage:
    python scripts/demo_forced.py --n 128 --nus 1e-3 2e-3 5e-3 --t-final 40
    python scripts/demo_forced.py --device cuda --n 256 --dtype float32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nfds.forcing import KolmogorovForcing
from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--nus", type=float, nargs="+", default=[1e-3, 2e-3, 5e-3])
    p.add_argument("--drag", type=float, default=0.1)
    p.add_argument("--t-final", type=float, default=40.0)
    p.add_argument("--save-dt", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--out", default="figures/forced.png")
    args = p.parse_args()

    g_np = SpectralGrid(args.n)
    g = TorchSpectralGrid(args.n, device=args.device, dtype=getattr(torch, args.dtype))
    forcing = KolmogorovForcing(wavenumber=4)
    b = len(args.nus)
    w0 = np.stack([random_field(g_np, k_peak=4, energy=0.5, seed=i) for i in range(b)])
    solver = TorchVorticitySolver(g, args.nus, drag=args.drag,
                                  forcing_vorticity=forcing.vorticity(g_np))

    t0 = time.perf_counter()
    res = solver.simulate(w0, args.t_final, args.save_dt, cfl=0.5)
    wall = time.perf_counter() - t0
    print(f"{b} trajectories, {res.n_steps} steps in {wall:.1f} s on {args.device}")

    w = res.vorticity.to(g.device, g.dtype)  # (B, T, n, n)
    w_hat = g.fft(w)
    e = g.energy(w_hat).cpu().numpy()  # (B, T)
    rr = g.resolution_ratio(w_hat).cpu().numpy()
    half = res.times.numpy() >= args.t_final / 2
    for i, nu in enumerate(args.nus):
        print(f"nu = {nu:g}: stationary energy {e[i, half].mean():.3f} ± {e[i, half].std():.3f}, "
              f"max resolution ratio {rr[i].max():.1e}"
              + ("  (under-resolved!)" if rr[i].max() > 1e-4 else ""))

    fig, axes = plt.subplots(2, b, figsize=(4 * b, 7.5), constrained_layout=True,
                             squeeze=False)
    for i, nu in enumerate(args.nus):
        final = res.vorticity[i, -1].float().numpy()
        vmax = np.percentile(np.abs(final), 99.5)
        ax = axes[0, i]
        ax.imshow(final, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=(0, 2 * np.pi, 0, 2 * np.pi))
        ax.set_title(f"nu = {nu:g}, t = {args.t_final:g}  (res. ratio {rr[i].max():.0e})",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, i]
        ax.plot(res.times.numpy(), e[i])
        ax.axvspan(args.t_final / 2, args.t_final, color="0.9", zorder=0)
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        ax.set_title("energy (shaded: window for statistics)", fontsize=10)
    fig.suptitle(f"Kolmogorov-forced turbulence, {args.n}^2, k_f = 4, drag = {args.drag}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
