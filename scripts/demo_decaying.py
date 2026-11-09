"""Decaying 2D turbulence: snapshots, energy spectrum and energy budget.

Usage:  python scripts/demo_decaying.py --n 256 --nu 1e-3 --t-final 20 --out figures/decaying.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nfds.diagnostics import energy, energy_spectrum, enstrophy, resolution_ratio
from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field
from nfds.solver import SpectralVorticitySolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--k-peak", type=float, default=6.0)
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--save-dt", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/decaying.png")
    args = p.parse_args()

    g = SpectralGrid(args.n)
    solver = SpectralVorticitySolver(g, nu=args.nu)
    w0 = random_field(g, k_peak=args.k_peak, energy=0.5, seed=args.seed)
    t0 = time.perf_counter()
    res = solver.simulate(w0, args.t_final, args.save_dt, cfl=0.5)
    wall = time.perf_counter() - t0
    print(f"{res.n_steps} steps in {wall:.1f} s ({1e3 * wall / res.n_steps:.1f} ms/step)")

    w_hats = [g.fft(w) for w in res.vorticity]
    e = np.array([energy(g, w) for w in w_hats])
    z = np.array([enstrophy(g, w) for w in w_hats])
    de = np.gradient(e, res.times)
    budget_resid = np.abs(de + 2 * args.nu * z)[1:-1].max() / np.abs(2 * args.nu * z).max()
    res_ratio = max(resolution_ratio(g, w) for w in w_hats)
    print(f"max resolution ratio E(k near cutoff)/E(peak): {res_ratio:.1e}"
          + ("  (under-resolved!)" if res_ratio > 1e-4 else ""))
    print(f"energy budget residual (finite-difference dE/dt vs -2 nu Z): {budget_resid:.2e}")

    fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    picks = [0, len(res.times) // 8, len(res.times) // 3, len(res.times) - 1]
    vmax = np.percentile(np.abs(res.vorticity[picks[1]]), 99.5)
    for j, i in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(res.vorticity[i], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=(0, 2 * np.pi, 0, 2 * np.pi))
        ax.set_title(f"vorticity, t = {res.times[i]:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :2])
    for i in picks[1:]:
        k, ek = energy_spectrum(g, w_hats[i])
        ax.loglog(k[1:], ek[1:], label=f"t = {res.times[i]:.1f}")
    kref = np.array([10.0, 40.0])
    ax.loglog(kref, 2e-2 * (kref / kref[0]) ** -3, "k--", lw=1, label="k^-3 (enstrophy cascade)")
    ax.axvline(g.n // 3, color="grey", lw=0.8, ls=":")
    ax.text(g.n // 3, ax.get_ylim()[1] * 0.3, " dealias cut", color="grey", fontsize=8)
    ax.set_ylim(bottom=1e-14)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Energy spectrum")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(res.times, e / e[0], label="E / E0")
    ax.plot(res.times, z / z[0], label="Z / Z0")
    ax.set_xlabel("time")
    ax.set_title("Energy decays slowly, enstrophy fast "
                 f"(budget residual {budget_resid:.1e})")
    ax.title.set_fontsize(10)
    ax.legend()

    fig.suptitle(f"Decaying 2D turbulence, {args.n}^2, nu = {args.nu:g}, IF-RK4 "
                 f"(resolution ratio {res_ratio:.0e})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
