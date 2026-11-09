"""Produce the solver verification figure: temporal convergence and Taylor-Green decay.

Usage:  python scripts/verify.py --out figures/verification.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field, taylor_green, taylor_green_exact
from nfds.solver import SpectralVorticitySolver


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def temporal_convergence(ax: plt.Axes) -> None:
    g = SpectralGrid(32)
    w0 = random_field(g, k_peak=3, energy=0.5, seed=7)
    t_final, nu = 1.0, 1e-2
    ref = SpectralVorticitySolver(g, nu, scheme="ifrk4").simulate(
        w0, t_final, t_final, dt=1 / 4000).vorticity[-1]
    dts = 1 / np.array([10, 20, 40, 80, 160])
    for scheme, order, marker in [("ifrk2", 2, "o"), ("ifrk4", 4, "s")]:
        errs = [rel_l2(SpectralVorticitySolver(g, nu, scheme=scheme)
                       .simulate(w0, t_final, t_final, dt=dt).vorticity[-1], ref) for dt in dts]
        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        ax.loglog(dts, errs, marker + "-", label=f"{scheme}: slope {slope:.2f}")
        ax.loglog(dts, errs[0] * (dts / dts[0]) ** order, "k:", lw=0.8)
    ax.set_xlabel("time step dt")
    ax.set_ylabel("relative L2 error at t = 1")
    ax.set_title("Temporal convergence (dotted: ideal order)")
    ax.legend()


def taylor_green_panel(ax: plt.Axes) -> None:
    g = SpectralGrid(32)
    nu = 0.05
    res = SpectralVorticitySolver(g, nu).simulate(taylor_green(g), 5.0, 0.25, dt=0.05)
    errs = [rel_l2(w, taylor_green_exact(g, t, nu)) for t, w in zip(res.times, res.vorticity,
                                                                     strict=True)]
    amp = [np.max(np.abs(w)) for w in res.vorticity]
    ax.semilogy(res.times, amp, "o", ms=3, label="solver: max |omega|")
    exact = 2 * np.exp(-2 * nu * res.times)
    ax.semilogy(res.times, exact, "k-", lw=1, label="exact: 2 exp(-2 nu t)")
    ax.set_xlabel("time")
    ax.set_title(f"Taylor-Green decay (max rel. error {max(errs):.1e})")
    ax.legend()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/verification.png")
    args = p.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    temporal_convergence(axes[0])
    taylor_green_panel(axes[1])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
