"""Throughput of the NumPy and PyTorch solvers: simulated steps per second per trajectory.

Usage:
    python scripts/benchmark_solver.py                      # CPU, plus CUDA if available
    python scripts/benchmark_solver.py --sizes 64 128 256 --batches 1 16 64 --out bench.json

Timing protocol: warm-up steps first, torch.cuda.synchronize() around every
timed region, median of several repeats, state already on the device.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field
from nfds.solver import SpectralVorticitySolver


def time_fn(fn, sync=lambda: None, warmup: int = 3, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_numpy(n: int, steps: int) -> float:
    g = SpectralGrid(n)
    s = SpectralVorticitySolver(g, 1e-3)
    state = {"w": g.truncate(g.fft(random_field(g, seed=0)))}

    def run():
        for _ in range(steps):
            state["w"] = s.step(state["w"], 1e-3)

    return time_fn(run, warmup=1, repeats=3) / steps  # seconds per step


def bench_torch(n: int, batch: int, steps: int, device: str, dtype_name: str) -> float:
    import torch

    from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver

    dtype = getattr(torch, dtype_name)
    g = TorchSpectralGrid(n, device=device, dtype=dtype)
    w0 = np.stack([random_field(SpectralGrid(n), seed=i) for i in range(batch)])
    s = TorchVorticitySolver(g, np.full(batch, 1e-3))
    state = {"w": s.initial_state(w0)}
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    @torch.no_grad()
    def run():
        for _ in range(steps):
            state["w"] = s.step(state["w"], 1e-3)

    return time_fn(run, sync=sync) / steps  # seconds per batched step


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    rows = []
    try:
        import torch

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        torch_info = {"torch": torch.__version__, "threads": torch.get_num_threads(),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        devices, torch_info = [], {}

    for n in args.sizes:
        t = bench_numpy(n, args.steps)
        rows.append({"solver": "numpy", "device": "cpu", "dtype": "float64", "n": n, "batch": 1,
                     "ms_per_step": 1e3 * t, "traj_steps_per_s": 1 / t})
        for device in devices:
            for dtype in ("float64", "float32"):
                for b in args.batches:
                    t = bench_torch(n, b, args.steps, device, dtype)
                    rows.append({"solver": "torch", "device": device, "dtype": dtype, "n": n,
                                 "batch": b, "ms_per_step": 1e3 * t, "traj_steps_per_s": b / t})

    base = {r["n"]: r["traj_steps_per_s"] for r in rows if r["solver"] == "numpy"}
    print(f"{'solver':6} {'device':6} {'dtype':7} {'n':>4} {'batch':>5} {'ms/step':>9} "
          f"{'traj-steps/s':>13} {'vs numpy':>9}")
    for r in rows:
        r["speedup_vs_numpy"] = r["traj_steps_per_s"] / base[r["n"]]
        print(f"{r['solver']:6} {r['device']:6} {r['dtype']:7} {r['n']:4d} {r['batch']:5d} "
              f"{r['ms_per_step']:9.2f} {r['traj_steps_per_s']:13.1f} "
              f"{r['speedup_vs_numpy']:8.1f}x")

    if args.out:
        meta = {"platform": platform.platform(), "processor": platform.processor(),
                "numpy": np.__version__, **torch_info}
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
