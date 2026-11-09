"""Inference speed and the accuracy-vs-cost Pareto front (proposal section 7.4).

    python -m nfds.evaluation.speed --data data/forced_v1 --split test_id \\
        --checkpoints runs/m5/*/best.pt --solver-resolutions 32 48 64 96 128 256 \\
        --horizon 10 --batch 16 --out results/m5/pareto.json --figure figures/pareto.png

Every model and the reference solver at several grid resolutions are scored on the same
trajectories: error at a fixed horizon (on the training grid) against wall-clock cost per
simulated time unit per trajectory. A learned model is only useful if it lies below the
solver's curve. Timing: warm-up, CUDA synchronisation, median of repeats, data already on
the device, nothing else running.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import h5py
import torch

from nfds.data.dataset import load_split, load_stats
from nfds.data.transforms import spectral_resample
from nfds.evaluation import metrics as M
from nfds.evaluation.rollout import rollout
from nfds.evaluation.steppers import CoarseSolverStepper, NeuralStepper, Stepper


class FastNeuralStepper(Stepper):
    """A NeuralStepper with optional torch.compile and bf16 autocast."""

    def __init__(self, base: NeuralStepper, compile: bool = False, bf16: bool = False) -> None:
        self.base, self.bf16 = base, bf16
        self.n_in = base.n_in
        self.name = base.name + ("+compile" if compile else "") + ("+bf16" if bf16 else "")
        self.model = torch.compile(base.model) if compile else base.model

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        b = self.base
        x = (frames.to(b.device, torch.float32) - b.mean) / b.std
        with torch.autocast(device_type=b.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            out = self.model(x, b.nu_norm(nu.to(b.device, torch.float32)))
        return (out[:, 0].float() * b.std + b.mean).to(frames.device, frames.dtype)


def time_stepper(stepper: Stepper, frames: torch.Tensor, nu: torch.Tensor, steps: int = 5,
                 warmup: int = 2, repeats: int = 5) -> float:
    """Median wall time per step per trajectory (seconds), batch = frames.shape[0]."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def run():
        window = frames
        for _ in range(steps):
            nxt = stepper(window, nu)
            window = torch.cat([window[:, 1:], nxt[:, None]], dim=1)

    for _ in range(warmup):
        run()
    sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) / (steps * frames.shape[0])


def pareto(data_root: str | Path, split: str, checkpoints: list[str],
           solver_resolutions: list[int], horizon: int = 10, batch: int = 16,
           max_trajectories: int = 32, device: str = "cpu", compile: bool = False,
           bf16: bool = False) -> dict:
    from nfds.training.checkpoint import load_stepper

    n_train = load_stats(data_root)["resolution"]
    with h5py.File(Path(data_root) / f"{split}.h5") as f:
        stored = sorted(int(k.split("_")[1]) for k in f if k.startswith("vorticity_"))
        attrs = dict(f.attrs)
    truth = load_split(data_root, split, n_train)
    w_train = torch.from_numpy(truth["vorticity"][:max_trajectories]).double()
    nu = torch.from_numpy(truth["nu"][:max_trajectories]).double()
    save_dt = float(attrs["save_dt"])

    def score(pred: torch.Tensor, n_in: int) -> dict:
        tgt = w_train[:, n_in : n_in + horizon]
        rel = M.relative_l2(pred, tgt)
        return {"rel_l2_at_horizon": M.finite_or_none(float(M.nanmean(rel[:, -1]))),
                "rel_l2_mean": M.finite_or_none(float(M.nanmean(rel.flatten())))}

    points = []
    src_n = max(stored)
    w_src = torch.from_numpy(load_split(data_root, split, src_n)["vorticity"][:max_trajectories])
    for n in solver_resolutions:
        if n > src_n:
            print(f"skipping solver at {n}²: largest stored grid is {src_n}²")
            continue
        stepper = CoarseSolverStepper.from_dataset_attrs(attrs, n=n, device=device,
                                                         dtype=torch.float32)
        w_n = spectral_resample(w_src.double(), n)[:, :1].float().to(device)
        pred = rollout(stepper, w_n.cpu().double(), nu, horizon).double()
        pred = spectral_resample(pred, n_train)
        t = time_stepper(stepper, w_n[:batch], nu[:batch].float().to(device))
        points.append({"name": f"solver {n}²", "kind": "solver", "resolution": n,
                       "ms_per_step": 1e3 * t, "ms_per_time_unit": 1e3 * t / save_dt,
                       **score(pred, 1)})
        print(f"solver {n:4d}²: {points[-1]}")

    for ck in checkpoints:
        base = load_stepper(ck, device=device)
        stepper = FastNeuralStepper(base, compile=compile, bf16=bf16)
        pred = rollout(stepper, w_train, nu, horizon).double()
        frames = w_train[:batch, : base.n_in].float().to(device)
        t = time_stepper(stepper, frames, nu[:batch].float().to(device))
        points.append({"name": stepper.name, "kind": "model", "checkpoint": str(ck),
                       "resolution": n_train, "ms_per_step": 1e3 * t,
                       "ms_per_time_unit": 1e3 * t / save_dt, **score(pred, base.n_in)})
        print(f"{stepper.name}: {points[-1]}")
    return {"split": split, "horizon": horizon, "horizon_time": horizon * save_dt,
            "batch": batch, "device": device, "n_trajectories": int(w_train.shape[0]),
            "train_resolution": n_train, "compile": compile, "bf16": bf16, "points": points}


def plot_pareto(res: dict, out: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    sol = sorted((p for p in res["points"] if p["kind"] == "solver"),
                 key=lambda p: p["ms_per_time_unit"])
    ax.plot([p["ms_per_time_unit"] for p in sol], [p["rel_l2_at_horizon"] for p in sol], "k--o",
            label="reference solver at different grids")
    for p in sol:
        ax.annotate(f"{p['resolution']}²", (p["ms_per_time_unit"], p["rel_l2_at_horizon"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    for p in res["points"]:
        if p["kind"] == "model" and p["rel_l2_at_horizon"] is not None:
            ax.scatter(p["ms_per_time_unit"], p["rel_l2_at_horizon"], s=50, zorder=3,
                       label=p["name"])
    ax.set_xscale("log")
    ax.set_xlabel(f"wall time per simulated time unit per trajectory (ms, batch {res['batch']}, "
                  f"{res['device']})")
    ax.set_ylabel(f"relative L2 error at t = {res['horizon_time']:g}")
    ax.set_title("Accuracy vs cost: useful models lie below the solver curve")
    ax.legend(fontsize=8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Accuracy-vs-cost benchmark")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test_id")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--solver-resolutions", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-trajectories", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--figure")
    args = p.parse_args(argv)
    res = pareto(args.data, args.split, args.checkpoints, args.solver_resolutions,
                 args.horizon, args.batch, args.max_trajectories, args.device, args.compile,
                 args.bf16)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(M.jsonable(res), indent=1, allow_nan=False))
    if args.figure:
        plot_pareto(res, args.figure)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
