"""Evaluate a checkpoint or a baseline on one split.

    python -m nfds.evaluation.evaluate --data data/forced_dev --split test_id \\
        --checkpoint runs/unet_dev/best.pt --horizon 40 --out results/forced_dev
    python -m nfds.evaluation.evaluate --data data/forced_dev --split test_id \\
        --baseline coarse_solver --horizon 40 --out results/forced_dev

Writes <out>/<name>__<split>.json with error/correlation/energy curves,
high-correlation times, spectra, spectral errors, failure statistics and timing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from nfds.data.dataset import load_split
from nfds.evaluation import metrics as M
from nfds.evaluation.rollout import rollout
from nfds.evaluation.steppers import CoarseSolverStepper, PersistenceStepper, Stepper
from nfds.solver.spectral_torch import TorchSpectralGrid


def make_baseline(name: str, data_root: str | Path, split: str, n: int,
                  device: str = "cpu", dtype: torch.dtype = torch.float32) -> Stepper:
    """Baselines run in float32 by default, like the networks, so timings are comparable."""
    if name == "persistence":
        return PersistenceStepper()
    if name == "coarse_solver":
        with h5py.File(Path(data_root) / f"{split}.h5") as f:
            attrs = dict(f.attrs)
        return CoarseSolverStepper.from_dataset_attrs(attrs, n=n, device=device, dtype=dtype)
    raise KeyError(f"unknown baseline {name!r}")


def evaluate_stepper(stepper: Stepper, data_root: str | Path, split: str, horizon: int,
                     resolution: int | None = None, max_trajectories: int | None = None,
                     batch_size: int = 64, corr_threshold: float = 0.8) -> dict:
    data = load_split(data_root, split, resolution)
    w = torch.from_numpy(data["vorticity"]).double()
    nu = torch.from_numpy(data["nu"]).double()
    if max_trajectories:
        w, nu = w[:max_trajectories], nu[:max_trajectories]
    n_traj, n_t, n, _ = w.shape
    n_in = stepper.n_in
    horizon = min(horizon, n_t - n_in)
    save_dt = float(data["times"][1] - data["times"][0])

    t0 = time.perf_counter()
    pred = rollout(stepper, w, nu, horizon, batch_size=batch_size).double()
    seconds = time.perf_counter() - t0
    true = w[:, n_in : n_in + horizon]

    grid = TorchSpectralGrid(n)
    rel = M.relative_l2(pred, true)
    corr = M.correlation(pred, true)
    hct, censored = M.high_correlation_time(corr, save_dt, corr_threshold)
    e_pred, e_true = M.field_energy(pred, grid), M.field_energy(true, grid)
    z_pred, z_true = M.field_enstrophy(pred), M.field_enstrophy(true)
    fail = M.failure_step(pred, e_true.mean(dim=1), grid)
    spec_p, spec_t = M.mean_spectrum(pred, grid), M.mean_spectrum(true, grid)
    last = slice(max(0, horizon - max(1, horizon // 4)), horizon)  # last quarter of horizon
    bands = M.default_bands(n)

    return {
        "stepper": stepper.name, "split": split, "n_trajectories": n_traj, "n_in": n_in,
        "horizon": horizon, "save_dt": save_dt, "resolution": n,
        "times": (np.arange(1, horizon + 1) * save_dt).tolist(),
        "rel_l2": M.summarise_curve(rel),
        "correlation": M.summarise_curve(corr),
        "energy_ratio": M.summarise_curve(e_pred / e_true),
        "enstrophy_ratio": M.summarise_curve(z_pred / z_true),
        "rel_l2_first_step": M.finite_or_none(float(M.nanmean(rel[:, 0]))),
        "rel_l2_mean_over_horizon": M.finite_or_none(float(M.nanmean(rel.flatten()))),
        "high_corr_threshold": corr_threshold,
        "high_corr_time_mean": float(hct.mean()),
        "high_corr_time_median": float(hct.median()),
        "high_corr_censored_fraction": float(censored.double().mean()),
        "failure_fraction": float((fail < horizon).double().mean()),
        "failure_step_median": float(fail.double().median()),
        "spectrum_k": list(range(spec_t.shape[-1])),
        "spectrum_pred_last": spec_p[last].mean(0).tolist(),
        "spectrum_true_last": spec_t[last].mean(0).tolist(),
        "spectral_log_error_last": M.spectral_error(spec_p[last].mean(0), spec_t[last].mean(0),
                                                    bands),
        "seconds": seconds,
        "ms_per_step_per_trajectory": 1e3 * seconds / (horizon * n_traj),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a model or baseline on one split")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test_id")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint")
    g.add_argument("--baseline", choices=["persistence", "coarse_solver"])
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--max-trajectories", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", help="label for the results file (default: run or baseline name)")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    if args.checkpoint:
        from nfds.training.checkpoint import load_stepper

        stepper = load_stepper(args.checkpoint, device=args.device)
    else:
        n = json.loads((Path(args.data) / "stats.json").read_text())["resolution"]
        stepper = make_baseline(args.baseline, args.data, args.split, n, args.device)
    name = args.name or stepper.name
    res = evaluate_stepper(stepper, args.data, args.split, args.horizon,
                           max_trajectories=args.max_trajectories)
    res["name"] = name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}__{args.split}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"{name} on {args.split}: first-step rel. L2 {res['rel_l2_first_step']:.4f}, "
          f"mean over {res['horizon']} steps {res['rel_l2_mean_over_horizon']}, "
          f"high-corr time {res['high_corr_time_mean']:.2f} "
          f"({100 * res['high_corr_censored_fraction']:.0f}% censored), "
          f"failures {100 * res['failure_fraction']:.0f}%  -> {path}")


if __name__ == "__main__":
    main()
