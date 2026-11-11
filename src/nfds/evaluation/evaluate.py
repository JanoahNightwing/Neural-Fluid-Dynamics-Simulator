"""Evaluate a checkpoint or a baseline on one split.

    python -m nfds.evaluation.evaluate --data data/forced_dev --split test_id \\
        --checkpoint runs/unet_dev/best.pt --horizon 40 --out results/forced_dev
    python -m nfds.evaluation.evaluate --data data/forced_dev --split test_id \\
        --baseline coarse_solver --horizon 40 --out results/forced_dev

Writes <out>/<name>__<split>.json with error/correlation/energy curves,
high-correlation times, spectra, spectral errors, PDE-residual and distribution metrics,
long-run statistics, failure statistics, timing and per-trajectory values.
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


def dataset_physics(data_root: str | Path, split: str, n: int) -> dict:
    """Drag, forcing (on an n² grid) and save interval used to generate a split."""
    with h5py.File(Path(data_root) / f"{split}.h5") as f:
        attrs = dict(f.attrs)
    cfg = json.loads(attrs["config_json"])
    forcing = None
    if cfg["forcing"]["type"] == "kolmogorov":
        from nfds.forcing import KolmogorovForcing
        from nfds.grid import SpectralGrid

        forcing = torch.as_tensor(KolmogorovForcing(
            cfg["forcing"]["wavenumber"], cfg["forcing"].get("amplitude", 1.0)).vorticity(
            SpectralGrid(n, length=cfg["domain_length"])))
    return {"drag": float(attrs["drag"]), "forcing": forcing, "save_dt": float(attrs["save_dt"])}


def evaluate_stepper(stepper: Stepper, data_root: str | Path, split: str, horizon: int,
                     resolution: int | None = None, max_trajectories: int | None = None,
                     batch_size: int = 64, corr_threshold: float = 0.8,
                     trajectories: tuple[int, int] | list[int] | None = None,
                     physics: bool = True) -> dict:
    """Roll a stepper out on a split and compute every metric of proposal section 7.

    Short-horizon accuracy (relative L2, correlation, high-correlation time), physical
    consistency (energy/enstrophy ratios, spectra, PDE residual relative to the truth's,
    vorticity distribution), long-run statistics over the second half of the horizon,
    failures, and timing. Per-trajectory scalars are included for confidence intervals.
    """
    data = load_split(data_root, split, resolution, trajectories=trajectories)
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
    late = slice(horizon // 2, horizon)  # second half: long-run statistics
    bands = M.default_bands(n)
    ok = fail >= horizon  # trajectories that never failed

    def ratio_of_means(p, t):
        if not ok.any():
            return None
        return M.finite_or_none(float(p[ok][:, late].mean() / t[ok][:, late].mean()))

    w_std = float(true.std())
    out = {
        "stepper": stepper.name, "split": split, "n_trajectories": n_traj, "n_in": n_in,
        "horizon": horizon, "save_dt": save_dt, "resolution": n,
        "trajectory_range": list(trajectories) if trajectories else None,
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
        "failure_fraction": float((~ok).double().mean()),
        "failure_step_median": float(fail.double().median()),
        "spectrum_k": list(range(spec_t.shape[-1])),
        "spectrum_pred_last": spec_p[last].mean(0).tolist(),
        "spectrum_true_last": spec_t[last].mean(0).tolist(),
        "spectral_log_error_last": M.spectral_error(spec_p[last].mean(0), spec_t[last].mean(0),
                                                    bands),
        # long-run statistics over the second half of the horizon (non-failed trajectories)
        "late_energy_ratio": ratio_of_means(e_pred, e_true),
        "late_enstrophy_ratio": ratio_of_means(z_pred, z_true),
        "late_vorticity_w1": M.finite_or_none(
            M.wasserstein_1d(pred[ok][:, late], true[ok][:, late]) / w_std) if ok.any() else None,
        # spectra over the same non-failed trajectories as the other long-run statistics
        "late_n_trajectories": int(ok.sum()),
        "late_spectrum_pred": (M.mean_spectrum(pred[ok][:, late], grid).mean(0).tolist()
                               if ok.any() else None),
        "late_spectrum_true": (M.mean_spectrum(true[ok][:, late], grid).mean(0).tolist()
                               if ok.any() else None),
        "seconds": seconds,
        "ms_per_step_per_trajectory": 1e3 * seconds / (horizon * n_traj),
        "per_trajectory": {
            "nu": nu.tolist(),
            "rel_l2_first_step": rel[:, 0].tolist(),
            "rel_l2_mean": torch.nan_to_num(rel, nan=float("inf")).mean(1).tolist(),
            "high_corr_time": hct.tolist(),
            "censored": censored.tolist(),
            "failure_step": fail.tolist(),
        },
    }
    if physics:
        phys = dataset_physics(data_root, split, n)
        seq_pred = torch.cat([w[:, n_in - 1 : n_in], pred], dim=1)
        seq_true = w[:, n_in - 1 : n_in + horizon]
        r_pred = M.pde_residual(seq_pred, nu, save_dt, grid, phys["drag"], phys["forcing"])
        r_true = M.pde_residual(seq_true, nu, save_dt, grid, phys["drag"], phys["forcing"])
        ratio = r_pred.mean(1) / r_true.mean(1)
        out["pde_residual_ratio"] = M.finite_or_none(float(M.nanmean(ratio)))
        out["per_trajectory"]["pde_residual_ratio"] = ratio.tolist()
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a model or baseline on one split")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test_id")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint")
    g.add_argument("--baseline", choices=["persistence", "coarse_solver"])
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--resolution", type=int, help="stored grid to evaluate on (default: the "
                   "training resolution); e.g. 128 for a zero-shot resolution-transfer test")
    p.add_argument("--max-trajectories", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", help="label for the results file (default: run or baseline name)")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    stats_path = Path(args.data) / "stats.json"
    train_res = json.loads(stats_path.read_text(encoding="utf-8"))["resolution"]
    n = args.resolution or train_res
    if args.checkpoint:
        from nfds.training.checkpoint import load_stepper

        stepper = load_stepper(args.checkpoint, device=args.device)
    else:
        stepper = make_baseline(args.baseline, args.data, args.split, n, args.device)
    name = args.name or stepper.name
    res = evaluate_stepper(stepper, args.data, args.split, args.horizon, resolution=n,
                           max_trajectories=args.max_trajectories)
    res["name"] = name
    res["train_resolution"] = train_res
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.split if n == train_res else f"{args.split}@{n}"
    path = out / f"{name}__{label}.json"
    path.write_text(json.dumps(M.jsonable(res), indent=1, allow_nan=False), encoding="utf-8")
    print(f"{name} on {label}: first-step rel. L2 {res['rel_l2_first_step']:.4f}, "
          f"mean over {res['horizon']} steps {res['rel_l2_mean_over_horizon']}, "
          f"high-corr time {res['high_corr_time_mean']:.2f} "
          f"({100 * res['high_corr_censored_fraction']:.0f}% censored), "
          f"failures {100 * res['failure_fraction']:.0f}%  -> {path}")


if __name__ == "__main__":
    main()
