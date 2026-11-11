"""Generate a dataset from a YAML config.

    python -m nfds.data.generate configs/datasets/forced_v1.yaml --out data/forced_v1 --device cuda

Output directory layout::

    <out>/
      train.h5, val.h5, test_*.h5   one file per split
      stats.json                    normalisation statistics (train split only)
      manifest.json                 sha256, sizes, counts, config fingerprint
      config.yaml                   the exact config used
      DATASHEET.md                  human-readable description

Each split file contains, for N trajectories and T saved frames:

    vorticity_<n>     (N, T, n, n) float32, one dataset per stored resolution
    times             (T,)  time since the end of spin-up
    nu, k_peak, resolution_ratio, max_cfl, energy_mean   (N,) float64
    accepted          (N,) bool    passed the quality checks
    traj_index        (N,) int64   index used to derive the trajectory's seed
    ic_kind           (N,) str
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

import nfds
from nfds.data.config import DatasetConfig, SplitConfig, load_config
from nfds.data.transforms import spectral_downsample
from nfds.diagnostics import energy as energy_np
from nfds.forcing import KolmogorovForcing
from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field, shear_layer, taylor_green, vortex_dipole
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver

# --------------------------------------------------------------------- seeding


def trajectory_rng(base_seed: int, split: str, index: int) -> np.random.Generator:
    """Independent stream per (dataset seed, split name, trajectory index).

    Independent of batch size, device and the order in which splits are
    generated, so any trajectory can be regenerated on its own.
    """
    ss = np.random.SeedSequence([base_seed, zlib.crc32(split.encode()), index])
    return np.random.default_rng(ss)


def sample_parameters(cfg: DatasetConfig, split: SplitConfig, index: int) -> dict[str, Any]:
    rng = trajectory_rng(cfg.base_seed, split.name, index)
    nu_cfg = split.nu
    if nu_cfg.dist == "loguniform":
        # Sample log(nu) uniformly over [low, high] minus the excluded band, by
        # mapping one uniform draw onto the remaining length (no rejection loop).
        a, b = np.log(nu_cfg.low), np.log(nu_cfg.high)
        if nu_cfg.exclude is None:
            nu = float(np.exp(rng.uniform(a, b)))
        else:
            lo, hi = np.log(nu_cfg.exclude[0]), np.log(nu_cfg.exclude[1])
            x = rng.uniform(0.0, (lo - a) + (b - hi))
            nu = float(np.exp(a + x if x < lo - a else hi + (x - (lo - a))))
    elif nu_cfg.dist == "choice":
        nu = float(nu_cfg.values[index % len(nu_cfg.values)])  # balanced across values
    else:  # fixed
        nu = float(nu_cfg.values[0])
    ic = split.ic
    kind = "random" if ic.type == "random" else ic.kinds[index % len(ic.kinds)]
    return {
        "index": index,
        "nu": nu,
        "ic_kind": kind,
        "k_peak": float(rng.uniform(*ic.k_peak)) if len(ic.k_peak) == 2 else float(ic.k_peak[0]),
        "ic_seed": int(rng.integers(2**62)),
        "ic_u": rng.uniform(size=4).tolist(),  # spare uniforms for structured ICs
        "energy": ic.energy,
    }


def make_initial_condition(grid: SpectralGrid, p: dict[str, Any]) -> np.ndarray:
    kind, u = p["ic_kind"], p["ic_u"]
    if kind == "random":
        return random_field(grid, k_peak=p["k_peak"], energy=p["energy"], seed=p["ic_seed"])
    if kind == "shear_layer":
        w = shear_layer(grid, thickness=0.02 + 0.02 * u[0], perturbation=0.05,
                        mode=1 + int(3 * u[1]))
    elif kind == "dipole":
        w = vortex_dipole(grid, separation=0.5 + 0.5 * u[0], radius=0.2 + 0.15 * u[1],
                          strength=10.0)
    elif kind == "taylor_green_perturbed":
        k = (1, 2, 4)[int(3 * u[0])]
        # Perturbation peaks at k = 5 so its spectral tail stays far below the
        # dealiasing cutoff even on a 128² grid (k = 8 tripped the quality check).
        w = taylor_green(grid, k=k) + random_field(grid, k_peak=5, energy=0.05,
                                                   seed=p["ic_seed"])
    else:
        raise ValueError(f"unknown initial condition kind {kind!r}")
    w_hat = grid.truncate(grid.fft(w))
    w_hat[0, 0] = 0.0
    w_hat *= math.sqrt(p["energy"] / energy_np(grid, w_hat))
    return grid.ifft(w_hat)


# ------------------------------------------------------------------ simulation


class CFLViolation(RuntimeError):
    pass


def simulate_batch(
    cfg: DatasetConfig, split: SplitConfig, params: list[dict[str, Any]], device: str
) -> dict[str, Any]:
    """Run one batch of trajectories and return downsampled frames + diagnostics."""
    sim, r = cfg.simulation, cfg.resolved(split)
    n, dt = r["sim_n"], r["dt"]
    dtype = getattr(torch, sim.dtype)
    g_np = SpectralGrid(n, length=cfg.domain_length)
    g = TorchSpectralGrid(n, length=cfg.domain_length, device=device, dtype=dtype)
    forcing = None
    if cfg.forcing.type == "kolmogorov":
        forcing = KolmogorovForcing(cfg.forcing.wavenumber, cfg.forcing.amplitude).vorticity(g_np)
    elif cfg.forcing.type != "none":
        raise ValueError(f"unknown forcing {cfg.forcing.type!r}")

    w0 = np.stack([make_initial_condition(g_np, p) for p in params])
    solver = TorchVorticitySolver(g, [p["nu"] for p in params], drag=cfg.drag,
                                  forcing_vorticity=forcing, scheme=sim.scheme)
    b = len(params)
    n_sub = round(sim.save_dt / dt)
    max_cfl = torch.zeros(b, dtype=torch.float64, device=device)
    rr_max = torch.zeros(b, dtype=torch.float64, device=device)
    e_sum = torch.zeros(b, dtype=torch.float64, device=device)

    def check(w_hat: torch.Tensor) -> None:
        cfl = g.max_velocity(w_hat).double() * dt / g.dx
        max_cfl.copy_(torch.maximum(max_cfl, cfl))
        if not bool(torch.isfinite(torch.view_as_real(w_hat)).all()):
            raise FloatingPointError("non-finite values in simulation")
        if float(cfl.max()) > sim.max_cfl:
            raise CFLViolation(f"CFL number {float(cfl.max()):.2f} exceeds {sim.max_cfl} "
                               f"(dt = {dt}, n = {n}); reduce dt for split {split.name!r}")

    frames: dict[int, list[torch.Tensor]] = {res: [] for res in r["resolutions"]}
    with torch.no_grad():
        w_hat = solver.initial_state(w0)
        check(w_hat)
        for i in range(round(r["spinup"] / dt)):
            w_hat = solver.step(w_hat, dt)
            if (i + 1) % n_sub == 0:
                check(w_hat)
        for k in range(r["n_frames"]):
            if k > 0:
                for _ in range(n_sub):
                    w_hat = solver.step(w_hat, dt)
            check(w_hat)
            rr_max = torch.maximum(rr_max, g.resolution_ratio(w_hat).double())
            e_sum += g.energy(w_hat).double()
            for res in r["resolutions"]:
                frames[res].append(spectral_downsample(w_hat, n, res).float().cpu())

    return {
        "frames": {res: torch.stack(f, dim=1).numpy() for res, f in frames.items()},
        "max_cfl": max_cfl.cpu().numpy(),
        "resolution_ratio": rr_max.cpu().numpy(),
        "energy_mean": (e_sum / r["n_frames"]).cpu().numpy(),
    }


# --------------------------------------------------------------------- storage


def set_attrs(f: h5py.File, attrs: dict[str, Any]) -> None:
    """Write attributes, storing strings with an explicit variable-length string dtype."""
    for key, value in attrs.items():
        if isinstance(value, str):
            f.attrs.create(key, value, dtype=h5py.string_dtype())
        else:
            f.attrs[key] = value


def git_revision() -> str:
    try:
        root = Path(nfds.__file__).resolve().parents[2]
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def generate_split(cfg: DatasetConfig, split: SplitConfig, out_dir: Path, device: str,
                   verbose: bool = True) -> dict[str, Any]:
    sim, r = cfg.simulation, cfg.resolved(split)
    n_traj, t_frames = split.n_traj, r["n_frames"]
    path = out_dir / f"{split.name}.h5"
    params = [sample_parameters(cfg, split, i) for i in range(n_traj)]
    t_start = time.perf_counter()
    with h5py.File(path, "w") as f:
        for res in r["resolutions"]:
            f.create_dataset(f"vorticity_{res}", shape=(n_traj, t_frames, res, res),
                             dtype="float32", chunks=(1, min(t_frames, 16), res, res))
        f["times"] = np.arange(t_frames) * sim.save_dt
        f["nu"] = np.array([p["nu"] for p in params])
        f["k_peak"] = np.array([p["k_peak"] for p in params])
        f["traj_index"] = np.arange(n_traj, dtype=np.int64)
        f.create_dataset("ic_kind", data=[p["ic_kind"] for p in params],
                         dtype=h5py.string_dtype())
        for name in ("resolution_ratio", "max_cfl", "energy_mean"):
            f.create_dataset(name, shape=(n_traj,), dtype="float64")
        f.create_dataset("accepted", shape=(n_traj,), dtype=bool)

        for start in range(0, n_traj, sim.batch_size):
            batch = params[start : start + sim.batch_size]
            out = simulate_batch(cfg, split, batch, device)
            end = start + len(batch)
            for res, arr in out["frames"].items():
                f[f"vorticity_{res}"][start:end] = arr
            for name in ("resolution_ratio", "max_cfl", "energy_mean"):
                f[name][start:end] = out[name]
            f["accepted"][start:end] = out["resolution_ratio"] <= cfg.quality.max_resolution_ratio
            if verbose:
                el = time.perf_counter() - t_start
                eta = el / end * (n_traj - end)
                print(f"  [{split.name}] {end}/{n_traj} trajectories, "
                      f"{el:.0f} s elapsed, ~{eta:.0f} s left", flush=True)

        set_attrs(f, {
            "split": split.name, "dataset": cfg.name, "config_json": cfg.to_json(),
            "config_fingerprint": cfg.fingerprint(), "nfds_version": nfds.__version__,
            "git_revision": git_revision(), "torch_version": torch.__version__,
            "device": device, "created_utc": datetime.now(timezone.utc).isoformat(),
            "sim_n": r["sim_n"], "dt": r["dt"], "spinup": r["spinup"],
            "save_dt": sim.save_dt, "drag": cfg.drag,
        })
        accepted = f["accepted"][:]
        summary = {
            "split": split.name, "n_traj": n_traj, "n_accepted": int(accepted.sum()),
            "n_frames": t_frames, "resolutions": r["resolutions"], "sim_n": r["sim_n"],
            "dt": r["dt"], "spinup": r["spinup"],
            "nu_min": float(f["nu"][:].min()), "nu_max": float(f["nu"][:].max()),
            "ic_kinds": sorted(set(p["ic_kind"] for p in params)),
            "max_resolution_ratio": float(f["resolution_ratio"][:].max()),
            "max_cfl": float(f["max_cfl"][:].max()),
            "seconds": round(time.perf_counter() - t_start, 1),
            "split_fingerprint": cfg.split_fingerprint(split.name),
        }
    return summary


def compute_stats(path: Path, resolution: int, chunk: int = 32) -> dict[str, float]:
    """Mean/std of vorticity and of one-step increments over accepted trajectories."""
    s1 = s2 = d1 = d2 = 0.0
    n_w = n_d = 0
    with h5py.File(path, "r") as f:
        acc = np.flatnonzero(f["accepted"][:])
        ds = f[f"vorticity_{resolution}"]
        for i in range(0, len(acc), chunk):
            w = ds[acc[i : i + chunk]].astype(np.float64)
            d = np.diff(w, axis=1)
            s1, s2, n_w = s1 + w.sum(), s2 + (w**2).sum(), n_w + w.size
            d1, d2, n_d = d1 + d.sum(), d2 + (d**2).sum(), n_d + d.size
    if n_w == 0:
        raise RuntimeError("no accepted training trajectories; cannot compute statistics")
    mean, dmean = s1 / n_w, d1 / n_d
    return {
        "resolution": resolution,
        "vorticity_mean": mean, "vorticity_std": math.sqrt(s2 / n_w - mean**2),
        "delta_mean": dmean, "delta_std": math.sqrt(d2 / n_d - dmean**2),
        "n_trajectories": int(len(acc)),
    }


def sha256(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_datasheet(cfg: DatasetConfig, summaries: list[dict], stats: dict, out_dir: Path,
                    command: str) -> None:
    sim = cfg.simulation
    forcing = ("none" if cfg.forcing.type == "none" else
               f"Kolmogorov, f = ({cfg.forcing.amplitude} sin({cfg.forcing.wavenumber}y), 0)")
    rows = "\n".join(
        f"| {s['split']} | {s['n_traj']} | {s['n_accepted']} | {s['nu_min']:.2e} – "
        f"{s['nu_max']:.2e} | {', '.join(s['ic_kinds'])} | {s['sim_n']} | "
        f"{', '.join(str(x) for x in s['resolutions'])} | {s['n_frames']} | {s['spinup']:g} | "
        f"{s['max_resolution_ratio']:.1e} | {s['max_cfl']:.2f} |"
        for s in summaries)
    text = f"""# Datasheet: {cfg.name} (v{cfg.version})

{cfg.description.strip()}

Generated by `nfds` {nfds.__version__} (git `{git_revision()}`), PyTorch {torch.__version__},
on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. Config fingerprint `{cfg.fingerprint()}`.

## Physics

2D incompressible Navier–Stokes in vorticity form on the periodic box [0, {cfg.domain_length:.4f})²,
viscosity ν (varies per trajectory), linear drag α = {cfg.drag}, forcing: {forcing}.
Re is reported as 1/ν (unit velocity and length scales).

## Simulation

Pseudo-spectral, 2/3 dealiasing, {sim.scheme.upper()} with exact integrating factor, {sim.dtype},
fixed time step per split (below), frames saved every {sim.save_dt} time units after spin-up,
{sim.n_frames} frames per trajectory unless a split overrides it (see the table). Stored fields are spectrally truncated (ideal low-pass) from the
simulation grid to each stored resolution.

## Splits

Splits are disjoint by construction: every trajectory's random stream is derived from
(base seed {cfg.base_seed}, split name, trajectory index).

| Split | Trajectories | Accepted | ν range | Initial conditions | Sim grid | Stored grids | Frames | Spin-up | Max res. ratio | Max CFL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

## Quality control

A trajectory is accepted when its resolution ratio (max over saved frames of E(k) near the dealiasing
cutoff divided by peak E(k)) stays at or below {cfg.quality.max_resolution_ratio:g}. Rejected
trajectories are kept in the files with `accepted = False` and excluded by the default loader.
Generation aborts if any trajectory exceeds CFL {sim.max_cfl}.

## Normalisation (train split, accepted trajectories, {stats['resolution']}² grid)

| Quantity | Mean | Std |
| --- | --- | --- |
| vorticity | {stats['vorticity_mean']:.4g} | {stats['vorticity_std']:.4g} |
| one-frame increment | {stats['delta_mean']:.4g} | {stats['delta_std']:.4g} |

## Reproducibility

Regenerate with:

```bash
{command}
```

The time step is fixed per split, so each trajectory is independent of batch size and generation
order. Results are bitwise reproducible on the same device type and software versions. Across
devices (CPU vs GPU) FFT round-off differs, and because the flow is chaotic, individual trajectories
decorrelate after a few eddy turnover times; their statistics do not change.

## Known limitations

- Single forcing geometry (Kolmogorov, k_f = {cfg.forcing.wavenumber}); other forcings are out of distribution.
- Re = 1/ν is a nominal value; the effective Reynolds number based on the measured r.m.s. velocity
  differs (see `energy_mean`).
"""
    (out_dir / "DATASHEET.md").write_text(text, encoding="utf-8")


def generate(cfg: DatasetConfig, out_dir: str | Path, device: str = "cpu",
             splits: list[str] | None = None, command: str = "", verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(_config_to_yaml(cfg), sort_keys=False), encoding="utf-8")
    todo = [s for s in cfg.splits if splits is None or s.name in splits]
    summaries = []
    for s in todo:
        if verbose:
            r = cfg.resolved(s)
            print(f"split {s.name}: {s.n_traj} trajectories at {r['sim_n']}², dt = {r['dt']}, "
                  f"spin-up {r['spinup']}", flush=True)
        summaries.append(generate_split(cfg, s, out_dir, device, verbose))

    if not (out_dir / "train.h5").exists():
        raise RuntimeError("train.h5 is missing: generate the train split first "
                           "(normalisation statistics come from it)")
    stats = compute_stats(out_dir / "train.h5", cfg.simulation.train_n)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Merge with splits generated by earlier runs into the same directory, so that
    # regenerating one split keeps the others in the manifest and datasheet.
    # A previous entry is kept only if that split's settings are unchanged, so adding a
    # new split to the config and generating just that split keeps the existing ones.
    previous = {}
    if (out_dir / "manifest.json").exists():
        old = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        current = {s.name for s in cfg.splits}
        for k, v in old["files"].items():
            name = k.removesuffix(".h5")
            if name in current and v.get("split_fingerprint") == cfg.split_fingerprint(name):
                previous[name] = {kk: vv for kk, vv in v.items() if kk not in ("sha256", "bytes")}
    merged = {**previous, **{s["split"]: s for s in summaries}}
    order = [s.name for s in cfg.splits]
    all_summaries = [merged[name] for name in order if name in merged]
    manifest = {
        "dataset": cfg.name, "version": cfg.version, "config_fingerprint": cfg.fingerprint(),
        "git_revision": git_revision(), "nfds_version": nfds.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {f"{s['split']}.h5": {**s, "sha256": sha256(out_dir / f"{s['split']}.h5"),
                                       "bytes": (out_dir / f"{s['split']}.h5").stat().st_size}
                  for s in all_summaries},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_datasheet(cfg, all_summaries, stats, out_dir, command)
    if verbose:
        for s in summaries:
            flag = "" if s["n_accepted"] == s["n_traj"] else "  <-- rejected trajectories"
            print(f"{s['split']:>8}: {s['n_accepted']}/{s['n_traj']} accepted, "
                  f"max res. ratio {s['max_resolution_ratio']:.1e}, max CFL {s['max_cfl']:.2f}, "
                  f"{s['seconds']:.0f} s{flag}")
    return manifest


def _config_to_yaml(cfg: DatasetConfig) -> dict:
    d = cfg.to_dict()
    d["splits"] = {s.pop("name"): s for s in d["splits"]}
    return d


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", nargs="+", help="only generate these splits (train is required "
                   "for statistics, so include it)")
    p.add_argument("--limit", type=int, help="cap trajectories per split (smoke runs)")
    args = p.parse_args(argv)
    cfg = load_config(args.config).with_limit(args.limit)
    command = (f"python -m nfds.data.generate {args.config} --out {args.out}"
               + (f" --limit {args.limit}" if args.limit else ""))
    generate(cfg, args.out, args.device, args.splits, command)


if __name__ == "__main__":
    main()
