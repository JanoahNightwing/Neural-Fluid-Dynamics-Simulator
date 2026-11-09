"""Which predictors the app can offer, with the physics and viscosity range each was trained on."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

DEFAULT_PHYSICS = {"drag": 0.1, "forcing_wavenumber": 4, "forcing_amplitude": 1.0,
                   "save_dt": 0.25, "length": 2 * math.pi}


@dataclass
class Entry:
    id: str
    label: str
    kind: str  # "model" or "baseline"
    description: str
    resolution: int
    n_in: int
    nu_min: float
    nu_max: float
    physics: dict[str, Any]
    physics_source: str
    path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("path")
        return d


def _physics_from_data(data_dir: Path, split: str) -> tuple[dict, dict | None] | None:
    """Physics and training viscosity range from a dataset, if it is present on disk."""
    h5 = data_dir / f"{split}.h5"
    if not h5.exists():
        return None
    import h5py

    with h5py.File(h5) as f:
        attrs = dict(f.attrs)
    cfg = json.loads(attrs["config_json"])
    forc = cfg["forcing"]
    physics = {"drag": float(attrs["drag"]), "save_dt": float(attrs["save_dt"]),
               "forcing_wavenumber": forc["wavenumber"] if forc["type"] == "kolmogorov" else 0,
               "forcing_amplitude": forc.get("amplitude", 1.0), "length": cfg["domain_length"]}
    nu_cfg = next((s["nu"] for s in cfg["splits"] if s["name"] == split), None)
    return physics, nu_cfg


def entry_from_checkpoint(path: Path) -> Entry:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tc = ckpt.get("train_cfg", {})
    run = path.parent.name
    model_cfg = dict(ckpt["model_cfg"])
    model_type = model_cfg.pop("name")
    found, resolution = None, 64
    data_dir = Path(tc["data"]) if tc.get("data") else None  # relative to the working directory
    if data_dir is not None and data_dir.exists():
        found = _physics_from_data(data_dir, tc.get("train_split", "train"))
        stats = data_dir / "stats.json"
        if stats.exists():
            resolution = int(json.loads(stats.read_text())["resolution"])
    if found:
        physics, nu_cfg = found
        source = "dataset"
    else:
        physics, nu_cfg, source = dict(DEFAULT_PHYSICS), None, "defaults (dataset not found)"
    if nu_cfg and nu_cfg.get("dist") == "loguniform":
        nu_min, nu_max = nu_cfg["low"], nu_cfg["high"]
    else:  # log-uniform has std = width / sqrt(12): recover the range from the normaliser
        half = math.sqrt(3) * ckpt["nu_log_std"]
        nu_min = math.exp(ckpt["nu_log_mean"] - half)
        nu_max = math.exp(ckpt["nu_log_mean"] + half)
    with torch.no_grad():
        n_params = sum(v.numel() for k, v in ckpt["model_state"].items()
                       if not k.endswith("_metadata") and torch.is_tensor(v))
    regime = tc.get("regime", {})
    names = {"cnn": "Residual CNN", "unet": "U-Net", "fno": "Fourier neural operator",
             "neuralop_fno": "FNO (neuraloperator)"}
    desc = (f"{names.get(model_type, model_type)}, {n_params / 1e6:.2f}M parameters, "
            f"{regime.get('type', 'onestep')} training, epoch {ckpt.get('epoch', '?')}")
    return Entry(id=run, label=f"{names.get(model_type, model_type)} ({run})", kind="model",
                 description=desc, resolution=resolution,
                 n_in=int(ckpt["n_in"]), nu_min=float(nu_min), nu_max=float(nu_max),
                 physics=physics, physics_source=source, path=str(path),
                 extra={"model_type": model_type, "regime": regime.get("type"),
                        "val_rollout": ckpt.get("metrics", {}).get("val_rollout"),
                        "data": tc.get("data")})


def discover(paths: list[str | Path]) -> list[Entry]:
    """Checkpoints given directly, or every best.pt found under the given directories."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_file() and p.suffix == ".pt":
            files.append(p)
        elif p.is_dir():
            files += sorted(p.rglob("best.pt"))
    entries, seen = [], set()
    for f in files:
        try:
            e = entry_from_checkpoint(f)
        except Exception as err:  # unreadable checkpoints are skipped, not fatal
            print(f"skipping {f}: {err}")
            continue
        if e.id in seen:
            e.id = f"{e.id}-{len(seen)}"
        seen.add(e.id)
        entries.append(e)
    if not entries:
        return []
    ref = entries[0]
    for name, label, desc in (
        ("coarse_solver", "Solver on the network's grid",
         "The reference solver run directly on the coarse grid: the honest baseline"),
        ("persistence", "Persistence", "Predicts that nothing changes: the floor"),
    ):
        entries.append(Entry(id=name, label=label, kind="baseline", description=desc,
                             resolution=ref.resolution, n_in=1, nu_min=ref.nu_min,
                             nu_max=ref.nu_max, physics=ref.physics,
                             physics_source=ref.physics_source))
    return entries
