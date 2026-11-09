"""Dataset configuration: a YAML file fully determines a dataset.

Every split is generated from ``base_seed`` and the split's name, so a
dataset can be regenerated exactly from its config (see DATASHEET.md for the
precise reproducibility guarantees).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ForcingConfig:
    type: str = "kolmogorov"  # "kolmogorov" | "none"
    wavenumber: int = 4
    amplitude: float = 1.0


@dataclass
class ViscosityConfig:
    dist: str = "loguniform"  # "loguniform" | "choice" | "fixed"
    low: float | None = None
    high: float | None = None
    values: list[float] | None = None
    exclude: list[float] | None = None  # loguniform only: [lo, hi] band held out

    def validate(self) -> None:
        if self.dist == "loguniform":
            if not (self.low and self.high and 0 < self.low < self.high):
                raise ValueError("loguniform viscosity needs 0 < low < high")
            if self.exclude is not None:
                lo, hi = self.exclude
                if not (self.low < lo < hi < self.high):
                    raise ValueError("exclude band must lie strictly inside [low, high]")
        elif self.dist in ("choice", "fixed"):
            if not self.values:
                raise ValueError(f"{self.dist} viscosity needs a non-empty 'values' list")
        else:
            raise ValueError(f"unknown viscosity dist {self.dist!r}")


@dataclass
class ICConfig:
    type: str = "random"  # "random" | "mixed"
    k_peak: list[float] = field(default_factory=lambda: [3.0, 6.0])  # uniform range
    energy: float = 0.5
    kinds: list[str] = field(
        default_factory=lambda: ["shear_layer", "dipole", "taylor_green_perturbed"])


@dataclass
class SplitConfig:
    name: str
    n_traj: int
    nu: ViscosityConfig
    ic: ICConfig = field(default_factory=ICConfig)
    # Optional per-split overrides of the simulation settings
    sim_n: int | None = None
    dt: float | None = None
    spinup: float | None = None
    resolutions: list[int] | None = None
    n_frames: int | None = None


@dataclass
class SimulationConfig:
    sim_n: int = 256
    train_n: int = 64
    dt: float = 2.5e-3
    save_dt: float = 0.25
    n_frames: int = 100
    spinup: float = 10.0
    scheme: str = "ifrk4"
    dtype: str = "float32"
    batch_size: int = 64
    max_cfl: float = 1.0  # abort if any trajectory exceeds this CFL number


@dataclass
class QualityConfig:
    max_resolution_ratio: float = 2e-5


@dataclass
class DatasetConfig:
    name: str
    description: str = ""
    version: int = 1
    base_seed: int = 0
    domain_length: float = 6.283185307179586
    drag: float = 0.1
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    splits: list[SplitConfig] = field(default_factory=list)

    # ------------------------------------------------------------- helpers
    def split(self, name: str) -> SplitConfig:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(name)

    def resolved(self, split: SplitConfig) -> dict[str, Any]:
        """Effective simulation settings for one split (overrides applied)."""
        sim = self.simulation
        return {
            "sim_n": split.sim_n or sim.sim_n,
            "dt": split.dt or sim.dt,
            "spinup": sim.spinup if split.spinup is None else split.spinup,
            "resolutions": split.resolutions or [sim.train_n],
            "n_frames": split.n_frames or sim.n_frames,
        }

    def validate(self) -> None:
        sim = self.simulation
        names = [s.name for s in self.splits]
        if len(set(names)) != len(names):
            raise ValueError("split names must be unique")
        if "train" not in names:
            raise ValueError("a 'train' split is required (normalisation statistics come from it)")
        for s in self.splits:
            s.nu.validate()
            r = self.resolved(s)
            for label, val in (("save_dt", sim.save_dt), ("spinup", r["spinup"])):
                k = val / r["dt"]
                if abs(k - round(k)) > 1e-6:
                    raise ValueError(f"split {s.name}: {label}={val} is not a multiple of dt")
            for res in r["resolutions"]:
                if res > r["sim_n"] or res % 2:
                    raise ValueError(f"split {s.name}: resolution {res} must be even and "
                                     f"<= sim_n {r['sim_n']}")
            if s.ic.type not in ("random", "mixed"):
                raise ValueError(f"unknown ic type {s.ic.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def split_fingerprint(self, name: str) -> str:
        """Hash of everything that determines one split's data: the shared settings
        (physics, simulation, quality, seed) plus that split's own config. Adding or
        changing another split leaves it unchanged."""
        d = self.to_dict()
        d.pop("splits")
        for key in ("name", "description", "version"):
            d.pop(key, None)
        split = next(asdict(s) for s in self.splits if s.name == name)
        blob = json.dumps({"shared": d, "split": split}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_limit(self, limit: int | None) -> DatasetConfig:
        """Copy with every split capped at ``limit`` trajectories (for smoke tests)."""
        cfg = copy.deepcopy(self)
        if limit is not None:
            for s in cfg.splits:
                s.n_traj = min(s.n_traj, limit)
        return cfg


def _build(cls, data: dict | None):
    return cls(**(data or {}))


def config_from_dict(d: dict[str, Any]) -> DatasetConfig:
    d = copy.deepcopy(d)
    splits = []
    for name, s in d.pop("splits", {}).items():
        s = dict(s)
        nu = ViscosityConfig(**s.pop("nu"))
        ic = _build(ICConfig, s.pop("ic", None))
        splits.append(SplitConfig(name=name, nu=nu, ic=ic, **s))
    cfg = DatasetConfig(
        forcing=_build(ForcingConfig, d.pop("forcing", None)),
        simulation=_build(SimulationConfig, d.pop("simulation", None)),
        quality=_build(QualityConfig, d.pop("quality", None)),
        splits=splits,
        **d,
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> DatasetConfig:
    with open(path) as f:
        return config_from_dict(yaml.safe_load(f))
