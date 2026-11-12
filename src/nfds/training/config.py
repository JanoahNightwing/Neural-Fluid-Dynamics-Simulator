"""Training configuration (YAML)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGIMES = ("onestep", "noise", "unroll", "pushforward")


@dataclass
class RegimeConfig:
    """How targets are formed (the core ablation of the project).

    onestep:     input = true frame, predict one step (teacher forcing)
    noise:       like onestep, with Gaussian noise of ``noise_std`` (normalised units)
                 added to the input so the model learns to correct its own drift
    unroll:      roll ``unroll`` steps and backpropagate through all of them
    pushforward: roll ``unroll - 1`` steps without gradients, then train the last step
                 on the model's own (detached) predictions (Brandstetter et al., 2022)
    """

    type: str = "onestep"
    unroll: int = 1
    noise_std: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 32
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.01
    # Split each batch into micro-batches whose gradients are accumulated before the optimiser
    # step. Mathematically the same update as the full batch (the loss is a per-sample mean);
    # used when unrolled training does not fit in GPU memory.
    micro_batch_size: int | None = None
    # Fixed number of optimiser steps per epoch (the loader is cycled if the dataset is
    # smaller), so runs on datasets of different sizes train for the same number of steps.
    # None: one pass over the training windows.
    steps_per_epoch: int | None = None


@dataclass
class ValConfig:
    horizon: int = 20  # rollout length used for model selection
    every: int = 1


@dataclass
class TrainConfig:
    name: str
    data: str
    model: dict[str, Any] = field(default_factory=lambda: {"name": "cnn"})
    n_in: int = 1
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    val: ValConfig = field(default_factory=ValConfig)
    seed: int = 0
    device: str = "auto"
    amp: bool = False  # bf16 autocast on CUDA
    num_workers: int = 0
    out_dir: str = "runs"
    train_split: str = "train"
    val_split: str = "val"
    # (start, stop) ranges of trajectories: data-scaling subsets and few-shot fine-tuning
    train_trajectories: list[int] | None = None
    val_trajectories: list[int] | None = None
    # Start from a trained checkpoint (fine-tuning). Model architecture, input window and
    # normalisation are taken from the checkpoint; ``model`` and ``n_in`` are ignored.
    init_checkpoint: str | None = None
    max_batches_per_epoch: int | None = None  # cap for quick/dry runs
    # Cap on the GPU memory PyTorch may allocate. Above it training fails with an out-of-memory
    # error instead of silently spilling into system RAM (the Windows driver's default), which
    # makes training 20-30x slower (E011).
    gpu_memory_fraction: float = 0.95

    def validate(self) -> None:
        r = self.regime
        if r.type not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if r.type in ("onestep", "noise") and r.unroll != 1:
            raise ValueError(f"regime {r.type!r} uses unroll = 1")
        if r.type in ("unroll", "pushforward") and r.unroll < 2:
            raise ValueError(f"regime {r.type!r} needs unroll >= 2")
        if r.type == "noise" and r.noise_std <= 0:
            raise ValueError("regime 'noise' needs noise_std > 0")
        mb = self.optim.micro_batch_size
        if mb is not None and (mb < 1 or self.optim.batch_size % mb != 0):
            raise ValueError(f"micro_batch_size {mb} must divide batch_size "
                             f"{self.optim.batch_size}")
        if self.optim.steps_per_epoch is not None and self.optim.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        if "name" not in self.model:
            raise ValueError("model config needs a 'name'")
        for rng in (self.train_trajectories, self.val_trajectories):
            if rng is not None and (len(rng) != 2 or not 0 <= rng[0] < rng[1]):
                raise ValueError(f"trajectory ranges must be [start, stop], got {rng}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name


def config_from_dict(d: dict[str, Any]) -> TrainConfig:
    d = copy.deepcopy(d)
    cfg = TrainConfig(
        regime=RegimeConfig(**(d.pop("regime", None) or {})),
        optim=OptimConfig(**(d.pop("optim", None) or {})),
        val=ValConfig(**(d.pop("val", None) or {})),
        **d,
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """Load YAML, applying ``key.sub=value`` overrides (values parsed as YAML)."""
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config_from_dict(d)
