"""Checkpoints carry everything needed to rebuild a stepper: weights, model config,
normalisation of vorticity and viscosity, input window length and training config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nfds.evaluation.steppers import NeuralStepper, NuNormalizer
from nfds.models import build_model


def save_checkpoint(path: str | Path, model: torch.nn.Module, model_cfg: dict, n_in: int,
                    stats: dict, nu_norm: NuNormalizer, train_cfg: dict, epoch: int,
                    metrics: dict, optimizer: torch.optim.Optimizer | None = None) -> None:
    state = {
        "model_state": model.state_dict(), "model_cfg": model_cfg, "n_in": n_in,
        "vorticity_mean": stats["vorticity_mean"], "vorticity_std": stats["vorticity_std"],
        "nu_log_mean": nu_norm.log_mean, "nu_log_std": nu_norm.log_std,
        "train_cfg": train_cfg, "epoch": epoch, "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                    ) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_cfg"])
    model = build_model(cfg.pop("name"), n_in=ckpt["n_in"], **cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def load_stepper(path: str | Path, device: str | torch.device = "cpu",
                 name: str | None = None) -> NeuralStepper:
    model, ckpt = load_checkpoint(path, device)
    return NeuralStepper(
        model, ckpt["vorticity_mean"], ckpt["vorticity_std"],
        NuNormalizer(ckpt["nu_log_mean"], ckpt["nu_log_std"]), n_in=ckpt["n_in"],
        name=name or ckpt["train_cfg"]["name"], device=device)
