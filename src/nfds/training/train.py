"""Train a surrogate.

    python -m nfds.training.train configs/train/unet_dev.yaml
    python -m nfds.training.train configs/train/unet_dev.yaml optim.epochs=5 name=unet_quick

Writes <out_dir>/<name>/: config.yaml, metrics.jsonl (one line per epoch),
best.pt (lowest validation rollout error) and last.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from nfds.data.dataset import TrajectoryWindowDataset, load_split, load_stats
from nfds.evaluation.metrics import relative_l2
from nfds.evaluation.rollout import rollout
from nfds.evaluation.steppers import NeuralStepper, NuNormalizer
from nfds.models import build_model, count_parameters
from nfds.training.checkpoint import save_checkpoint
from nfds.training.config import TrainConfig, load_config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of per-sample relative L2 error."""
    num = (pred - target).flatten(1).norm(dim=1)
    return (num / target.flatten(1).norm(dim=1).clamp_min(1e-8)).mean()


def compute_loss(model: torch.nn.Module, batch: dict, nu_norm: NuNormalizer, cfg: TrainConfig,
                 device: torch.device) -> torch.Tensor:
    x = batch["input"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)
    nu_f = nu_norm(batch["nu"].to(device))
    r = cfg.regime
    if r.type in ("onestep", "noise"):
        if r.type == "noise":
            x = x + r.noise_std * torch.randn_like(x)
        return rel_l2_loss(model(x, nu_f)[:, 0], y[:, 0])
    if r.type == "unroll":
        window, losses = x, []
        for k in range(r.unroll):
            p = model(window, nu_f)
            losses.append(rel_l2_loss(p[:, 0], y[:, k]))
            window = torch.cat([window[:, 1:], p], dim=1)
        return torch.stack(losses).mean()
    # pushforward
    window = x
    with torch.no_grad():
        for _ in range(r.unroll - 1):
            window = torch.cat([window[:, 1:], model(window, nu_f)], dim=1)
    return rel_l2_loss(model(window.detach(), nu_f)[:, 0], y[:, r.unroll - 1])


def lr_lambda(cfg: TrainConfig, steps_per_epoch: int):
    total = cfg.optim.epochs * steps_per_epoch
    warm = max(1, int(cfg.optim.warmup_epochs * steps_per_epoch))
    floor = cfg.optim.min_lr_ratio

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


@torch.no_grad()
def validate(model: torch.nn.Module, val_loader: DataLoader, val_data: dict, stats: dict,
             nu_norm: NuNormalizer, cfg: TrainConfig, device: torch.device) -> dict:
    model.eval()
    losses = [float(rel_l2_loss(model(b["input"].to(device), nu_norm(b["nu"].to(device)))[:, 0],
                                b["target"][:, 0].to(device))) for b in val_loader]
    stepper = NeuralStepper(model, stats["vorticity_mean"], stats["vorticity_std"], nu_norm,
                            cfg.n_in, device=device)
    w = torch.from_numpy(val_data["vorticity"])
    nu = torch.from_numpy(val_data["nu"]).float()
    horizon = min(cfg.val.horizon, w.shape[1] - cfg.n_in)
    pred = rollout(stepper, w, nu, horizon)
    rel = relative_l2(pred.double(), w[:, cfg.n_in : cfg.n_in + horizon].double())
    rel = torch.nan_to_num(rel, nan=10.0, posinf=10.0).clamp(max=10.0)
    model.train()
    return {"val_onestep": float(np.mean(losses)), "val_rollout": float(rel.mean()),
            "val_rollout_last": float(rel[:, -1].mean())}


def train(cfg: TrainConfig, max_batches_per_epoch: int | None = None,
          verbose: bool = True) -> dict:
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    stats = load_stats(cfg.data)
    unroll = cfg.regime.unroll
    train_ds = TrajectoryWindowDataset(cfg.data, cfg.train_split, n_in=cfg.n_in, n_out=unroll)
    val_ds = TrajectoryWindowDataset(cfg.data, cfg.val_split, n_in=cfg.n_in, n_out=1, stride=4)
    val_data = load_split(cfg.data, cfg.val_split)
    nu_norm = NuNormalizer.from_values(train_ds.meta["nu"])
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.optim.batch_size, shuffle=True,
                              drop_last=True, generator=g, num_workers=cfg.num_workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=64)

    model_cfg = dict(cfg.model)
    model = build_model(model_cfg["name"], n_in=cfg.n_in,
                        **{k: v for k, v in model_cfg.items() if k != "name"}).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            weight_decay=cfg.optim.weight_decay)
    steps_per_epoch = min(len(train_loader), max_batches_per_epoch or len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps_per_epoch))
    use_amp = cfg.amp and device.type == "cuda"

    if verbose:
        print(f"{cfg.name}: {model_cfg['name']} with {n_params:,} parameters, "
              f"{len(train_ds):,} training windows, regime {cfg.regime.type} "
              f"(unroll {unroll}), device {device}", flush=True)
    best, history = math.inf, []
    log_path = run_dir / "metrics.jsonl"
    log_path.write_text("")
    t_start = time.perf_counter()
    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        t0, losses = time.perf_counter(), []
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = compute_loss(model, batch, nu_norm, cfg, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}, step {i}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "lr": sched.get_last_lr()[0], "epoch_seconds": time.perf_counter() - t0}
        if epoch % cfg.val.every == 0 or epoch == cfg.optim.epochs:
            rec.update(validate(model, val_loader, val_data, stats, nu_norm, cfg, device))
            ckpt_args = (model, model_cfg, cfg.n_in, stats, nu_norm, cfg.to_dict(), epoch, rec)
            if rec["val_rollout"] < best:
                best = rec["val_rollout"]
                save_checkpoint(run_dir / "best.pt", *ckpt_args)
                rec["best"] = True
            save_checkpoint(run_dir / "last.pt", *ckpt_args, optimizer=opt)
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if verbose:
            val = (f", val 1-step {rec['val_onestep']:.4f}, val rollout {rec['val_rollout']:.4f}"
                   + (" *" if rec.get("best") else "")) if "val_rollout" in rec else ""
            print(f"  epoch {epoch:3d}: train {rec['train_loss']:.4f}{val} "
                  f"({rec['epoch_seconds']:.0f} s)", flush=True)

    summary = {"name": cfg.name, "parameters": n_params, "best_val_rollout": best,
               "epochs": cfg.optim.epochs, "train_seconds": time.perf_counter() - t_start,
               "final": history[-1]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a surrogate model")
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    summary = train(cfg)
    print(f"done: best validation rollout error {summary['best_val_rollout']:.4f}, "
          f"{summary['train_seconds'] / 60:.1f} min -> {Path(cfg.run_dir) / 'best.pt'}")


if __name__ == "__main__":
    main()
