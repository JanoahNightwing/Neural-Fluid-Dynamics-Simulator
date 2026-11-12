"""Measure training cost per model and regime on the full dataset, to size the M5 study.

    python scripts/time_training.py                       # all models, ~10 min on a GPU
    python scripts/time_training.py --models unet fno --batches 30

For each (model, regime) this trains two short epochs of ``--batches`` batches with the real
v1 configs (the second epoch is timed, so CUDA warm-up and autotuning are excluded) and runs
validation twice. It reports seconds per batch, seconds per validation, peak GPU memory and the
extrapolated time for one full epoch and for the configured number of epochs. A run that does
not fit in GPU memory is retried with gradient accumulation over smaller micro-batches (same
batch size, same update), halving the micro-batch until it fits. Results go to
figures/train_timing.json. Scratch runs are written to runs/_timing (safe to delete).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from nfds.training.config import load_config
from nfds.training.train import train

CONFIGS = {m: f"configs/train/{m}_v1.yaml" for m in ("cnn", "unet", "fno", "hybrid")}
REGIMES = {
    "onestep": ["regime.type=onestep", "regime.unroll=1"],
    "unroll4": ["regime.type=unroll", "regime.unroll=4"],
    "pushforward4": ["regime.type=pushforward", "regime.unroll=4"],
}
SKIP = {("hybrid", "pushforward4")}  # not part of the study plan


def probe(model: str, regime: str, batches: int, micro: int | None, out_dir: str) -> dict:
    overrides = [*REGIMES[regime], "optim.epochs=2", "optim.warmup_epochs=0",
                 f"out_dir={out_dir}", f"name={model}-{regime}"]
    if micro is not None:
        overrides.append(f"optim.micro_batch_size={micro}")
    cfg = load_config(CONFIGS[model], overrides)
    full_epochs = load_config(CONFIGS[model]).optim.epochs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    summary = train(cfg, max_batches_per_epoch=batches, verbose=False)
    recs = [json.loads(line) for line in
            (cfg.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    s_batch = recs[-1]["epoch_seconds"] / batches
    s_val = (summary["train_seconds"] - sum(r["epoch_seconds"] for r in recs)) / len(recs)
    full_batches = summary["train_windows"] // cfg.optim.batch_size
    epoch_min = (full_batches * s_batch + s_val) / 60
    return {
        "model": model, "regime": regime, "batch_size": cfg.optim.batch_size,
        "micro_batch_size": cfg.optim.micro_batch_size or cfg.optim.batch_size,
        "parameters": summary["parameters"], "s_per_batch": s_batch, "s_per_val": s_val,
        "batches_per_epoch": full_batches,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "epoch_min": epoch_min, "epochs": full_epochs, "run_hours": epoch_min * full_epochs / 60,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    p.add_argument("--batches", type=int, default=40, help="timed batches per probe")
    p.add_argument("--out", default="figures/train_timing.json")
    p.add_argument("--scratch", default="runs/_timing")
    args = p.parse_args()

    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("device: CPU (no CUDA) - timings will not be representative", file=sys.stderr)

    rows = []
    header = (f"{'model':7s} {'regime':13s} {'batch':>5s} {'micro':>5s} {'ms/batch':>9s} "
              f"{'val s':>6s} {'peak GB':>8s} {'min/epoch':>10s} {'epochs':>6s} {'h/run':>6s}")
    print(header, flush=True)
    for model in args.models:
        for regime in args.regimes:
            if (model, regime) in SKIP:
                continue
            micro, row = None, None
            while row is None:
                try:
                    row = probe(model, regime, args.batches, micro, args.scratch)
                except torch.cuda.OutOfMemoryError:
                    current = micro or load_config(CONFIGS[model]).optim.batch_size
                    if current <= 4:
                        print(f"{model:7s} {regime:13s} out of memory even with micro-batch 4",
                              flush=True)
                        break
                    micro = current // 2
                    print(f"{model:7s} {regime:13s} out of memory with micro-batch {current}, "
                          f"retrying with {micro}", flush=True)
            if row is None:
                continue
            rows.append(row)
            peak = f"{row['peak_gb']:.2f}" if row["peak_gb"] is not None else "-"
            print(f"{model:7s} {regime:13s} {row['batch_size']:5d} {row['micro_batch_size']:5d} "
                  f"{row['s_per_batch'] * 1e3:9.1f} {row['s_per_val']:6.1f} {peak:>8s} "
                  f"{row['epoch_min']:10.1f} {row['epochs']:6d} {row['run_hours']:6.2f}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out.write_text(json.dumps({"device": device, "timed_batches": args.batches, "rows": rows},
                              indent=2), encoding="utf-8")
    shutil.rmtree(args.scratch, ignore_errors=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
