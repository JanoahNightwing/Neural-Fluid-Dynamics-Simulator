"""Visual summary of a generated dataset.

    python -m nfds.data.inspect data/forced_dev --out figures/dataset_forced_dev.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = list(manifest["files"])
    res = json.loads((root / "stats.json").read_text())["resolution"]

    fig = plt.figure(figsize=(3.0 * max(4, len(splits)), 9), constrained_layout=True)
    gs = fig.add_gridspec(3, max(4, len(splits)))
    nus, rr = {}, {}
    for j, fname in enumerate(splits):
        with h5py.File(root / fname) as f:
            name = fname.removesuffix(".h5")
            w = f[f"vorticity_{res}"]
            nus[name], rr[name] = f["nu"][:], f["resolution_ratio"][:]
            kind = f["ic_kind"][0]
            kind = kind.decode() if isinstance(kind, bytes) else kind
            for row, t in enumerate((0, w.shape[1] - 1)):
                ax = fig.add_subplot(gs[row, j])
                frame = w[0, t]
                vmax = np.percentile(np.abs(frame), 99.5)
                ax.imshow(frame, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{name} #0 ({kind})\nnu={f['nu'][0]:.1e}, frame {t}"
                             if row == 0 else f"frame {t}", fontsize=9)

    ax = fig.add_subplot(gs[2, : max(2, len(splits) // 2)])
    for name, v in nus.items():
        ax.hist(v, bins=np.logspace(-4, -1.7, 50), alpha=0.6, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("viscosity nu")
    ax.set_ylabel("trajectories")
    ax.set_title("Viscosity per split (gap = held-out interpolation band)", fontsize=10)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, max(2, len(splits) // 2) :])
    for name, v in rr.items():
        ax.scatter(nus[name], v, s=8, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("viscosity nu")
    ax.set_ylabel("max resolution ratio")
    ax.set_title("Quality check per trajectory", fontsize=10)
    fig.suptitle(f"{manifest['dataset']}: sample trajectories at {res}²")
    out = args.out or str(root / "summary.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
