"""PyTorch datasets over generated trajectories.

    ds = TrajectoryWindowDataset("data/forced_v1", "train", n_in=1, n_out=4)
    sample = ds[0]   # {"input": (1, 64, 64), "target": (4, 64, 64), "nu": (), ...}

Windows never cross trajectory boundaries, and trajectories rejected by the
quality checks are excluded by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Affine map between physical vorticity and the network's normalised units."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def from_stats(cls, stats: dict) -> Normalizer:
        return cls(stats["vorticity_mean"], stats["vorticity_std"])

    def encode(self, w):
        return (w - self.mean) / self.std

    def decode(self, z):
        return z * self.std + self.mean


def load_stats(root: str | Path) -> dict:
    return json.loads((Path(root) / "stats.json").read_text(encoding="utf-8"))


def load_split(root: str | Path, split: str, resolution: int | None = None,
               accepted_only: bool = True,
               trajectories: tuple[int, int] | list[int] | None = None) -> dict[str, np.ndarray]:
    """Load a split into memory: vorticity (N, T, n, n) plus per-trajectory metadata.

    ``trajectories = (start, stop)`` selects a contiguous range of the (accepted)
    trajectories, e.g. (0, 100) for a data-scaling subset, or disjoint ranges of
    one split for few-shot fine-tuning and evaluation.
    """
    path = Path(root) / f"{split}.h5"
    with h5py.File(path, "r") as f:
        if resolution is None:
            resolution = load_stats(root)["resolution"]
        key = f"vorticity_{resolution}"
        if key not in f:
            avail = sorted(k for k in f if k.startswith("vorticity_"))
            raise KeyError(f"{split}.h5 has no {key}; available: {avail}")
        keep = f["accepted"][:] if accepted_only else np.ones(len(f["nu"]), dtype=bool)
        idx = np.flatnonzero(keep)
        if trajectories is not None:
            start, stop = trajectories
            if not 0 <= start < stop:
                raise ValueError(f"invalid trajectory range {trajectories}")
            if stop > len(idx):
                raise ValueError(f"{split} has {len(idx)} trajectories, range {trajectories} "
                                 "is out of bounds")
            idx = idx[start:stop]
        out = {"vorticity": f[key][idx] if len(idx) else np.empty((0,) + f[key].shape[1:],
                                                                   np.float32)}
        for name in ("nu", "k_peak", "resolution_ratio", "energy_mean", "traj_index"):
            out[name] = f[name][:][idx]
        out["ic_kind"] = np.array([s.decode() if isinstance(s, bytes) else s
                                   for s in f["ic_kind"][:][idx]])
        out["times"] = f["times"][:]
    return out


class TrajectoryWindowDataset(Dataset):
    """Sliding windows (n_in input frames -> n_out target frames) within trajectories.

    Parameters
    ----------
    root : dataset directory
    split : split name, e.g. "train"
    n_in, n_out : frames in the input and target windows
    stride : spacing between window start times
    resolution : stored grid to use (default: the training resolution)
    normalize : apply the train-split normalisation to input and target
    trajectories : optional (start, stop) range of trajectories to use
    """

    def __init__(self, root: str | Path, split: str, n_in: int = 1, n_out: int = 1,
                 stride: int = 1, resolution: int | None = None, normalize: bool = True,
                 accepted_only: bool = True,
                 trajectories: tuple[int, int] | list[int] | None = None) -> None:
        if n_in < 1 or n_out < 1 or stride < 1:
            raise ValueError("n_in, n_out and stride must be >= 1")
        self.root, self.split = Path(root), split
        self.n_in, self.n_out = n_in, n_out
        data = load_split(root, split, resolution, accepted_only, trajectories)
        self.normalizer = Normalizer.from_stats(load_stats(root)) if normalize else None
        w = data["vorticity"]
        if self.normalizer is not None:
            w = self.normalizer.encode(w).astype(np.float32)
        self.vorticity = torch.from_numpy(np.ascontiguousarray(w))
        self.nu = torch.from_numpy(data["nu"]).float()
        self.meta = {k: v for k, v in data.items() if k != "vorticity"}
        n_traj, n_t = self.vorticity.shape[:2]
        span = n_in + n_out
        if n_t < span:
            raise ValueError(f"trajectories have {n_t} frames, window needs {span}")
        starts = np.arange(0, n_t - span + 1, stride)
        self.index = [(i, int(t)) for i in range(n_traj) for t in starts]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        i, t = self.index[k]
        w = self.vorticity[i]
        return {
            "input": w[t : t + self.n_in],
            "target": w[t + self.n_in : t + self.n_in + self.n_out],
            "nu": self.nu[i],
            "traj": torch.tensor(i),
            "t0": torch.tensor(t),
        }

    @property
    def num_trajectories(self) -> int:
        return self.vorticity.shape[0]
