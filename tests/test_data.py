"""Dataset pipeline: downsampling, determinism, splits, statistics, quality flags, loading."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from helpers import tiny_config  # noqa: E402

from nfds.data.config import config_from_dict  # noqa: E402
from nfds.data.dataset import TrajectoryWindowDataset, load_split  # noqa: E402
from nfds.data.generate import (  # noqa: E402
    CFLViolation,
    compute_stats,
    generate,
    sample_parameters,
    sha256,
)
from nfds.data.transforms import spectral_downsample  # noqa: E402
from nfds.grid import SpectralGrid  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    generate(tiny_config(), out, verbose=False)
    return out


# ----------------------------------------------------------------- transforms
def test_spectral_downsample_exact_for_band_limited_field():
    fine, coarse = SpectralGrid(64), SpectralGrid(16)
    def field(g):
        x, y = g.coords
        return np.sin(3 * x) * np.cos(5 * y) + 0.3 * np.cos(7 * x + 2 * y)
    w_hat = torch.fft.rfft2(torch.as_tensor(field(fine)))
    out = spectral_downsample(w_hat, 64, 16).numpy()
    np.testing.assert_allclose(out, field(coarse), atol=1e-12)


def test_spectral_downsample_removes_unresolved_modes():
    fine = SpectralGrid(64)
    x, y = fine.coords
    w_hat = torch.fft.rfft2(torch.as_tensor(np.sin(12 * x)))  # above coarse Nyquist (8)
    assert np.abs(spectral_downsample(w_hat, 64, 16).numpy()).max() < 1e-12


# --------------------------------------------------------------------- config
def test_config_rejects_bad_settings():
    with pytest.raises(ValueError):
        tiny_config(simulation={"sim_n": 32, "train_n": 16, "dt": 0.03, "save_dt": 0.05})
    with pytest.raises(ValueError):
        config_from_dict({"name": "x", "splits": {"val": {"n_traj": 1, "nu": {
            "dist": "fixed", "values": [0.01]}}}})  # no train split


def test_parameters_are_independent_of_split_order_and_distinct_across_splits():
    cfg = tiny_config()
    a = sample_parameters(cfg, cfg.split("train"), 0)
    b = sample_parameters(cfg, cfg.split("val"), 0)
    assert a["ic_seed"] != b["ic_seed"]
    assert sample_parameters(cfg, cfg.split("train"), 0) == a


def test_excluded_viscosity_band_is_never_sampled():
    cfg = config_from_dict({"name": "x", "splits": {"train": {"n_traj": 1, "nu": {
        "dist": "loguniform", "low": 1e-3, "high": 5e-3, "exclude": [1.8e-3, 2.4e-3]}}}})
    nus = np.array([sample_parameters(cfg, cfg.split("train"), i)["nu"] for i in range(2000)])
    assert nus.min() >= 1e-3 and nus.max() <= 5e-3
    assert not np.any((nus > 1.8e-3) & (nus < 2.4e-3))
    assert np.any(nus < 1.8e-3) and np.any(nus > 2.4e-3)
    with pytest.raises(ValueError):
        config_from_dict({"name": "x", "splits": {"train": {"n_traj": 1, "nu": {
            "dist": "loguniform", "low": 1e-3, "high": 5e-3, "exclude": [5e-4, 2e-3]}}}})


# ----------------------------------------------------------------- generation
def test_files_shapes_and_metadata(tiny_dataset):
    for name in ("train", "val", "test_re", "test_ic"):
        assert (tiny_dataset / f"{name}.h5").exists()
    with h5py.File(tiny_dataset / "train.h5") as f:
        assert f["vorticity_16"].shape == (5, 6, 16, 16)
        assert f["accepted"][:].all()
        assert f.attrs["sim_n"] == 32 and "config_json" in f.attrs
        assert np.all((f["nu"][:] >= 0.01) & (f["nu"][:] <= 0.05))
    with h5py.File(tiny_dataset / "test_ic.h5") as f:
        assert f["vorticity_32"].shape == (3, 6, 32, 32)
        assert sorted(s.decode() for s in f["ic_kind"][:]) == [
            "dipole", "shear_layer", "taylor_green_perturbed"]
    with h5py.File(tiny_dataset / "test_re.h5") as f:
        assert sorted(f["nu"][:].tolist()) == [0.005, 0.1]
        assert f.attrs["sim_n"] == 48
    for name in ("stats.json", "manifest.json", "config.yaml", "DATASHEET.md"):
        assert (tiny_dataset / name).exists()


def test_manifest_hashes_match_files(tiny_dataset):
    manifest = json.loads((tiny_dataset / "manifest.json").read_text())
    for fname, entry in manifest["files"].items():
        assert entry["sha256"] == sha256(tiny_dataset / fname)


def test_generation_is_deterministic_and_batch_size_independent(tiny_dataset, tmp_path):
    cfg = tiny_config(simulation={"sim_n": 32, "train_n": 16, "dt": 0.01, "save_dt": 0.05,
                                  "n_frames": 6, "spinup": 0.1, "dtype": "float64",
                                  "batch_size": 2})
    generate(cfg, tmp_path, splits=["train"], verbose=False)
    with h5py.File(tiny_dataset / "train.h5") as a, h5py.File(tmp_path / "train.h5") as b:
        np.testing.assert_array_equal(a["vorticity_16"][:], b["vorticity_16"][:])


def test_regenerating_one_split_keeps_others_in_manifest(tmp_path):
    cfg = tiny_config()
    generate(cfg, tmp_path, splits=["train", "val"], verbose=False)
    generate(cfg, tmp_path, splits=["train"], verbose=False)
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert set(files) == {"train.h5", "val.h5"}
    assert "| val |" in (tmp_path / "DATASHEET.md").read_text()


def test_stats_come_from_train_split_only(tiny_dataset):
    stats = json.loads((tiny_dataset / "stats.json").read_text())
    with h5py.File(tiny_dataset / "train.h5") as f:
        w = f["vorticity_16"][:].astype(np.float64)
    assert np.isclose(stats["vorticity_std"], w.std(), rtol=1e-10)
    assert np.isclose(stats["delta_std"], np.diff(w, axis=1).std(), rtol=1e-10)
    assert stats["n_trajectories"] == 5


def test_quality_flag_excludes_trajectories(tmp_path):
    cfg = tiny_config(quality={"max_resolution_ratio": 1e-30})
    cfg.split("train").n_traj = 2
    with pytest.raises(RuntimeError, match="no accepted"):
        generate(cfg, tmp_path, splits=["train"], verbose=False)
    with h5py.File(tmp_path / "train.h5") as f:
        assert not f["accepted"][:].any()
    assert load_split(tmp_path, "train", resolution=16)["vorticity"].shape[0] == 0
    assert load_split(tmp_path, "train", resolution=16, accepted_only=False)[
        "vorticity"].shape[0] == 2


def test_cfl_guard(tmp_path):
    cfg = tiny_config(simulation={"sim_n": 32, "train_n": 16, "dt": 0.05, "save_dt": 0.05,
                                  "n_frames": 2, "spinup": 0.0, "max_cfl": 0.01})
    with pytest.raises(CFLViolation):
        generate(cfg, tmp_path, splits=["train"], verbose=False)


# -------------------------------------------------------------------- loading
def test_window_dataset(tiny_dataset):
    ds = TrajectoryWindowDataset(tiny_dataset, "train", n_in=2, n_out=3)
    assert len(ds) == 5 * (6 - 5 + 1)
    s = ds[1]  # trajectory 0, start 1
    assert s["input"].shape == (2, 16, 16) and s["target"].shape == (3, 16, 16)
    raw = load_split(tiny_dataset, "train")["vorticity"]
    np.testing.assert_allclose(ds.normalizer.decode(s["target"]).numpy(), raw[0, 3:6],
                               rtol=1e-5, atol=1e-5)
    assert float(s["nu"]) == pytest.approx(float(load_split(tiny_dataset, "train")["nu"][0]),
                                           rel=1e-6)


def test_window_dataset_never_crosses_trajectories(tiny_dataset):
    ds = TrajectoryWindowDataset(tiny_dataset, "val", n_in=1, n_out=5)
    assert len(ds) == 2
    assert [int(ds[k]["traj"]) for k in range(2)] == [0, 1]


def test_recomputed_stats_helper(tiny_dataset):
    stats = compute_stats(tiny_dataset / "train.h5", 16)
    assert stats["vorticity_std"] > 0


def test_per_split_frame_count_override(tmp_path):
    cfg = tiny_config()
    cfg.split("val").n_frames = 9
    generate(cfg, tmp_path, splits=["train", "val"], verbose=False)
    with h5py.File(tmp_path / "val.h5") as f, h5py.File(tmp_path / "train.h5") as g:
        assert f["vorticity_16"].shape[1] == 9 and len(f["times"]) == 9
        assert g["vorticity_16"].shape[1] == 6
    assert "| 9 |" in (tmp_path / "DATASHEET.md").read_text()


def test_spectral_resample_up_and_down_exact_for_band_limited_fields():
    from nfds.data.transforms import spectral_resample

    def field(n):
        x, y = SpectralGrid(n).coords
        return torch.as_tensor(np.sin(2 * x + y) + 0.4 * np.cos(3 * y - x))

    for n_in, n_out in ((16, 64), (64, 16), (32, 48), (48, 32)):
        np.testing.assert_allclose(spectral_resample(field(n_in), n_out).numpy(),
                                   field(n_out).numpy(), atol=1e-12)


def test_adding_a_split_keeps_existing_ones_in_manifest(tmp_path):
    cfg = tiny_config()
    generate(cfg, tmp_path, splits=["train", "val"], verbose=False)
    cfg2 = tiny_config()
    cfg2.splits.append(copy.deepcopy(cfg2.split("val")))
    cfg2.splits[-1].name = "test_new"
    generate(cfg2, tmp_path, splits=["train", "test_new"], verbose=False)
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert set(files) == {"train.h5", "val.h5", "test_new.h5"}
    # changing a split's settings invalidates its old entry
    cfg3 = tiny_config()
    cfg3.split("val").n_traj = 3
    generate(cfg3, tmp_path, splits=["train"], verbose=False)
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert "val.h5" not in files
