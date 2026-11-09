"""Hybrid solver-in-the-loop model: starts as the coarse solver, learns through the solver."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import tiny_config  # noqa: E402

from nfds.data.dataset import load_stats  # noqa: E402
from nfds.data.generate import generate  # noqa: E402
from nfds.evaluation.evaluate import evaluate_stepper, make_baseline  # noqa: E402
from nfds.evaluation.steppers import NeuralStepper, NuNormalizer  # noqa: E402
from nfds.forcing import KolmogorovForcing  # noqa: E402
from nfds.grid import SpectralGrid  # noqa: E402
from nfds.initial_conditions import random_field  # noqa: E402
from nfds.models import build_model  # noqa: E402
from nfds.models.hybrid import PHYSICS_KEYS, hybrid_physics_kwargs  # noqa: E402
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver  # noqa: E402
from nfds.training.checkpoint import load_stepper  # noqa: E402
from nfds.training.config import config_from_dict  # noqa: E402
from nfds.training.train import train  # noqa: E402

PHYS = dict(drag=0.1, forcing_wavenumber=4, forcing_amplitude=1.0, save_dt=0.1,
            length=2 * math.pi, vorticity_mean=0.1, vorticity_std=2.0,
            nu_log_mean=math.log(5e-3), nu_log_std=0.4)


def hybrid(**kw):
    return build_model("hybrid", corrector="cnn", corrector_kwargs={"width": 8, "depth": 1},
                       solver_substeps=10, **{**PHYS, **kw})


def test_untrained_hybrid_is_exactly_the_coarse_solver():
    m = hybrid().eval()
    g_np = SpectralGrid(32)
    w = torch.as_tensor(np.stack([random_field(g_np, seed=s) for s in range(2)])).float()
    nu = torch.tensor([3e-3, 8e-3])
    nu_feat = (torch.log(nu) - PHYS["nu_log_mean"]) / PHYS["nu_log_std"]
    x = ((w - PHYS["vorticity_mean"]) / PHYS["vorticity_std"])[:, None]
    out = m(x, nu_feat)[:, 0] * PHYS["vorticity_std"] + PHYS["vorticity_mean"]

    grid = TorchSpectralGrid(32, dtype=torch.float32)
    solver = TorchVorticitySolver(grid, nu, drag=0.1,
                                  forcing_vorticity=KolmogorovForcing(4).vorticity(g_np))
    w_hat = solver.initial_state(w)
    for _ in range(10):
        w_hat = solver.step(w_hat, 0.01)
    torch.testing.assert_close(out, grid.ifft(w_hat), atol=1e-4, rtol=1e-4)


def test_gradients_flow_through_the_solver():
    m = hybrid(n_in=2)
    x = torch.randn(2, 2, 32, 32, requires_grad=True)
    loss = m(x, torch.zeros(2)).pow(2).mean()
    loss.backward()
    assert m.corrector.proj.weight.grad.abs().sum() > 0  # the corrector learns
    assert x.grad[:, -1].abs().sum() > 0  # gradient reaches the input through the solver
    # at initialisation the corrector's output layer is exactly zero (pure physics), so the
    # history channel only matters once that layer has moved off zero
    assert x.grad[:, 0].abs().sum() == 0
    with torch.no_grad():
        m.corrector.proj.weight.normal_(0, 0.01)
    x.grad = None
    m(x, torch.zeros(2)).pow(2).mean().backward()
    assert x.grad[:, 0].abs().sum() > 0


def test_unet_corrector_and_bad_corrector():
    m = build_model("hybrid", corrector="unet",
                    corrector_kwargs={"widths": [8, 16], "blocks_per_level": 1, "emb_dim": 8},
                    solver_substeps=4, **PHYS)
    assert m(torch.randn(1, 1, 32, 32), torch.zeros(1)).shape == (1, 1, 32, 32)
    with pytest.raises(ValueError):
        build_model("hybrid", corrector="transformer", **PHYS)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("hybrid_data")
    generate(tiny_config(), out, verbose=False)
    return out


def test_physics_read_from_dataset(tiny_dataset):
    kw = hybrid_physics_kwargs(str(tiny_dataset), "train", load_stats(tiny_dataset), -4.0, 0.5)
    assert set(kw) == set(PHYSICS_KEYS)
    assert kw["forcing_wavenumber"] == 2 and kw["save_dt"] == 0.05 and kw["drag"] == 0.1


def test_untrained_hybrid_matches_coarse_solver_baseline_in_evaluation(tiny_dataset):
    stats = load_stats(tiny_dataset)
    nu_norm = NuNormalizer(math.log(0.02), 0.5)
    kw = hybrid_physics_kwargs(str(tiny_dataset), "train", stats, nu_norm.log_mean,
                               nu_norm.log_std)
    m = build_model("hybrid", corrector="cnn", corrector_kwargs={"width": 4, "depth": 1},
                    solver_substeps=10, **kw)
    stepper = NeuralStepper(m, stats["vorticity_mean"], stats["vorticity_std"], nu_norm, 1)
    r_h = evaluate_stepper(stepper, tiny_dataset, "val", horizon=3, physics=False)
    r_s = evaluate_stepper(make_baseline("coarse_solver", tiny_dataset, "val", 16),
                           tiny_dataset, "val", horizon=3, physics=False)
    for a, b in zip(r_h["rel_l2"]["mean"], r_s["rel_l2"]["mean"], strict=True):
        assert a == pytest.approx(b, rel=1e-2, abs=1e-4)


def test_hybrid_trains_and_checkpoint_carries_physics(tiny_dataset, tmp_path):
    cfg = config_from_dict({
        "name": "hyb", "data": str(tiny_dataset), "out_dir": str(tmp_path), "device": "cpu",
        "model": {"name": "hybrid", "corrector": "cnn",
                  "corrector_kwargs": {"width": 4, "depth": 1}, "solver_substeps": 5},
        "regime": {"type": "unroll", "unroll": 2},
        "optim": {"epochs": 2, "batch_size": 4}, "val": {"horizon": 2}})
    summary = train(cfg, verbose=False)
    assert summary["best_val_rollout"] < 10
    ck = torch.load(tmp_path / "hyb" / "best.pt", weights_only=False)
    assert ck["model_cfg"]["forcing_wavenumber"] == 2 and ck["model_cfg"]["save_dt"] == 0.05
    assert ck["model_cfg"]["nu_log_mean"] == pytest.approx(ck["nu_log_mean"])
    stepper = load_stepper(tmp_path / "hyb" / "best.pt")
    out = stepper(torch.randn(2, 1, 16, 16, dtype=torch.float64), torch.tensor([0.02, 0.03]))
    assert out.shape == (2, 16, 16) and torch.isfinite(out).all()
