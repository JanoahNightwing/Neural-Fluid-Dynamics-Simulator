"""Evaluation: metric definitions, rollout mechanics, baselines."""


import pytest

torch = pytest.importorskip("torch")

from nfds.evaluation import metrics as M  # noqa: E402
from nfds.evaluation.evaluate import evaluate_stepper, make_baseline  # noqa: E402
from nfds.evaluation.rollout import rollout  # noqa: E402
from nfds.evaluation.steppers import PersistenceStepper, Stepper  # noqa: E402
from nfds.solver.spectral_torch import TorchSpectralGrid  # noqa: E402


def test_metrics_on_known_inputs():
    true = torch.randn(3, 4, 16, 16, dtype=torch.float64)
    assert torch.allclose(M.relative_l2(true, true), torch.zeros(3, 4, dtype=torch.float64))
    assert torch.allclose(M.correlation(true, true), torch.ones(3, 4, dtype=torch.float64))
    assert torch.allclose(M.correlation(-true, true), -torch.ones(3, 4, dtype=torch.float64))
    assert torch.allclose(M.relative_l2(2 * true, true), torch.ones(3, 4, dtype=torch.float64))


def test_high_correlation_time_and_censoring():
    corr = torch.tensor([[0.99, 0.9, 0.7, 0.95], [0.99, 0.95, 0.9, 0.85],
                         [0.9, float("nan"), 0.9, 0.9]])
    t, cens = M.high_correlation_time(corr, dt=0.5, threshold=0.8)
    assert t.tolist() == [1.0, 2.0, 0.5]
    assert cens.tolist() == [False, True, False]


def test_failure_step_detects_nan_and_blow_up():
    g = TorchSpectralGrid(16)
    x, y = g.coords()
    base = torch.sin(x) * torch.sin(y)
    pred = base.expand(3, 5, 16, 16).clone()
    pred[1, 2] = float("nan")
    pred[2, 3] *= 10  # energy x100
    true_e = M.field_energy(base.expand(3, 1, 16, 16), g)[:, 0]
    assert M.failure_step(pred, true_e, g).tolist() == [5, 2, 3]


class ShiftStepper(Stepper):
    """Toy stepper with n_in = 2 that returns the sum of its two input frames."""

    name, n_in = "shift", 2

    def __call__(self, frames, nu):
        return frames[:, 0] + frames[:, 1]


def test_rollout_feeds_back_predictions_with_window():
    traj = torch.zeros(2, 10, 4, 4, dtype=torch.float64)
    traj[:, 0], traj[:, 1] = 1.0, 1.0  # Fibonacci when fed back
    pred = rollout(ShiftStepper(), traj, torch.ones(2), horizon=4, batch_size=1)
    assert pred[0, :, 0, 0].tolist() == [2.0, 3.0, 5.0, 8.0]
    assert pred.shape == (2, 4, 4, 4)


def test_persistence_rollout_is_constant():
    traj = torch.randn(3, 6, 8, 8)
    pred = rollout(PersistenceStepper(), traj, torch.ones(3), horizon=5)
    assert torch.equal(pred, traj[:, :1].expand(-1, 5, -1, -1))


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from helpers import tiny_config

    from nfds.data.generate import generate

    out = tmp_path_factory.mktemp("tiny_eval")
    generate(tiny_config(), out, verbose=False)
    return out


def test_coarse_solver_at_simulation_resolution_reproduces_data(tiny_dataset):
    """At the grid the data was simulated on, the baseline is the same solver: near-exact."""
    solver = make_baseline("coarse_solver", tiny_dataset, "test_ic", n=32)
    pers = make_baseline("persistence", tiny_dataset, "test_ic", n=32)
    r_s = evaluate_stepper(solver, tiny_dataset, "test_ic", horizon=3, resolution=32)
    r_p = evaluate_stepper(pers, tiny_dataset, "test_ic", horizon=3, resolution=32)
    assert r_s["rel_l2_mean_over_horizon"] < 1e-4 < r_p["rel_l2_mean_over_horizon"]
    assert r_s["failure_fraction"] == 0.0
    assert len(r_s["rel_l2"]["mean"]) == r_s["horizon"] == 3
    assert sum(r_s["spectrum_true_last"]) > 0


def test_coarse_solver_has_truncation_floor_on_coarser_grid(tiny_dataset):
    """A 2/3-dealiased solver on an n² grid keeps only |k| < n/3, so stored modes between
    n/3 and n/2 are lost immediately: a constant error floor, not a bug."""
    solver = make_baseline("coarse_solver", tiny_dataset, "val", n=16)
    rel = evaluate_stepper(solver, tiny_dataset, "val", horizon=3)["rel_l2"]["mean"]
    assert rel[0] > 0.05 and max(rel) / min(rel) < 1.5


def test_evaluate_cli_resolution_transfer_label(tiny_dataset, tmp_path):
    from nfds.evaluation.evaluate import main

    main(["--data", str(tiny_dataset), "--split", "test_ic", "--baseline", "persistence",
          "--resolution", "32", "--horizon", "2", "--out", str(tmp_path)])
    main(["--data", str(tiny_dataset), "--split", "test_ic", "--baseline", "persistence",
          "--horizon", "2", "--out", str(tmp_path)])
    import json

    hi = json.loads((tmp_path / "persistence__test_ic@32.json").read_text())
    lo = json.loads((tmp_path / "persistence__test_ic.json").read_text())
    assert hi["resolution"] == 32 and lo["resolution"] == 16 and hi["train_resolution"] == 16
