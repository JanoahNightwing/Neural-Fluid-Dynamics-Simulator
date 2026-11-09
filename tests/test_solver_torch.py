"""The PyTorch solver must reproduce the verified NumPy reference to round-off."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nfds.diagnostics import energy, energy_spectrum, enstrophy  # noqa: E402
from nfds.forcing import KolmogorovForcing  # noqa: E402
from nfds.grid import SpectralGrid  # noqa: E402
from nfds.initial_conditions import random_field, taylor_green, taylor_green_exact  # noqa: E402
from nfds.solver import SpectralVorticitySolver  # noqa: E402
from nfds.solver.spectral_torch import TorchSpectralGrid, TorchVorticitySolver  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


# ------------------------------------------------------------------ grid
def test_grid_matches_numpy(device):
    g_np, g_t = SpectralGrid(32), TorchSpectralGrid(32, device=device)
    for name in ("kx", "ky", "k2", "inv_k2"):
        np.testing.assert_array_equal(getattr(g_t, name).cpu().numpy(), getattr(g_np, name))
    np.testing.assert_array_equal(g_t.mask.cpu().numpy().astype(bool), g_np.mask)


def test_batched_diagnostics_match_numpy():
    g_np, g_t = SpectralGrid(32), TorchSpectralGrid(32)
    ws = np.stack([random_field(g_np, seed=s) for s in range(3)])
    w_hat_t = g_t.fft(g_t.as_tensor(ws))
    for i in range(3):
        w_hat = g_np.fft(ws[i])
        assert math.isclose(g_t.energy(w_hat_t)[i].item(), energy(g_np, w_hat), rel_tol=1e-12)
        assert math.isclose(g_t.enstrophy(w_hat_t)[i].item(), enstrophy(g_np, w_hat),
                            rel_tol=1e-12)
        np.testing.assert_allclose(g_t.energy_spectrum(w_hat_t)[i].numpy(),
                                   energy_spectrum(g_np, w_hat)[1], rtol=1e-10, atol=1e-18)


# ------------------------------------------------------------- equivalence
@pytest.mark.parametrize("scheme", ["ifrk2", "ifrk4"])
def test_single_step_matches_numpy(scheme, device):
    g_np, g_t = SpectralGrid(64), TorchSpectralGrid(64, device=device)
    f = KolmogorovForcing().vorticity(g_np)
    w0 = random_field(g_np, seed=1)
    ref = SpectralVorticitySolver(g_np, 1e-3, 0.1, f, scheme).step(g_np.truncate(g_np.fft(w0)),
                                                                   0.01)
    solver = TorchVorticitySolver(g_t, 1e-3, 0.1, f, scheme)
    out = solver.step(solver.initial_state(w0), 0.01)[0].cpu().numpy()
    assert rel(out, ref) < 1e-13


def test_batch_with_different_viscosities_matches_separate_numpy_runs(device):
    g_np, g_t = SpectralGrid(32), TorchSpectralGrid(32, device=device)
    nus = [1e-3, 5e-3, 2e-2]
    w0 = np.stack([random_field(g_np, seed=s) for s in range(3)])
    res = TorchVorticitySolver(g_t, nus).simulate(w0, 1.0, 0.5, dt=0.01)
    assert res.vorticity.shape == (3, 3, 32, 32)
    for i, nu in enumerate(nus):
        ref = SpectralVorticitySolver(g_np, nu).simulate(w0[i], 1.0, 0.5, dt=0.01)
        assert rel(res.vorticity[i].numpy(), ref.vorticity) < 1e-12


def test_adaptive_stepping_matches_numpy():
    """Same CFL logic -> same number of steps and the same answer."""
    g_np, g_t = SpectralGrid(32), TorchSpectralGrid(32)
    w0 = random_field(g_np, seed=4)
    ref = SpectralVorticitySolver(g_np, 1e-3).simulate(w0, 1.0, 0.25, cfl=0.5)
    res = TorchVorticitySolver(g_t, 1e-3).simulate(w0, 1.0, 0.25, cfl=0.5)
    assert res.n_steps == ref.n_steps
    assert rel(res.vorticity[0].numpy(), ref.vorticity) < 1e-12


# ---------------------------------------------------------- exact physics
def test_taylor_green_exact(device):
    g_np, g_t = SpectralGrid(32), TorchSpectralGrid(32, device=device)
    res = TorchVorticitySolver(g_t, 0.05, drag=0.1).simulate(taylor_green(g_np), 2.0, 0.5, dt=0.05)
    for t, w in zip(res.times.tolist(), res.vorticity[0], strict=True):
        assert rel(w.numpy(), taylor_green_exact(g_np, t, 0.05, 0.1)) < 1e-12


def test_inviscid_conservation():
    g_np, g_t = SpectralGrid(64), TorchSpectralGrid(64)
    res = TorchVorticitySolver(g_t, 0.0).simulate(random_field(g_np, seed=0), 1.0, 0.5, dt=0.005)
    w_hat = g_t.fft(res.vorticity[0])
    e, z = g_t.energy(w_hat), g_t.enstrophy(w_hat)
    assert (e / e[0] - 1).abs().max() < 1e-8
    assert (z / z[0] - 1).abs().max() < 1e-7


# -------------------------------------------------------- differentiability
def test_gradient_matches_finite_difference():
    """d E(t=0.5) / d omega_0 along a random direction: autograd vs central difference."""
    g_np, g_t = SpectralGrid(16), TorchSpectralGrid(16)
    solver = TorchVorticitySolver(g_t, 1e-2)
    w0 = g_t.as_tensor(random_field(g_np, k_peak=2, seed=0)).requires_grad_(True)
    direction = g_t.as_tensor(random_field(g_np, k_peak=2, seed=1))

    def final_energy(w):
        w_hat = solver.initial_state(w)
        for _ in range(25):
            w_hat = solver.step(w_hat, 0.02)
        return g_t.energy(w_hat).sum()

    final_energy(w0).backward()
    autograd = float((w0.grad * direction).sum())
    eps = 1e-6
    with torch.no_grad():
        fd = (final_energy(w0 + eps * direction) - final_energy(w0 - eps * direction)) / (2 * eps)
    assert math.isclose(autograd, float(fd), rel_tol=1e-6)


def test_simulate_requires_grad_keeps_graph():
    g_t = TorchSpectralGrid(16)
    nu = torch.tensor([1e-2], dtype=torch.float64, requires_grad=True)
    solver = TorchVorticitySolver(g_t, nu)
    w0 = g_t.as_tensor(random_field(SpectralGrid(16), k_peak=2, seed=0))
    res = solver.simulate(w0, 0.2, 0.1, dt=0.02, requires_grad=True)
    loss = (res.vorticity[:, -1] ** 2).mean()
    loss.backward()
    assert nu.grad is not None and nu.grad.item() < 0  # more viscosity -> less enstrophy


# ------------------------------------------------------------- precision
def test_float32_close_to_float64():
    g_np = SpectralGrid(64)
    w0 = random_field(g_np, seed=2)
    r64 = TorchVorticitySolver(TorchSpectralGrid(64), 1e-3).simulate(w0, 1.0, 0.5, dt=0.01)
    g32 = TorchSpectralGrid(64, dtype=torch.float32)
    r32 = TorchVorticitySolver(g32, 1e-3).simulate(w0, 1.0, 0.5, dt=0.01)
    assert r32.vorticity.dtype == torch.float32
    assert rel(r32.vorticity.double().numpy(), r64.vorticity.numpy()) < 1e-5


def test_batch_size_mismatch_raises():
    g_t = TorchSpectralGrid(16)
    w0 = np.zeros((2, 16, 16))
    with pytest.raises(ValueError):
        TorchVorticitySolver(g_t, [1e-3, 2e-3, 3e-3]).simulate(w0, 0.1, 0.1)


def test_blow_up_is_detected():
    g_t = TorchSpectralGrid(32)
    w0 = random_field(SpectralGrid(32), seed=0, energy=50.0)
    with pytest.raises(FloatingPointError):
        TorchVorticitySolver(g_t, 0.0).simulate(w0, 5.0, 1.0, dt=1.0)
