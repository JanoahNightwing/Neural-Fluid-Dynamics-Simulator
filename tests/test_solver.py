"""Verification suite for the spectral vorticity solver (proposal section 2.4)."""

import numpy as np
import pytest

from nfds.diagnostics import energy, enstrophy, palinstrophy
from nfds.forcing import KolmogorovForcing
from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field, taylor_green, taylor_green_exact
from nfds.solver import SpectralVorticitySolver


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# --------------------------------------------------------------- operators
def test_advection_term_matches_analytic():
    """psi = a sin x + b sin 2y  =>  -u.grad(w) = 6ab cos x cos 2y."""
    g = SpectralGrid(32)
    x, y = g.coords
    a, b = 0.7, -1.3
    w = a * np.sin(x) + 4 * b * np.sin(2 * y)
    solver = SpectralVorticitySolver(g, nu=0.0)
    n_phys = g.ifft(solver.advection(g.fft(w)))
    np.testing.assert_allclose(n_phys, 6 * a * b * np.cos(x) * np.cos(2 * y), atol=1e-12)


# --------------------------------------------------------- exact solutions
@pytest.mark.parametrize("scheme", ["ifrk2", "ifrk4"])
@pytest.mark.parametrize("drag", [0.0, 0.1])
def test_taylor_green_decay(scheme, drag):
    g = SpectralGrid(32)
    nu, t_final = 0.05, 2.0
    solver = SpectralVorticitySolver(g, nu=nu, drag=drag, scheme=scheme)
    res = solver.simulate(taylor_green(g), t_final, save_dt=0.5, dt=0.05)
    for t, w in zip(res.times, res.vorticity, strict=True):
        assert rel_l2(w, taylor_green_exact(g, t, nu, drag)) < 1e-12


def test_kolmogorov_spin_up_to_laminar_state():
    """From rest, omega(t) = omega_s (1 - exp(-(nu k_f^2 + alpha) t)) exactly."""
    g = SpectralGrid(32)
    f = KolmogorovForcing(wavenumber=4)
    nu, drag = 0.05, 0.1
    solver = SpectralVorticitySolver(g, nu, drag, forcing_vorticity=f.vorticity(g))
    res = solver.simulate(np.zeros((32, 32)), 2.0, save_dt=0.5, dt=0.01)
    rate = nu * 16 + drag
    for t, w in zip(res.times[1:], res.vorticity[1:], strict=True):
        exact = f.laminar_vorticity(g, nu, drag) * (1 - np.exp(-rate * t))
        assert rel_l2(w, exact) < 1e-9


# ------------------------------------------------------------- invariants
def test_inviscid_conservation_of_energy_and_enstrophy():
    """The 2/3-truncated system conserves E and Z exactly when nu = alpha = 0.

    Any drift is therefore pure time-stepping error, which must vanish at
    least as fast as dt^4 for IF-RK4 (halving dt cuts it by >= 16x).
    """
    g = SpectralGrid(64)
    solver = SpectralVorticitySolver(g, nu=0.0)
    w0 = random_field(g, k_peak=4, energy=0.5, seed=0)

    def drift(dt):
        res = solver.simulate(w0, 1.0, save_dt=0.5, dt=dt)
        e = np.array([energy(g, g.fft(w)) for w in res.vorticity])
        z = np.array([enstrophy(g, g.fft(w)) for w in res.vorticity])
        return np.max(np.abs(e / e[0] - 1)), np.max(np.abs(z / z[0] - 1))

    (e1, z1), (e2, z2) = drift(0.01), drift(0.005)
    assert e2 < 1e-8 and z2 < 1e-7
    assert e1 / e2 > 16 and z1 / z2 > 16


def test_mean_vorticity_and_divergence_stay_zero():
    g = SpectralGrid(64)
    solver = SpectralVorticitySolver(g, nu=1e-3)
    res = solver.simulate(random_field(g, seed=2), 1.0, save_dt=0.5)
    for w in res.vorticity:
        assert abs(w.mean()) < 1e-14
        u, v = g.velocity(g.fft(w))
        assert np.max(np.abs(g.divergence(u, v))) < 1e-11


def test_energy_and_enstrophy_budgets_decaying():
    """dE/dt = -2 nu Z - 2 alpha E  and  dZ/dt = -2 nu P - 2 alpha Z."""
    g = SpectralGrid(64)
    nu, drag = 5e-3, 0.05
    solver = SpectralVorticitySolver(g, nu, drag)
    h = 2e-3
    res = solver.simulate(random_field(g, seed=5), 0.5, save_dt=h * 5, dt=h)
    w_hats = [g.fft(w) for w in res.vorticity]
    e = np.array([energy(g, w) for w in w_hats])
    z = np.array([enstrophy(g, w) for w in w_hats])
    p = np.array([palinstrophy(g, w) for w in w_hats])
    dt_save = res.times[1] - res.times[0]
    de = (e[2:] - e[:-2]) / (2 * dt_save)  # 2nd-order central difference
    dz = (z[2:] - z[:-2]) / (2 * dt_save)
    pred_e = -2 * nu * z[1:-1] - 2 * drag * e[1:-1]
    pred_z = -2 * nu * p[1:-1] - 2 * drag * z[1:-1]
    assert np.max(np.abs(de - pred_e)) / np.max(np.abs(pred_e)) < 1e-3
    assert np.max(np.abs(dz - pred_z)) / np.max(np.abs(pred_z)) < 1e-3


def test_forced_energy_budget():
    g = SpectralGrid(64)
    f = KolmogorovForcing()
    nu, drag = 5e-3, 0.1
    solver = SpectralVorticitySolver(g, nu, drag, forcing_vorticity=f.vorticity(g))
    fx, fy = f.velocity(g)
    res = solver.simulate(random_field(g, seed=6), 0.4, save_dt=0.01, dt=0.002)
    w_hats = [g.fft(w) for w in res.vorticity]
    e = np.array([energy(g, w) for w in w_hats])
    z = np.array([enstrophy(g, w) for w in w_hats])
    inj = []
    for w in w_hats:
        u, v = g.velocity(w)
        inj.append(np.mean(u * fx + v * fy))
    inj = np.array(inj)
    de = (e[2:] - e[:-2]) / (2 * 0.01)
    pred = -2 * nu * z[1:-1] - 2 * drag * e[1:-1] + inj[1:-1]
    assert np.max(np.abs(de - pred)) / np.max(np.abs(pred)) < 2e-3


# ---------------------------------------------------- convergence order
@pytest.mark.slow
@pytest.mark.parametrize(("scheme", "order"), [("ifrk2", 2), ("ifrk4", 4)])
def test_temporal_convergence_order(scheme, order):
    g = SpectralGrid(32)
    w0 = random_field(g, k_peak=3, energy=0.5, seed=7)
    t_final = 1.0

    def run(dt):
        s = SpectralVorticitySolver(g, nu=1e-2, scheme=scheme)
        return s.simulate(w0, t_final, save_dt=t_final, dt=dt).vorticity[-1]

    ref = SpectralVorticitySolver(g, nu=1e-2, scheme="ifrk4").simulate(
        w0, t_final, save_dt=t_final, dt=1 / 2000).vorticity[-1]
    dts = np.array([1 / 25, 1 / 50, 1 / 100])
    errs = np.array([rel_l2(run(dt), ref) for dt in dts])
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert abs(slope - order) < 0.25, (slope, errs)


@pytest.mark.slow
def test_spatial_convergence_spectral():
    """Errors fall much faster than any power of n for a smooth flow."""
    t_final, nu = 1.0, 0.05
    ref_g = SpectralGrid(128)
    # Build the IC on the finest grid and spectrally restrict it to coarser
    # ones, so every run starts from the same continuous field.
    w0_ref_hat = ref_g.fft(random_field(ref_g, k_peak=2, energy=0.05, seed=0))

    def restrict(w_hat_fine, n):
        g = SpectralGrid(n)
        out = np.zeros(g.spectral_shape, dtype=complex)
        h = n // 2
        out[:h, :h + 1] = w_hat_fine[:h, :h + 1]
        out[-h:, :h + 1] = w_hat_fine[-h:, :h + 1]
        out *= (n / ref_g.n) ** 2
        return g, out

    def run(n):
        g, w_hat = restrict(w0_ref_hat, n)
        s = SpectralVorticitySolver(g, nu=nu)
        w = s.simulate(g.ifft(w_hat), t_final, save_dt=t_final, dt=2e-3).vorticity[-1]
        return g, g.fft(w)

    _, ref_hat = run(128)
    errs = []
    for n in (16, 24, 32, 48):
        g, w_hat = run(n)
        _, ref_c = restrict(ref_hat, n)
        errs.append(np.linalg.norm(w_hat - ref_c) / np.linalg.norm(ref_c))
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    # Algebraic convergence would give roughly constant ratios; spectral
    # convergence gives ratios that grow with n.
    assert ratios[0] < ratios[1] < ratios[2], errs
    assert errs[-1] < 1e-4, errs
