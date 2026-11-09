"""Spectral operators must be exact (to round-off) on resolved Fourier modes."""

import numpy as np
import pytest

from nfds.diagnostics import energy, energy_spectrum
from nfds.grid import SpectralGrid
from nfds.initial_conditions import random_field


@pytest.fixture
def grid():
    return SpectralGrid(32)


def test_wavenumbers_are_integers_on_2pi_box(grid):
    assert grid.kx[0, 1] == 1 and grid.ky[1, 0] == 1 and grid.ky[-1, 0] == -1


def test_odd_n_rejected():
    with pytest.raises(ValueError):
        SpectralGrid(33)


def test_derivatives_exact(grid):
    x, y = grid.coords
    f = np.sin(3 * x) * np.cos(2 * y)
    f_hat = grid.fft(f)
    np.testing.assert_allclose(grid.ifft(grid.ddx(f_hat)), 3 * np.cos(3 * x) * np.cos(2 * y),
                               atol=1e-12)
    np.testing.assert_allclose(grid.ifft(grid.ddy(f_hat)), -2 * np.sin(3 * x) * np.sin(2 * y),
                               atol=1e-12)
    np.testing.assert_allclose(grid.ifft(grid.laplacian(f_hat)), -13 * f, atol=1e-11)


def test_poisson_inverse(grid):
    x, y = grid.coords
    psi = np.cos(x + 2 * y) + 0.5 * np.sin(4 * x)
    w = -grid.ifft(grid.laplacian(grid.fft(psi)))
    np.testing.assert_allclose(grid.ifft(grid.streamfunction(grid.fft(w))), psi, atol=1e-12)


def test_velocity_is_divergence_free_and_recovers_vorticity(grid):
    w = random_field(grid, k_peak=4, seed=0)
    u, v = grid.velocity(grid.fft(w))
    assert np.max(np.abs(grid.divergence(u, v))) < 1e-12
    np.testing.assert_allclose(grid.vorticity_from_velocity(u, v), w, atol=1e-11)


def test_spectrum_sums_to_energy(grid):
    w_hat = grid.fft(random_field(grid, k_peak=4, energy=0.7, seed=1))
    _, e_k = energy_spectrum(grid, w_hat)
    assert np.isclose(e_k.sum(), energy(grid, w_hat), rtol=1e-12)
    assert np.isclose(energy(grid, w_hat), 0.7, rtol=1e-12)


def test_dealias_mask_removes_top_third():
    g = SpectralGrid(48)  # n/2 = 24, cut = 16
    assert g.mask[0, 15] and not g.mask[0, 16]
    assert g.mask[15, 0] and not g.mask[16, 0] and not g.mask[-16, 0] and g.mask[-15, 0]


def test_random_field_reproducible(grid):
    a = random_field(grid, seed=3)
    b = random_field(grid, seed=3)
    c = random_field(grid, seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert abs(a.mean()) < 1e-14


def test_resolution_ratio_flags_grid_scale_energy(grid):
    from nfds.diagnostics import resolution_ratio

    smooth = grid.fft(random_field(grid, k_peak=2, seed=0))
    rough = smooth + grid.truncate(grid.fft(np.random.default_rng(0).standard_normal((32, 32))))
    assert resolution_ratio(grid, smooth) < 1e-5
    assert resolution_ratio(grid, rough) > 1e-3
