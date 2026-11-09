"""FNO correctness: the spectral convolution is the Fourier multiplier it claims to be,
resolution invariance holds, symmetries are as documented, and the model can learn."""

import math

import pytest

torch = pytest.importorskip("torch")

from nfds.models import build_model  # noqa: E402
from nfds.models.fno import SpectralConv2d, periodic_positional_encoding  # noqa: E402


def grid(n):
    x = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    return xx, yy


def single_channel_conv(modes, value_pos, value_neg):
    conv = SpectralConv2d(1, 1, modes)
    with torch.no_grad():
        conv.weight_pos.fill_(value_pos)
        conv.weight_neg.fill_(value_neg)
    return conv


@pytest.mark.parametrize(("kx", "ky", "block"), [(3, 2, "pos"), (3, -2, "neg"), (0, 5, "pos")])
def test_spectral_conv_is_the_fourier_multiplier(kx, ky, block):
    """cos(kx x + ky y) -> Re(R e^{i(kx x + ky y)}) = |R| cos(kx x + ky y + arg R)."""
    r_pos, r_neg = complex(0.5, 0.8), complex(-1.2, 0.3)
    conv = single_channel_conv(8, r_pos, r_neg)
    xx, yy = grid(32)
    phase = kx * xx + ky * yy
    out = conv(torch.cos(phase)[None, None])[0, 0]
    r = r_pos if block == "pos" else r_neg
    expected = abs(r) * torch.cos(phase + math.atan2(r.imag, r.real))
    if kx == 0:  # a kx = 0 mode appears at both (ky) and (-ky): both blocks contribute
        expected = 0.5 * (abs(r_pos) * torch.cos(phase + math.atan2(r_pos.imag, r_pos.real))
                          + abs(r_neg) * torch.cos(-phase + math.atan2(r_neg.imag, r_neg.real)))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_spectral_conv_drops_modes_beyond_cutoff():
    conv = single_channel_conv(8, 1.0, 1.0)
    xx, yy = grid(32)
    for field in (torch.cos(9 * xx), torch.cos(9 * yy), torch.sin(10 * xx + 3 * yy)):
        assert conv(field[None, None]).abs().max() < 1e-5


def test_spectral_conv_is_resolution_invariant():
    """Same weights, band-limited input sampled on two grids -> the same output function."""
    torch.manual_seed(0)
    conv = SpectralConv2d(2, 3, 6)

    def field(n):
        xx, yy = grid(n)
        return torch.stack([torch.sin(2 * xx + yy) + torch.cos(5 * yy),
                            torch.cos(3 * xx - 4 * yy)])[None]

    coarse, fine = conv(field(32)), conv(field(64))
    torch.testing.assert_close(fine[..., ::2, ::2], coarse, atol=1e-5, rtol=1e-5)


def test_spectral_conv_rejects_too_small_grid():
    with pytest.raises(ValueError):
        SpectralConv2d(1, 1, 12)(torch.randn(1, 1, 16, 16))


def test_positional_encoding_is_periodic_and_resolution_consistent():
    pe32, pe64 = periodic_positional_encoding(32, 32), periodic_positional_encoding(64, 64)
    torch.testing.assert_close(pe64[:, ::2, ::2], pe32)
    assert pe32.shape == (4, 32, 32)


@pytest.fixture
def small_fno():
    torch.manual_seed(0)
    m = build_model("fno", width=8, modes=6, n_layers=2, proj_width=16)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


@pytest.mark.parametrize("channel_mlp", [False, True])
def test_shape_any_resolution_and_persistence_start(channel_mlp):
    m = build_model("fno", n_in=2, width=8, modes=6, n_layers=2, proj_width=16,
                    channel_mlp=channel_mlp)
    for n in (16, 32, 48):
        x = torch.randn(3, 2, n, n)
        out = m(x, torch.randn(3))
        assert out.shape == (3, 1, n, n)
        torch.testing.assert_close(out, x[:, -1:])  # zero-initialised projection


def test_full_model_approximately_resolution_consistent(small_fno):
    """Pointwise non-linearities create modes above the cutoff, so the full model is only
    approximately resolution invariant: check the mismatch is small for a smooth input."""
    def field(n):
        xx, yy = grid(n)
        return (torch.sin(xx + 2 * yy) + 0.5 * torch.cos(3 * xx))[None, None]

    nu = torch.zeros(1)
    coarse, fine = small_fno(field(32), nu), small_fno(field(64), nu)
    rel = (fine[..., ::2, ::2] - coarse).norm() / coarse.norm()
    assert rel < 1e-2


def test_exactly_translation_equivariant_without_positional_encoding():
    torch.manual_seed(0)
    m = build_model("fno", width=8, modes=6, n_layers=2, proj_width=16,
                    positional_encoding=False)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x, nu = torch.randn(2, 1, 32, 32), torch.randn(2)
    shift = (5, -7)
    torch.testing.assert_close(m(torch.roll(x, shift, dims=(-2, -1)), nu),
                               torch.roll(m(x, nu), shift, dims=(-2, -1)), atol=1e-5, rtol=1e-5)


def test_positional_encoding_breaks_equivariance(small_fno):
    x, nu = torch.randn(1, 1, 32, 32), torch.zeros(1)
    shifted = small_fno(torch.roll(x, 5, dims=-2), nu)
    assert not torch.allclose(shifted, torch.roll(small_fno(x, nu), 5, dims=-2), atol=1e-4)


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    m = build_model("fno", width=16, modes=8, n_layers=3, proj_width=32)
    xx, yy = grid(32)
    x = torch.stack([torch.sin(k * xx + yy) for k in range(1, 5)])[:, None]
    y = torch.roll(x, 2, dims=-1)
    nu = torch.zeros(4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        loss = ((m(x, nu) - y) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * first


def test_neuralop_reference_adapter():
    pytest.importorskip("neuralop")
    m = build_model("neuralop_fno", n_in=1, width=8, n_modes=12, n_layers=2)
    x, nu = torch.randn(2, 1, 32, 32), torch.randn(2)
    assert m(x, nu).shape == (2, 1, 32, 32)
    # state dict round trip (the library adds a "_metadata" entry)
    m2 = build_model("neuralop_fno", n_in=1, width=8, n_modes=12, n_layers=2)
    m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m2.eval()(x, nu), m.eval()(x, nu))
