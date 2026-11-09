"""Model architecture tests: shapes, initial identity, symmetries, learnability."""

import pytest

torch = pytest.importorskip("torch")

from nfds.models import build_model, count_parameters  # noqa: E402

CONFIGS = [("cnn", {"width": 16, "depth": 2}),
           ("unet", {"widths": [8, 16, 32], "blocks_per_level": 1, "emb_dim": 16})]


@pytest.mark.parametrize(("name", "kw"), CONFIGS)
@pytest.mark.parametrize("n_in", [1, 3])
def test_output_shape(name, kw, n_in):
    m = build_model(name, n_in=n_in, **kw)
    out = m(torch.randn(4, n_in, 32, 32), torch.randn(4))
    assert out.shape == (4, 1, 32, 32)
    assert count_parameters(m) > 0


@pytest.mark.parametrize(("name", "kw"), CONFIGS)
def test_starts_as_persistence(name, kw):
    """Zero-initialised output layer: an untrained model returns its last input frame."""
    m = build_model(name, n_in=2, **kw)
    x = torch.randn(3, 2, 32, 32)
    torch.testing.assert_close(m(x, torch.randn(3)), x[:, -1:])


def _perturb(m):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return m


def test_cnn_is_exactly_translation_equivariant():
    m = _perturb(build_model("cnn", width=16, depth=2)).eval()
    x, nu = torch.randn(2, 1, 32, 32), torch.randn(2)
    shifted = m(torch.roll(x, shifts=(5, -3), dims=(-2, -1)), nu)
    torch.testing.assert_close(shifted, torch.roll(m(x, nu), shifts=(5, -3), dims=(-2, -1)),
                               atol=1e-5, rtol=1e-5)


def test_unet_equivariant_to_shifts_by_its_downsample_factor():
    m = _perturb(build_model("unet", widths=[8, 16, 32], blocks_per_level=1,
                             emb_dim=16)).eval()
    x, nu = torch.randn(2, 1, 32, 32), torch.randn(2)
    s = m.downsample_factor  # 4 for three levels
    shifted = m(torch.roll(x, shifts=(s, 2 * s), dims=(-2, -1)), nu)
    torch.testing.assert_close(shifted, torch.roll(m(x, nu), shifts=(s, 2 * s), dims=(-2, -1)),
                               atol=1e-5, rtol=1e-5)


def test_unet_rejects_incompatible_grid():
    m = build_model("unet", widths=[8, 16, 32], blocks_per_level=1, emb_dim=16)
    with pytest.raises(ValueError):
        m(torch.randn(1, 1, 30, 30), torch.zeros(1))


@pytest.mark.parametrize(("name", "kw"), CONFIGS)
def test_viscosity_conditioning_changes_output(name, kw):
    m = _perturb(build_model(name, **kw)).eval()
    x = torch.randn(1, 1, 32, 32)
    assert not torch.allclose(m(x, torch.tensor([-1.0])), m(x, torch.tensor([1.0])))


@pytest.mark.parametrize(("name", "kw"), CONFIGS)
def test_can_overfit_a_single_batch(name, kw):
    """A learnable map (a fixed periodic shift) must be fit to low error quickly."""
    torch.manual_seed(0)
    m = build_model(name, **kw)
    x = torch.randn(8, 1, 32, 32)
    y = torch.roll(x, shifts=1, dims=-1)
    nu = torch.zeros(8)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = ((m(x, nu) - y) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_unknown_model_name():
    with pytest.raises(KeyError):
        build_model("transformer")
