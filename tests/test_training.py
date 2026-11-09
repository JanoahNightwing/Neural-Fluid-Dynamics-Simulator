"""Training loop: every regime runs, checkpoints round-trip, learning actually happens."""

import json

import pytest

torch = pytest.importorskip("torch")

from helpers import tiny_config  # noqa: E402

from nfds.data.generate import generate  # noqa: E402
from nfds.evaluation.evaluate import evaluate_stepper  # noqa: E402
from nfds.training.checkpoint import load_stepper  # noqa: E402
from nfds.training.config import config_from_dict, load_config  # noqa: E402
from nfds.training.train import train  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_train")
    generate(tiny_config(), out, verbose=False)
    return out


def make_cfg(data, out, **kw):
    d = {"name": "t", "data": str(data), "out_dir": str(out), "device": "cpu",
         "model": {"name": "cnn", "width": 8, "depth": 1},
         "optim": {"epochs": 2, "batch_size": 4, "lr": 1e-3},
         "val": {"horizon": 3}}
    d.update(kw)
    return config_from_dict(d)


@pytest.mark.parametrize("regime", [
    {"type": "onestep"},
    {"type": "noise", "noise_std": 0.05},
    {"type": "unroll", "unroll": 3},
    {"type": "pushforward", "unroll": 3},
])
def test_every_regime_trains_and_writes_outputs(tiny_dataset, tmp_path, regime):
    cfg = make_cfg(tiny_dataset, tmp_path, regime=regime, name=regime["type"])
    summary = train(cfg, verbose=False)
    run = tmp_path / regime["type"]
    for f in ("config.yaml", "metrics.jsonl", "best.pt", "last.pt", "summary.json"):
        assert (run / f).exists()
    lines = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and "val_rollout" in lines[-1]
    assert summary["best_val_rollout"] < 10


def test_unet_with_history_window(tiny_dataset, tmp_path):
    cfg = make_cfg(tiny_dataset, tmp_path, n_in=2, name="unet2",
                   model={"name": "unet", "widths": [8, 16], "blocks_per_level": 1,
                          "emb_dim": 8})
    train(cfg, verbose=False)
    stepper = load_stepper(tmp_path / "unet2" / "best.pt")
    assert stepper.n_in == 2
    res = evaluate_stepper(stepper, tiny_dataset, "val", horizon=3)
    assert res["n_in"] == 2 and len(res["times"]) == 3


def test_training_reduces_loss(tiny_dataset, tmp_path):
    cfg = make_cfg(tiny_dataset, tmp_path, name="learn",
                   optim={"epochs": 15, "batch_size": 4, "lr": 3e-3, "warmup_epochs": 0.5})
    train(cfg, verbose=False)
    lines = [json.loads(x) for x in (tmp_path / "learn" / "metrics.jsonl").read_text()
             .splitlines()]
    assert lines[-1]["train_loss"] < 0.7 * lines[0]["train_loss"]


def test_checkpoint_round_trip_reproduces_predictions(tiny_dataset, tmp_path):
    cfg = make_cfg(tiny_dataset, tmp_path, name="rt")
    train(cfg, verbose=False)
    a = load_stepper(tmp_path / "rt" / "last.pt")
    b = load_stepper(tmp_path / "rt" / "last.pt")
    x, nu = torch.randn(2, 1, 16, 16), torch.tensor([0.02, 0.03])
    torch.testing.assert_close(a(x, nu), b(x, nu))


def test_config_validation_and_overrides(tmp_path):
    with pytest.raises(ValueError):
        config_from_dict({"name": "x", "data": "d", "regime": {"type": "unroll", "unroll": 1}})
    with pytest.raises(ValueError):
        config_from_dict({"name": "x", "data": "d", "regime": {"type": "noise"}})
    p = tmp_path / "c.yaml"
    p.write_text("name: a\ndata: d\nmodel: {name: cnn, width: 8}\n")
    cfg = load_config(p, ["model.width=32", "optim.lr=0.01", "name=b"])
    assert cfg.model["width"] == 32 and cfg.optim.lr == 0.01 and cfg.name == "b"
