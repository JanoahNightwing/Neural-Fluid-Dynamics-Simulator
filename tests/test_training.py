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
    log = (run / "metrics.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(x) for x in log.splitlines()]
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
    log = (tmp_path / "learn" / "metrics.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(x) for x in log.splitlines()]
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
    p.write_text("name: a\ndata: d\nmodel: {name: cnn, width: 8}\n", encoding="utf-8")
    cfg = load_config(p, ["model.width=32", "optim.lr=0.01", "name=b"])
    assert cfg.model["width"] == 32 and cfg.optim.lr == 0.01 and cfg.name == "b"


def test_training_on_a_subset_of_trajectories(tiny_dataset, tmp_path):
    cfg = make_cfg(tiny_dataset, tmp_path, name="subset", train_trajectories=[0, 2],
                   optim={"epochs": 1, "batch_size": 4})
    summary = train(cfg, verbose=False)
    assert summary["train_trajectories"] == 2
    with pytest.raises(ValueError):
        train(make_cfg(tiny_dataset, tmp_path, name="bad", train_trajectories=[0, 99]),
              verbose=False)


def test_fine_tuning_keeps_architecture_and_normalisation(tiny_dataset, tmp_path):
    base = make_cfg(tiny_dataset, tmp_path, name="base", n_in=2,
                    model={"name": "cnn", "width": 8, "depth": 1})
    train(base, verbose=False)
    ft = make_cfg(tiny_dataset, tmp_path, name="ft", init_checkpoint=str(tmp_path / "base/best.pt"),
                  model={"name": "unet"},  # ignored when fine-tuning
                  train_split="test_re", train_trajectories=[0, 1],
                  val_split="test_re", val_trajectories=[1, 2],
                  optim={"epochs": 1, "batch_size": 2, "lr": 1e-4})
    train(ft, verbose=False)
    a, b = (torch.load(tmp_path / f"{r}/last.pt", weights_only=False) for r in ("base", "ft"))
    assert b["model_cfg"] == a["model_cfg"] and b["n_in"] == 2
    assert (b["nu_log_mean"], b["nu_log_std"]) == (a["nu_log_mean"], a["nu_log_std"])
    assert any(not torch.equal(a["model_state"][k], b["model_state"][k])
               for k in a["model_state"])  # weights actually changed


def test_max_batches_per_epoch_from_config(tiny_dataset, tmp_path):
    cfg = make_cfg(tiny_dataset, tmp_path, name="cap", max_batches_per_epoch=1)
    train(cfg, verbose=False)
    lines = (tmp_path / "cap" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


@pytest.mark.parametrize("regime", [{"type": "onestep"}, {"type": "unroll", "unroll": 3}])
def test_micro_batches_give_the_full_batch_gradient(tiny_dataset, tmp_path, regime):
    from torch.utils.data import DataLoader

    from nfds.data.dataset import TrajectoryWindowDataset
    from nfds.models import build_model
    from nfds.training.train import NuNormalizer, accumulate_gradients

    unroll = regime.get("unroll", 1)
    ds = TrajectoryWindowDataset(tiny_dataset, "train", n_in=1, n_out=unroll)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    nu_norm = NuNormalizer.from_values(ds.meta["nu"])
    grads, losses = [], []
    for mb in (None, 2):
        cfg = make_cfg(tiny_dataset, tmp_path, regime=regime,
                       optim={"batch_size": 8, "micro_batch_size": mb})
        torch.manual_seed(0)
        model = build_model("cnn", n_in=1, width=8, depth=1).double()
        torch.nn.init.normal_(model.proj.weight, std=0.1)  # zero-init would give zero grads
        b = {k: (v.double() if v.is_floating_point() else v) for k, v in batch.items()}
        losses.append(accumulate_gradients(model, b, nu_norm, cfg, torch.device("cpu"), False))
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert losses[0] == pytest.approx(losses[1], rel=1e-10)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-9, atol=1e-12)


def test_micro_batch_size_must_divide_batch_size(tiny_dataset, tmp_path):
    with pytest.raises(ValueError, match="micro_batch_size"):
        make_cfg(tiny_dataset, tmp_path, optim={"batch_size": 8, "micro_batch_size": 3})
