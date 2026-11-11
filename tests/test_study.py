"""Study orchestration: plan expansion, resumable execution, failure handling, aggregation."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import tiny_config  # noqa: E402

from nfds.data.generate import generate  # noqa: E402
from nfds.study.aggregate import aggregate, bootstrap_ci, paired_difference  # noqa: E402
from nfds.study.plan import build_plan, parse_run_name  # noqa: E402
from nfds.study.run import execute  # noqa: E402


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("study_data")
    cfg = tiny_config()
    cfg.split("test_re").n_traj = 6
    generate(cfg, out, verbose=False)
    return out


@pytest.fixture
def model_configs(tmp_path):
    paths = {}
    for name, model in (("cnn", "{name: cnn, width: 4, depth: 1}"),
                        ("fno", "{name: fno, width: 4, modes: 4, n_layers: 1, proj_width: 8}")):
        p = tmp_path / f"{name}.yaml"
        p.write_text(f"name: {name}\ndata: x\nmodel: {model}\n"
                     "optim: {epochs: 1, batch_size: 4}\nval: {horizon: 2}\ndevice: cpu\n",
                     encoding="utf-8")
        paths[name] = str(p)
    return paths


def plan_dict(data, tmp_path, models):
    return {
        "name": "tiny", "data": str(data), "runs_dir": str(tmp_path / "runs"),
        "results_dir": str(tmp_path / "results"), "models": models,
        "experiments": {
            "regimes": {"models": ["cnn", "fno"], "seeds": [0, 1],
                        "regimes": {"onestep": {"type": "onestep"},
                                    "pushforward2": {"type": "pushforward", "unroll": 2}}},
            "scaling": {"models": ["fno"], "regimes": {"onestep": {"type": "onestep"}},
                        "train_trajectories": [2, 4], "seeds": [0], "evaluate": ["val"]},
            "fewshot": {"from": {"experiment": "regimes", "regime": "pushforward2", "seed": 0},
                        "models": ["fno"], "split": "test_re", "split_trajectories": 6,
                        "shots": [1, 2], "val_trajectories": 1,
                        "overrides": ["optim.batch_size=2"]},
        },
        "evaluation": {"horizon": 3, "splits": ["val", "test_ic"],
                       "resolution_transfer": {"split": "test_ic", "resolutions": [32]},
                       "long": {"split": "test_long", "horizon": 4}},  # missing split
        "speed": {"split": "test_ic", "runs": "regimes-*-onestep-*-s0",
                  "solver_resolutions": [8, 16], "horizon": 2, "batch": 2},
    }


def test_plan_expansion_and_names(tiny_data, tmp_path, model_configs):
    plan = build_plan(plan_dict(tiny_data, tmp_path, model_configs))
    names = [r.name for r in plan.runs]
    assert len(names) == 2 * 2 * 2 + 2 + 2
    assert "regimes-fno-pushforward2-nall-s1" in names
    assert "scaling-fno-onestep-n4-s0" in names and "fewshot-fno-pushforward2-k2-s0" in names
    labels = [e.label for r in plan.runs for e in r.evals] + [e.label for e in plan.baseline_evals]
    assert not any(set(lab) & set('<>:"/\\|?*') for lab in labels)  # valid Windows file names
    for n in names:
        info = parse_run_name(n)
        assert info["model"] in ("cnn", "fno")
    fs = plan.run("fewshot-fno-pushforward2-k1-s0")
    assert fs.init_from == "regimes-fno-pushforward2-nall-s0"
    assert fs.evals[0].trajectories == [3, 6]  # after max shots (2) + val (1)
    base = plan.run("regimes-fno-pushforward2-nall-s0")
    assert "test_re[3-6]" in [e.label for e in base.evals]  # zero-shot reference
    assert [e.label for e in plan.run("scaling-fno-onestep-n2-s0").evals] == ["val"]


def test_plan_rejects_bad_names_and_missing_base(tiny_data, tmp_path, model_configs):
    d = plan_dict(tiny_data, tmp_path, model_configs)
    d["experiments"]["regimes"]["regimes"]["one-step"] = {"type": "onestep"}
    with pytest.raises(ValueError):
        build_plan(d)
    d = plan_dict(tiny_data, tmp_path, model_configs)
    d["experiments"]["fewshot"]["from"]["regime"] = "unroll9"
    with pytest.raises(KeyError):
        build_plan(d)


def test_execute_resume_and_aggregate(tiny_data, tmp_path, model_configs):
    plan = build_plan(plan_dict(tiny_data, tmp_path, model_configs))
    counts = execute(plan, device="cpu", verbose=False)
    assert counts["trained"] == len(plan.runs) and counts["failed"] == 0
    res = plan.results_dir
    run_eval = res / "runs" / "regimes-cnn-onestep-nall-s0"
    assert {p.name for p in run_eval.iterdir()} == {"val.json", "test_ic.json",
                                                    "test_ic@32.json"}  # test_long missing
    assert (res / "runs" / "fewshot-fno-pushforward2-k2-s0" / "test_re[3-6].json").exists()
    assert (res / "runs" / "regimes-fno-pushforward2-nall-s0" / "test_re[3-6].json").exists()
    assert (res / "baselines" / "coarse_solver" / "test_ic@32.json").exists()
    assert (res / "pareto.json").exists()
    fs = json.loads((res / "runs" / "fewshot-fno-pushforward2-k1-s0" / "test_re[3-6].json")
                    .read_text(encoding="utf-8"))
    assert fs["n_trajectories"] == 3

    again = execute(plan, device="cpu", verbose=False)  # resume: nothing left to do
    assert again["trained"] == 0 and again["evaluated"] == 0

    report = aggregate(plan)
    text = report.read_text(encoding="utf-8")
    assert "## Main results" in text and "regimes" not in text.split("## Main")[0]
    table = (res / "summary" / "main_table.md").read_text(encoding="utf-8")
    assert "| cnn | pushforward2 | 2 |" in table and "| coarse_solver | - |" in table
    effects = (res / "summary" / "regime_effects.md").read_text(encoding="utf-8")
    assert "pushforward2" in effects
    for fig in ("regime_rollouts.png", "fewshot.png", "scaling.png"):
        assert (res / "summary" / "figures" / fig).exists()


def test_failed_run_is_logged_and_study_continues(tiny_data, tmp_path, model_configs):
    models = {**model_configs, "cnn": str(tmp_path / "missing.yaml")}
    d = plan_dict(tiny_data, tmp_path, models)
    d["experiments"] = {"regimes": {"models": ["cnn", "fno"], "seeds": [0],
                                    "regimes": {"onestep": {"type": "onestep"}}}}
    d.pop("speed")
    plan = build_plan(d)
    counts = execute(plan, device="cpu", verbose=False)
    assert counts["failed"] == 1 and counts["trained"] == 1
    failures = json.loads((plan.results_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures[0]["what"] == "train regimes-cnn-onestep-nall-s0"


def test_statistics_helpers():
    x = np.full(50, 2.0)
    assert bootstrap_ci(x) == (2.0, 2.0)
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    d, lo, hi = paired_difference(a + 1.0, a)
    assert d == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    d, lo, hi = paired_difference(a + 0.1 * rng.normal(size=200), a)
    assert lo < 0.05 and hi > -0.05
