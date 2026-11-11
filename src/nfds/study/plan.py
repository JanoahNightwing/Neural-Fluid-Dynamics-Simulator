"""Study plans: one YAML file describes every training run and evaluation of a study.

A plan expands into a list of :class:`RunSpec` objects (training runs) and
:class:`EvalSpec` objects (one per run, or baseline, and test set). Names are
deterministic, so re-running a plan finds and skips completed work.

Run names: ``<experiment>-<model>-<regime>-n<train trajectories|all>-s<seed>``;
few-shot runs: ``fewshot-<model>-<base regime>-k<shots>-s<seed>``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalSpec:
    split: str
    horizon: int
    resolution: int | None = None  # None = training resolution
    trajectories: list[int] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.split + (f"@{self.resolution}" if self.resolution else "")


@dataclass
class RunSpec:
    name: str
    experiment: str
    model: str
    regime: str
    seed: int
    base_config: str
    overrides: list[str] = field(default_factory=list)
    n_train: int | None = None
    shots: int | None = None
    init_from: str | None = None  # run name whose best.pt initialises this run
    evals: list[EvalSpec] = field(default_factory=list)


@dataclass
class StudyPlan:
    name: str
    data: str
    runs_dir: Path
    results_dir: Path
    runs: list[RunSpec]
    baselines: list[str]
    baseline_evals: list[EvalSpec]
    speed: dict[str, Any] | None
    raw: dict[str, Any]

    def run(self, name: str) -> RunSpec:
        for r in self.runs:
            if r.name == name:
                return r
        raise KeyError(name)


def _regime_overrides(regime: dict[str, Any]) -> list[str]:
    out = [f"regime.type={regime['type']}"]
    out.append(f"regime.unroll={regime.get('unroll', 1)}")
    out.append(f"regime.noise_std={regime.get('noise_std', 0.0)}")
    return out


def _check_token(kind: str, value: str) -> None:
    if "-" in value or "/" in value or not value:
        raise ValueError(f"{kind} name {value!r} must be non-empty and contain no '-' or '/'")


def load_plan(path: str | Path, extra_overrides: list[str] | None = None) -> StudyPlan:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return build_plan(raw, extra_overrides)


def build_plan(raw: dict[str, Any], extra_overrides: list[str] | None = None) -> StudyPlan:
    raw = copy.deepcopy(raw)
    name, data = raw["name"], raw["data"]
    runs_dir = Path(raw.get("runs_dir", f"runs/{name}"))
    results_dir = Path(raw.get("results_dir", f"results/{name}"))
    models: dict[str, str] = raw["models"]
    for m in models:
        _check_token("model", m)
    global_over = list(raw.get("overrides", [])) + list(extra_overrides or [])
    model_over: dict[str, list[str]] = raw.get("model_overrides", {}) or {}

    ev = raw.get("evaluation", {})
    horizon = ev.get("horizon", 40)
    std_evals = [EvalSpec(s, horizon) for s in ev.get("splits", ["test_id"])]
    rt = ev.get("resolution_transfer")
    if rt:
        std_evals += [EvalSpec(rt["split"], horizon, resolution=r) for r in rt["resolutions"]]
    long = ev.get("long")
    if long:
        std_evals.append(EvalSpec(long["split"], long["horizon"], label=f"{long['split']}"))

    def common(model: str, seed: int) -> list[str]:
        return [f"data={data}", f"seed={seed}", f"out_dir={runs_dir}", *global_over,
                *model_over.get(model, [])]

    runs: list[RunSpec] = []
    exps = raw.get("experiments", {})
    for exp_name, exp in exps.items():
        if exp_name == "fewshot":
            continue
        _check_token("experiment", exp_name)
        for model in exp["models"]:
            if model not in models:
                raise KeyError(f"experiment {exp_name}: unknown model {model!r}")
            for regime_name, regime in exp["regimes"].items():
                _check_token("regime", regime_name)
                for n_train in exp.get("train_trajectories", [None]):
                    for seed in exp.get("seeds", [0]):
                        tag = "all" if n_train is None else str(n_train)
                        run_name = f"{exp_name}-{model}-{regime_name}-n{tag}-s{seed}"
                        over = [*common(model, seed), *_regime_overrides(regime),
                                f"name={run_name}"]
                        if n_train is not None:
                            over.append(f"train_trajectories=[0, {n_train}]")
                        evals = (std_evals if exp.get("evaluate", "all") == "all"
                                 else [e for e in std_evals if e.label in exp["evaluate"]])
                        runs.append(RunSpec(run_name, exp_name, model, regime_name, seed,
                                            models[model], over, n_train=n_train,
                                            evals=list(evals)))

    fewshot_evals: list[EvalSpec] = []
    fs = exps.get("fewshot")
    if fs:
        src = fs["from"]
        k_max, v = max(fs["shots"]), fs.get("val_trajectories", 8)
        n_split = fs["split_trajectories"]
        if k_max + v >= n_split:
            raise ValueError("few-shot: shots + val_trajectories must leave trajectories to test")
        eval_range = [k_max + v, n_split]
        fs_eval = EvalSpec(fs["split"], horizon, trajectories=eval_range,
                           # no ':' in the label: it becomes a file name, and Windows forbids ':'
                           label=f"{fs['split']}[{eval_range[0]}-{eval_range[1]}]")
        fewshot_evals.append(fs_eval)
        for model in fs["models"]:
            seed = src.get("seed", 0)
            base_name = (f"{src['experiment']}-{model}-{src['regime']}-"
                         f"n{src.get('n_train', 'all')}-s{seed}")
            for k in fs["shots"]:
                run_name = f"fewshot-{model}-{src['regime']}-k{k}-s{seed}"
                over = common(model, seed) + [
                    f"name={run_name}", f"train_split={fs['split']}",
                    f"train_trajectories=[0, {k}]", f"val_split={fs['split']}",
                    f"val_trajectories=[{k_max}, {k_max + v}]",
                    "regime.type=onestep", "regime.unroll=1", "regime.noise_std=0.0",
                    *fs.get("overrides", [])]
                runs.append(RunSpec(run_name, "fewshot", model, src["regime"], seed,
                                    models[model], over, shots=k, init_from=base_name,
                                    evals=[fs_eval]))
            # the base model, zero-shot, on the same held-out trajectories
            try:
                base = next(r for r in runs if r.name == base_name)
            except StopIteration as e:
                raise KeyError(f"few-shot base run {base_name!r} is not part of the plan") from e
            if fs_eval.label not in [e.label for e in base.evals]:
                base.evals.append(fs_eval)

    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate run names in plan")
    return StudyPlan(
        name=name, data=data, runs_dir=runs_dir, results_dir=results_dir, runs=runs,
        baselines=list(ev.get("baselines", ["persistence", "coarse_solver"])),
        baseline_evals=std_evals + fewshot_evals, speed=raw.get("speed"), raw=raw)


def parse_run_name(name: str) -> dict[str, Any]:
    """Inverse of the naming scheme above."""
    parts = name.split("-")
    if parts[0] == "fewshot":
        _, model, regime, k, s = parts
        return {"experiment": "fewshot", "model": model, "regime": regime,
                "shots": int(k[1:]), "seed": int(s[1:])}
    exp, model, regime, n, s = parts
    return {"experiment": exp, "model": model, "regime": regime,
            "n_train": None if n == "nall" else int(n[1:]), "seed": int(s[1:])}
