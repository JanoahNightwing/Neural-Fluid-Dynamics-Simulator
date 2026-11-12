"""Run (or resume) a study.

    python -m nfds.study.run configs/study/m5_main.yaml            # everything, resumable
    python -m nfds.study.run configs/study/m5_main.yaml --list     # show the plan and status
    python -m nfds.study.run configs/study/m5_main.yaml --only "regimes-fno-*"
    python -m nfds.study.run configs/study/m5_dev.yaml optim.epochs=1   # extra overrides

Completed work is detected from files on disk (a run's summary.json, an evaluation's
JSON), so the command can be interrupted and restarted at any time; a run stopped mid-training
continues from its last finished epoch. A run that fails is
recorded in <results>/failures.json and the study continues with the next one.

Layout:
    <runs_dir>/<run>/                      checkpoints and logs (from the training loop)
    <results_dir>/runs/<run>/<label>.json  evaluation of a run on one test set
    <results_dir>/baselines/<name>/<label>.json
    <results_dir>/pareto.json              accuracy-vs-cost benchmark (optional)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import torch

from nfds.evaluation.evaluate import evaluate_stepper, make_baseline
from nfds.evaluation.metrics import jsonable
from nfds.study.plan import EvalSpec, RunSpec, StudyPlan, load_plan
from nfds.training.checkpoint import load_stepper
from nfds.training.config import config_from_dict, load_config
from nfds.training.train import train


def run_done(plan: StudyPlan, run: RunSpec) -> bool:
    return (plan.runs_dir / run.name / "summary.json").exists()


def eval_path(plan: StudyPlan, owner: str, spec: EvalSpec, baseline: bool = False) -> Path:
    group = "baselines" if baseline else "runs"
    return plan.results_dir / group / owner / f"{spec.label}.json"


def split_available(plan: StudyPlan, spec: EvalSpec) -> bool:
    path = Path(plan.data) / f"{spec.split}.h5"
    if not path.exists():
        return False
    if spec.resolution is None:
        return True
    with h5py.File(path) as f:
        return f"vorticity_{spec.resolution}" in f


def train_config(plan: StudyPlan, run: RunSpec):
    overrides = list(run.overrides)
    if run.init_from:
        overrides.append(f"init_checkpoint={plan.runs_dir / run.init_from / 'best.pt'}")
    return load_config(run.base_config, overrides)


def log_failure(plan: StudyPlan, what: str, err: BaseException) -> None:
    path = plan.results_dir / "failures.json"
    failures = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    failures.append({"what": what, "error": f"{type(err).__name__}: {err}",
                     "traceback": traceback.format_exc(limit=5),
                     "time": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(failures, indent=1), encoding="utf-8")


def evaluate_one(plan: StudyPlan, stepper, owner: str, spec: EvalSpec, baseline: bool,
                 device: str) -> str:
    path = eval_path(plan, owner, spec, baseline)
    if path.exists():
        return "done"
    if not split_available(plan, spec):
        return "missing-split"
    res = evaluate_stepper(stepper, plan.data, spec.split, spec.horizon,
                           resolution=spec.resolution, trajectories=spec.trajectories)
    res.update({"name": owner, "label": spec.label, "train_resolution": None})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(res), indent=1, allow_nan=False), encoding="utf-8")
    return "new"


def status_table(plan: StudyPlan) -> list[dict]:
    rows = []
    for run in plan.runs:
        evals = [eval_path(plan, run.name, e).exists() for e in run.evals]
        rows.append({"run": run.name, "trained": run_done(plan, run),
                     "evaluated": f"{sum(evals)}/{len(evals)}"})
    return rows


def execute(plan: StudyPlan, only: str | None = None, device: str = "auto",
            skip_eval: bool = False, verbose: bool = True) -> dict:
    plan.results_dir.mkdir(parents=True, exist_ok=True)
    plan.runs_dir.mkdir(parents=True, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    selected = [r for r in plan.runs if only is None or fnmatch.fnmatch(r.name, only)]
    # train base runs before the few-shot runs that start from them
    selected.sort(key=lambda r: r.init_from is not None)
    counts = {"trained": 0, "skipped": 0, "failed": 0, "evaluated": 0}
    t_start = time.perf_counter()

    for i, run in enumerate(selected, 1):
        head = f"[{i}/{len(selected)}] {run.name}"
        if not run_done(plan, run):
            if run.init_from and not (plan.runs_dir / run.init_from / "best.pt").exists():
                print(f"{head}: base run {run.init_from} not trained yet, skipping")
                counts["skipped"] += 1
                continue
            if verbose:
                print(f"{head}: training", flush=True)
            try:
                cfg = train_config(plan, run)
                cfg = config_from_dict({**cfg.to_dict(), "device": dev})
                train(cfg, verbose=verbose, resume=True)
                counts["trained"] += 1
            except Exception as err:  # keep going; the failure is recorded
                print(f"{head}: FAILED ({type(err).__name__}: {err})")
                log_failure(plan, f"train {run.name}", err)
                counts["failed"] += 1
                continue
        elif verbose:
            print(f"{head}: already trained")
        if skip_eval:
            continue
        stepper = None
        for spec in run.evals:
            if eval_path(plan, run.name, spec).exists():
                continue
            try:
                stepper = stepper or load_stepper(plan.runs_dir / run.name / "best.pt",
                                                  device=dev, name=run.name)
                state = evaluate_one(plan, stepper, run.name, spec, False, dev)
                counts["evaluated"] += state == "new"
                if verbose and state != "done":
                    print(f"    eval {spec.label}: {state}", flush=True)
            except Exception as err:
                print(f"    eval {spec.label}: FAILED ({type(err).__name__}: {err})")
                log_failure(plan, f"eval {run.name} {spec.label}", err)
                counts["failed"] += 1

    if not skip_eval:  # baselines are cheap and cached: evaluate them in every session
        for name in plan.baselines:
            for spec in plan.baseline_evals:
                if eval_path(plan, name, spec, baseline=True).exists():
                    continue
                if not split_available(plan, spec):
                    continue
                try:
                    n = spec.resolution or json.loads(
                        (Path(plan.data) / "stats.json").read_text(encoding="utf-8"))["resolution"]
                    stepper = make_baseline(name, plan.data, spec.split, n, dev)
                    evaluate_one(plan, stepper, name, spec, True, dev)
                    counts["evaluated"] += 1
                    if verbose:
                        print(f"baseline {name} {spec.label}: done", flush=True)
                except Exception as err:
                    print(f"baseline {name} {spec.label}: FAILED ({err})")
                    log_failure(plan, f"baseline {name} {spec.label}", err)
                    counts["failed"] += 1
        if only is None and plan.speed and not (plan.results_dir / "pareto.json").exists():
            run_speed(plan, dev, verbose)

    counts["minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    (plan.results_dir / "status.json").write_text(json.dumps(
        {"counts": counts, "runs": status_table(plan),
         "updated": datetime.now(timezone.utc).isoformat()}, indent=1), encoding="utf-8")
    return counts


def run_speed(plan: StudyPlan, device: str, verbose: bool = True) -> None:
    from nfds.evaluation.speed import pareto, plot_pareto

    sp = plan.speed
    pattern = sp.get("runs", "*")
    ckpts = [str(plan.runs_dir / r.name / "best.pt") for r in plan.runs
             if fnmatch.fnmatch(r.name, pattern) and run_done(plan, r)]
    try:
        res = pareto(plan.data, sp.get("split", "test_id"), ckpts,
                     sp.get("solver_resolutions", [32, 48, 64, 96, 128]),
                     horizon=sp.get("horizon", 10), batch=sp.get("batch", 16),
                     max_trajectories=sp.get("max_trajectories", 32), device=device,
                     compile=sp.get("compile", False), bf16=sp.get("bf16", False))
        (plan.results_dir / "pareto.json").write_text(
            json.dumps(jsonable(res), indent=1, allow_nan=False), encoding="utf-8")
        plot_pareto(res, plan.results_dir / "figures" / "pareto.png")
        if verbose:
            print("speed benchmark: done")
    except Exception as err:
        print(f"speed benchmark: FAILED ({err})")
        log_failure(plan, "speed benchmark", err)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run or resume a study")
    p.add_argument("plan")
    p.add_argument("overrides", nargs="*", help="training overrides applied to every run")
    p.add_argument("--list", action="store_true", help="print the plan and its status, then exit")
    p.add_argument("--only", help="glob over run names, e.g. 'regimes-fno-*'")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args(argv)
    plan = load_plan(args.plan, args.overrides)
    if args.list:
        rows = status_table(plan)
        for r in rows:
            print(f"{'trained' if r['trained'] else 'pending':8} {r['evaluated']:>6}  {r['run']}")
        print(f"{len(rows)} runs, {sum(r['trained'] for r in rows)} trained; baselines: "
              f"{', '.join(plan.baselines)} on {len(plan.baseline_evals)} test sets")
        return
    counts = execute(plan, args.only, args.device, args.skip_eval)
    print(f"study {plan.name}: {counts}")
    print(f"next: python -m nfds.study.aggregate {args.plan}")


if __name__ == "__main__":
    main()
