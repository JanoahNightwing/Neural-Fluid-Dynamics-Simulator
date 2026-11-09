"""Aggregate a study's results into tables, figures and a Markdown report.

    python -m nfds.study.aggregate configs/study/m5_main.yaml

Writes to <results_dir>/summary/: REPORT.md, main_table.md/.csv, regime_effects.md and
figures (regime rollouts, error vs Reynolds number, data scaling, long-run statistics,
few-shot adaptation; the Pareto plot comes from the run step).

Statistics: table cells are mean ± standard deviation over seeds of each seed's test-set
mean. Confidence intervals are 95% bootstrap intervals over test trajectories, using each
trajectory's value averaged over seeds. Regime effects are paired: the same test
trajectories under two training regimes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from nfds.study.plan import StudyPlan, load_plan, parse_run_name  # noqa: E402

# (column title, test-set label, key in the result JSON, format, lower is better)
METRICS = [
    ("test_id 1-step L2", "test_id", "rel_l2_first_step", "{:.3f}", True),
    ("test_id corr. time", "test_id", "high_corr_time_mean", "{:.2f}", False),
    ("test_id mean L2", "test_id", "rel_l2_mean_over_horizon", "{:.3f}", True),
    ("interp corr. time", "test_interp", "high_corr_time_mean", "{:.2f}", False),
    ("Re-low corr. time", "test_re_low", "high_corr_time_mean", "{:.2f}", False),
    ("Re-high corr. time", "test_re_high", "high_corr_time_mean", "{:.2f}", False),
    ("struct. IC corr. time", "test_ic", "high_corr_time_mean", "{:.2f}", False),
    ("2× grid 1-step L2", "test_id@128", "rel_l2_first_step", "{:.3f}", True),
    ("PDE residual ratio", "test_id", "pde_residual_ratio", "{:.2f}", None),
    ("long: energy ratio", "test_long", "late_energy_ratio", "{:.2f}", None),
    ("long: vorticity W1", "test_long", "late_vorticity_w1", "{:.3f}", True),
    ("long: failures", "test_long", "failure_fraction", "{:.0%}", True),
]


# Experiments whose runs appear together in the main table, regime effects and figures.
MAIN_EXPERIMENTS = ("regimes", "hybrid")


def _exps(experiment: str) -> tuple[str, ...]:
    return MAIN_EXPERIMENTS if experiment == "regimes" else (experiment,)


# ------------------------------------------------------------------ statistics


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_difference(a: np.ndarray, b: np.ndarray, **kw) -> tuple[float, float, float]:
    """Mean of (a - b) over paired samples with a bootstrap CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    lo, hi = bootstrap_ci(d, **kw)
    return float(d.mean()) if len(d) else math.nan, lo, hi


# ---------------------------------------------------------------------- loading


def load_records(plan: StudyPlan) -> list[dict[str, Any]]:
    recs = []
    for kind in ("runs", "baselines"):
        root = plan.results_dir / kind
        if not root.exists():
            continue
        for owner_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(owner_dir.glob("*.json")):
                res = json.loads(f.read_text())
                info = parse_run_name(owner_dir.name) if kind == "runs" else {
                    "experiment": "baseline", "model": owner_dir.name, "regime": "-",
                    "seed": 0}
                recs.append({"owner": owner_dir.name, "kind": kind, "label": f.stem,
                             "res": res, **info})
    return recs


def group(recs: list[dict], **match) -> dict[tuple, list[dict]]:
    """Group records matching ``match`` by (model, regime)."""
    out = defaultdict(list)
    for r in recs:
        if all(r.get(k) == v for k, v in match.items()):
            out[(r["model"], r["regime"])].append(r)
    return out


def seed_values(rs: list[dict], key: str) -> list[float]:
    vals = [r["res"].get(key) for r in rs]
    return [v for v in vals if v is not None and math.isfinite(v)]


def per_traj_seed_mean(rs: list[dict], key: str) -> np.ndarray | None:
    """Per-trajectory values averaged over seeds (all seeds see the same test set)."""
    arrs = [np.array(r["res"]["per_trajectory"][key], dtype=float) for r in rs
            if "per_trajectory" in r["res"] and key in r["res"]["per_trajectory"]]
    if not arrs or len({len(a) for a in arrs}) != 1:
        return None
    return np.nanmean(np.stack(arrs), axis=0)


def fmt_cell(vals: list[float], f: str) -> str:
    if not vals:
        return "–"
    m = float(np.mean(vals))
    if len(vals) == 1:
        return f.format(m)
    return f"{f.format(m)} ± {f.format(float(np.std(vals, ddof=1)))}"


# ------------------------------------------------------------------------ tables


def main_table(recs: list[dict], experiment: str = "regimes") -> tuple[str, list[dict]]:
    labels = {r["label"] for r in recs}
    cols = [m for m in METRICS if m[1] in labels]
    rows_md, rows_csv = [], []
    keys = sorted(group([r for r in recs if r["experiment"] in _exps(experiment)]))
    baselines = sorted({r["model"] for r in recs if r["kind"] == "baselines"})
    for kind, entries in (("baselines", [(b, "-") for b in baselines]), ("runs", keys)):
        for model, regime in entries:
            cells, row = [], {"model": model, "regime": regime}
            seeds = set()
            for title, label, key, f, _ in cols:
                if kind == "baselines":
                    rs = [r for r in recs if r["kind"] == "baselines" and r["model"] == model
                          and r["label"] == label]
                else:
                    rs = [r for r in recs if r["experiment"] in _exps(experiment)
                          and r["model"] == model and r["regime"] == regime
                          and r["label"] == label]
                seeds |= {r["seed"] for r in rs}
                vals = seed_values(rs, key)
                cells.append(fmt_cell(vals, f))
                row[title] = float(np.mean(vals)) if vals else None
                if title == "test_id corr. time" and kind == "runs":
                    pt = per_traj_seed_mean(rs, "high_corr_time")
                    if pt is not None:
                        lo, hi = bootstrap_ci(pt)
                        cells[-1] += f" [{lo:.2f}, {hi:.2f}]"
            row["seeds"] = len(seeds) if kind == "runs" else None
            rows_md.append(f"| {model} | {regime} | {row['seeds'] or '–'} | " + " | ".join(cells)
                           + " |")
            rows_csv.append(row)
    head = "| Model | Regime | Seeds | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + " --- |" * (3 + len(cols))
    note = ("Mean ± std over seeds; test_id correlation time also shows a 95% bootstrap CI "
            "over test trajectories. Correlation times in simulation time units "
            "(threshold ρ = 0.8). PDE residual ratio: prediction residual / truth residual "
            "(1 = as consistent as the data). Long-run metrics over the second half of the "
            "test_long rollouts.")
    return "\n".join([head, sep, *rows_md]) + "\n\n" + note + "\n", rows_csv


def regime_effects(recs: list[dict], experiment: str = "regimes",
                   reference: str = "onestep", label: str = "test_id",
                   key: str = "high_corr_time") -> str:
    g = group([r for r in recs if r["experiment"] in _exps(experiment) and r["label"] == label])
    lines = [f"| Model | Regime | Δ {key} vs {reference} (paired, 95% CI) | Significant |",
             "| --- | --- | --- | --- |"]
    for model in sorted({m for m, _ in g}):
        ref = per_traj_seed_mean(g.get((model, reference), []), key)
        if ref is None:
            continue
        for (m, regime), rs in sorted(g.items()):
            if m != model or regime == reference:
                continue
            other = per_traj_seed_mean(rs, key)
            if other is None or len(other) != len(ref):
                continue
            d, lo, hi = paired_difference(other, ref)
            sig = "yes" if (lo > 0 or hi < 0) else "no"
            lines.append(f"| {model} | {regime} | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {sig} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- figures


def best_regime(recs: list[dict], model: str, experiment: str = "regimes",
                label: str = "test_id") -> str | None:
    """Regime with the longest mean high-correlation time on the primary test set."""
    g = group([r for r in recs if r["experiment"] in _exps(experiment) and r["label"] == label
               and r["model"] == model])
    scores = {reg: np.mean(seed_values(rs, "high_corr_time_mean")) for (_, reg), rs in g.items()
              if seed_values(rs, "high_corr_time_mean")}
    return max(scores, key=scores.get) if scores else None


def fig_regime_rollouts(recs: list[dict], out: Path, experiment: str = "regimes",
                        label: str = "test_id") -> bool:
    g = group([r for r in recs if r["experiment"] in _exps(experiment) and r["label"] == label])
    models = sorted({m for m, _ in g})
    if not models:
        return False
    base = {r["model"]: r["res"] for r in recs if r["kind"] == "baselines"
            and r["label"] == label}
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False,
                             constrained_layout=True, sharey=True)
    for ax, model in zip(axes[0], models, strict=True):
        for (m, regime), rs in sorted(g.items()):
            if m != model:
                continue
            curves = np.array([r["res"]["rel_l2"]["mean"] for r in rs], dtype=float)
            t = np.array(rs[0]["res"]["times"])
            mean = np.nanmean(curves, 0)
            line, = ax.plot(t, mean, label=f"{regime} ({len(rs)} seeds)")
            if len(rs) > 1:
                sd = np.nanstd(curves, 0, ddof=1)
                ax.fill_between(t, mean - sd, mean + sd, color=line.get_color(), alpha=0.15)
        for name, style in (("coarse_solver", "k--"), ("persistence", "k:")):
            if name in base:
                ax.plot(base[name]["times"], base[name]["rel_l2"]["mean"], style, lw=1,
                        label=name)
        ax.set(title=model, xlabel="rollout time", ylim=(0, 1.5))
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel(f"relative L2 error ({label})")
    fig.suptitle("Training regime vs rollout error (mean ± std over seeds)")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def training_nu_range(plan: StudyPlan) -> dict | None:
    path = Path(plan.data) / "config.yaml"
    if not path.exists():
        return None
    cfg = yaml.safe_load(path.read_text())
    return cfg.get("splits", {}).get("train", {}).get("nu")


def fig_reynolds(recs: list[dict], plan: StudyPlan, out: Path,
                 experiment: str = "regimes", primary: str = "test_id") -> bool:
    splits = ["test_re_low", "test_id", "test_interp", "test_re_high"]
    models = sorted({r["model"] for r in recs if r["experiment"] in _exps(experiment)})
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    entries = [(m, best_regime(recs, m, experiment, primary), "runs") for m in models]
    entries.append(("coarse_solver", "-", "baselines"))
    for model, regime, kind in entries:
        res_list, hct_list = [], []
        for split in splits:
            rs = [r for r in recs if r["label"] == split and r["model"] == model
                  and r["kind"] == kind and (kind == "baselines" or (
                      r["experiment"] in _exps(experiment) and r["regime"] == regime))]
            pt = per_traj_seed_mean(rs, "high_corr_time")
            if pt is None:
                continue
            res_list.append(1.0 / np.array(rs[0]["res"]["per_trajectory"]["nu"]))
            hct_list.append(pt)
        if not res_list:
            continue
        re_all, h_all = np.concatenate(res_list), np.concatenate(hct_list)
        order = np.argsort(re_all)
        re_all, h_all = re_all[order], h_all[order]
        bins = np.unique(np.quantile(np.log(re_all), np.linspace(0, 1, 9)))
        idx = np.clip(np.digitize(np.log(re_all), bins) - 1, 0, len(bins) - 2)
        centers = [np.exp(np.log(re_all[idx == b]).mean()) for b in range(len(bins) - 1)
                   if (idx == b).any()]
        means = [h_all[idx == b].mean() for b in range(len(bins) - 1) if (idx == b).any()]
        style = {"color": "k", "ls": "--"} if kind == "baselines" else {}
        line, = ax.plot(centers, means, marker="o", label=f"{model} ({regime})"
                        if kind == "runs" else model, **style)
        ax.scatter(re_all, h_all, s=6, alpha=0.25, color=line.get_color())
    nu = training_nu_range(plan)
    if nu and nu.get("low"):
        ax.axvspan(1 / nu["high"], 1 / nu["low"], color="tab:green", alpha=0.08,
                   label="training range")
        if nu.get("exclude"):
            ax.axvspan(1 / nu["exclude"][1], 1 / nu["exclude"][0], facecolor="white",
                       edgecolor="tab:green", hatch="//", alpha=0.6,
                       label="held-out band (test_interp)")
    ax.set(xscale="log", xlabel="Reynolds number Re = 1/ν",
           ylabel="time with correlation > 0.8",
           title="Generalisation across Reynolds number (points: trajectories; line: bin mean)")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def fig_scaling(recs: list[dict], out: Path, experiment: str = "scaling",
                label: str = "test_id") -> bool:
    rs = [r for r in recs if r["experiment"] in _exps(experiment) and r["label"] == label]
    if not rs:
        return False
    by = defaultdict(lambda: defaultdict(list))
    for r in rs:
        by[(r["model"], r["regime"])][r["n_train"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for (model, regime), d in sorted(by.items()):
        ns = np.array(sorted(d), dtype=float)
        l2 = np.array([np.mean(seed_values(d[n], "rel_l2_first_step") or [np.nan])
                       for n in sorted(d)])
        hct = np.array([np.mean(seed_values(d[n], "high_corr_time_mean") or [np.nan])
                        for n in sorted(d)])
        ok = np.isfinite(l2) & (l2 > 0)
        slope = (f", slope {np.polyfit(np.log(ns[ok]), np.log(l2[ok]), 1)[0]:.2f}"
                 if ok.sum() >= 2 else "")
        axes[0].plot(ns, l2, "o-", label=f"{model} ({regime}){slope}")
        axes[1].plot(ns, hct, "o-", label=f"{model} ({regime})")
    axes[0].set(xscale="log", yscale="log", xlabel="training trajectories",
                ylabel=f"{label} one-step relative L2",
                title="One-step error vs data (log-log slope in legend)")
    axes[1].set(xscale="log", xlabel="training trajectories",
                ylabel=f"{label} time with correlation > 0.8", title="Rollout skill vs data")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def fig_long(recs: list[dict], out: Path, experiment: str = "regimes",
             primary: str = "test_id") -> bool:
    rs = [r for r in recs if r["label"] == "test_long"]
    if not rs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    truth = None
    names, ratios = [], []
    models = sorted({r["model"] for r in rs if r["kind"] == "runs"})
    entries = [(m, best_regime(recs, m, experiment, primary), "runs") for m in models]
    entries += [(b, "-", "baselines") for b in sorted({r["model"] for r in rs
                                                       if r["kind"] == "baselines"})]
    for model, regime, kind in entries:
        sel = [r for r in rs if r["model"] == model and r["kind"] == kind
               and (kind == "baselines" or r["regime"] == regime)]
        if not sel:
            continue
        res = sel[0]["res"]
        k = np.array(res["spectrum_k"])[1:]
        fail = np.mean(seed_values(sel, "failure_fraction") or [np.nan])
        label = model if kind == "baselines" else f"{model} ({regime})"
        if fail > 0:
            label += f", {fail:.0%} failed"
        specs = [np.array(r["res"]["late_spectrum_pred"], float)[1:] for r in sel
                 if r["res"].get("late_spectrum_pred")]
        if specs:
            if res.get("late_spectrum_true") and model == "coarse_solver":
                truth = np.array(res["late_spectrum_true"])[1:]
            style = {"color": "k", "ls": "--"} if model == "coarse_solver" else (
                {"color": "0.6", "ls": ":"} if model == "persistence" else {})
            axes[0].loglog(k, np.clip(np.nanmean(specs, 0), 1e-14, None), label=label, **style)
        names.append(label)
        ratios.append(np.mean(seed_values(sel, "late_energy_ratio") or [np.nan]))
    if truth is not None:
        axes[0].loglog(k, truth, color="tab:red", lw=2.5, alpha=0.5, label="ground truth")
        axes[0].set_ylim(bottom=max(1e-12, truth[truth > 0].min() / 100))
    axes[0].set(xlabel="wavenumber k", ylabel="E(k)",
                title="Time-averaged spectrum, second half (non-failed rollouts)")
    axes[0].legend(fontsize=7)
    axes[1].barh(names, ratios)
    axes[1].axvline(1, color="k", lw=0.8)
    axes[1].set(xlabel="predicted / true mean energy (second half)",
                title="Long-run energy (1 = correct climate)")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def fig_fewshot(recs: list[dict], out: Path) -> bool:
    fs = [r for r in recs if r["experiment"] == "fewshot"]
    if not fs:
        return False
    label = fs[0]["label"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for model in sorted({r["model"] for r in fs}):
        pts = {r["shots"]: r["res"]["high_corr_time_mean"] for r in fs if r["model"] == model}
        regime = next(r["regime"] for r in fs if r["model"] == model)
        base = [r for r in recs if r["label"] == label and r["kind"] == "runs"
                and r["experiment"] != "fewshot" and r["model"] == model
                and r["regime"] == regime]
        if base:
            pts[0] = base[0]["res"]["high_corr_time_mean"]
        ks = sorted(pts)
        ax.plot(ks, [pts[k] for k in ks], "o-", label=model)
    for b in ("coarse_solver",):
        rb = [r for r in recs if r["kind"] == "baselines" and r["model"] == b
              and r["label"] == label]
        if rb:
            ax.axhline(rb[0]["res"]["high_corr_time_mean"], color="k", ls="--", lw=1, label=b)
    ax.set(xlabel="fine-tuning trajectories at the new Reynolds numbers (0 = zero-shot)",
           ylabel="time with correlation > 0.8", title=f"Few-shot adaptation on {label}")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------- report


def aggregate(plan: StudyPlan) -> Path:
    recs = load_records(plan)
    out = plan.results_dir / "summary"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    table_md, rows = main_table(recs)
    (out / "main_table.md").write_text(table_md)
    if rows:
        with open(out / "main_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [
                k for r in rows for k in r if k not in rows[0]])
            w.writeheader()
            w.writerows(rows)
    primary = plan.raw.get("evaluation", {}).get("splits", ["test_id"])[0]
    effects = regime_effects(recs, label=primary)
    (out / "regime_effects.md").write_text(effects)

    made = {
        "regime_rollouts.png": fig_regime_rollouts(recs, figs / "regime_rollouts.png",
                                                   label=primary),
        "reynolds.png": fig_reynolds(recs, plan, figs / "reynolds.png", primary=primary),
        "scaling.png": fig_scaling(recs, figs / "scaling.png", label=primary),
        "long_run.png": fig_long(recs, figs / "long_run.png", primary=primary),
        "fewshot.png": fig_fewshot(recs, figs / "fewshot.png"),
    }
    pareto = plan.results_dir / "figures" / "pareto.png"
    status = json.loads((plan.results_dir / "status.json").read_text()) \
        if (plan.results_dir / "status.json").exists() else {}
    failures = json.loads((plan.results_dir / "failures.json").read_text()) \
        if (plan.results_dir / "failures.json").exists() else []
    n_runs = len(plan.runs)
    n_done = sum((plan.runs_dir / r.name / "summary.json").exists() for r in plan.runs)
    lines = [f"# Study report: {plan.name}", "",
             f"Data: `{plan.data}`. Runs trained: {n_done}/{n_runs}. "
             f"Result files: {len(recs)}. Failures logged: {len(failures)}.", "",
             "## Main results", "", table_md, "## Effect of the training regime", "",
             "Paired comparison of each regime against one-step training on the same "
             f"{primary} trajectories (time with correlation > 0.8, averaged over seeds).", "",
             effects]
    for name, title in (("regime_rollouts.png", "Rollout error by training regime"),
                        ("reynolds.png", "Generalisation across Reynolds number"),
                        ("scaling.png", "Data scaling"),
                        ("long_run.png", "Long-run statistics"),
                        ("fewshot.png", "Few-shot adaptation")):
        if made[name]:
            lines += [f"## {title}", "", f"![{title}](figures/{name})", ""]
    if pareto.exists():
        lines += ["## Accuracy vs cost", "", "![Pareto](../figures/pareto.png)", ""]
    if status:
        lines += ["## Status", "", f"Last run: {status.get('updated', '?')}, "
                  f"counts: {status.get('counts')}", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    return out / "REPORT.md"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Aggregate a study into tables and figures")
    p.add_argument("plan")
    args = p.parse_args(argv)
    path = aggregate(load_plan(args.plan))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
