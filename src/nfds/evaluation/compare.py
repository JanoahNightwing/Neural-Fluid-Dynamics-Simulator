"""Compare evaluation results across models on one split: figure + Markdown table.

    python -m nfds.evaluation.compare results/forced_dev --split test_id \\
        --out figures/compare_dev_test_id.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE_STYLE = {"persistence": dict(color="0.6", ls=":"),
                  "coarse_solver": dict(color="k", ls="--")}


def load_results(root: str | Path, split: str) -> list[dict]:
    paths = sorted(Path(root).glob(f"*__{split}.json"))
    res = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    # baselines first, then models in name order
    return sorted(res, key=lambda r: (r["name"] not in BASELINE_STYLE, r["name"]))


def markdown_table(results: list[dict]) -> str:
    bands = list(results[0]["spectral_log_error_last"])
    head = ("| Model | 1-step rel. L2 | Mean rel. L2 (horizon) | High-corr. time (ρ>"
            f"{results[0]['high_corr_threshold']}) | Failures | "
            + " | ".join(f"log-spectrum err. {b}" for b in bands) + " | ms/step/traj |")
    sep = "|" + " --- |" * (6 + len(bands))
    rows = []
    for r in results:
        cens = r["high_corr_censored_fraction"]
        hct = f"{r['high_corr_time_mean']:.2f}" + (f" ({100 * cens:.0f}% censored)" if cens else "")
        mean_l2 = r["rel_l2_mean_over_horizon"]
        rows.append(
            f"| {r['name']} | {r['rel_l2_first_step']:.4f} | "
            f"{'diverged' if mean_l2 is None else f'{mean_l2:.3f}'} | {hct} | "
            f"{100 * r['failure_fraction']:.0f}% | "
            + " | ".join(f"{r['spectral_log_error_last'][b]:.2f}" for b in bands)
            + f" | {r['ms_per_step_per_trajectory']:.2f} |")
    return "\n".join([head, sep, *rows])


def plot(results: list[dict], out: str | Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for r in results:
        style = BASELINE_STYLE.get(r["name"], {})
        t = np.array(r["times"])
        for ax, key in ((axes[0, 0], "rel_l2"), (axes[0, 1], "correlation"),
                        (axes[1, 1], "energy_ratio")):
            m, s = np.array(r[key]["mean"]), np.array(r[key]["std"])
            line, = ax.plot(t, m, label=r["name"], **style)
            if key != "energy_ratio":
                ax.fill_between(t, m - s, m + s, color=line.get_color(), alpha=0.12)
        k = np.array(r["spectrum_k"])[1:]
        axes[1, 0].loglog(k, np.array(r["spectrum_pred_last"])[1:], label=r["name"], **style)
    ref = results[0]
    axes[1, 0].loglog(np.array(ref["spectrum_k"])[1:], np.array(ref["spectrum_true_last"])[1:],
                      color="tab:red", lw=2.5, alpha=0.5, label="ground truth")
    axes[0, 0].set(xlabel="time", ylabel="relative L2 error", title="Error vs rollout time",
                   ylim=(0, 1.5))
    axes[0, 1].set(xlabel="time", ylabel="correlation with truth",
                   title="Correlation vs rollout time", ylim=(-0.1, 1.02))
    axes[0, 1].axhline(ref["high_corr_threshold"], color="0.7", lw=0.8)
    axes[1, 0].set(xlabel="wavenumber k", ylabel="E(k)",
                   title="Energy spectrum, last quarter of the horizon")
    axes[1, 0].set_ylim(bottom=max(1e-12, min(ref["spectrum_true_last"][1:]) / 100))
    axes[1, 1].set(xlabel="time", ylabel="predicted / true energy", title="Energy ratio")
    axes[1, 1].axhline(1, color="0.7", lw=0.8)
    axes[1, 1].set_ylim(0, 2)
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle(title)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--split", default="test_id")
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    args = p.parse_args(argv)
    results = load_results(args.results, args.split)
    if not results:
        raise SystemExit(f"no results for split {args.split!r} in {args.results}")
    table = markdown_table(results)
    print(table)
    Path(args.out).with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    plot(results, args.out, args.title or f"{args.split}: {results[0]['n_trajectories']} "
         f"trajectories, {results[0]['resolution']}²")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
