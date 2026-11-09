#!/usr/bin/env python3
"""Thesis figure + LaTeX table: Flat PPO / Meta-PPO and PPO-RNN variants.

Uses normalized environment return mean ± sample standard deviation across seeds
(ddof=1 / ``statistics.stdev``), matching ``mean_std_across_seeds``.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ppo.configs import CRAFTAX_BUDGET_LABELS, CRAFTAX_BUDGET_STEPS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT_DIR = os.path.join(_PROJECT_ROOT, "runs/benchmark_reports/baseline_migration")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")
LATEX_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")

SERIES = (
    ("flat_512", "Flat PPO", "#2563EB", "-", "o"),
    ("meta_no_mask", "Meta-PPO", "#EA580C", "-", "s"),
    ("flat_rnn", "Flat PPO-RNN", "#7C3AED", "--", "^"),
    ("meta_rnn_no_mask", "Meta-PPO-RNN", "#059669", "--", "D"),
)

BUDGET_TICKS = (
    ("100k", "100K"),
    ("500k", "500K"),
    ("1m", "1M"),
    ("2m", "2M"),
    ("5m", "5M"),
    ("10m", "10M"),
    ("20m", "20M"),
)
PLOT_BUDGETS = tuple(b for b, _ in BUDGET_TICKS)

GREEDY_BOX_SERIES = (
    ("flat_512", "Flat PPO", "#2563EB"),
    ("meta_no_mask", "Meta-PPO", "#EA580C"),
    ("flat_rnn", "Flat PPO-RNN", "#7C3AED"),
    ("meta_rnn_no_mask", "Meta-PPO-RNN", "#059669"),
)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean ± sample std (ddof=1); std is 0.0 when n < 2."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def _collect(seeds: tuple[int, ...]) -> dict:
    """Aggregate normalized environment return (% of 22) across seeds."""
    out: dict[str, dict[str, dict]] = {pol: {} for pol, *_ in SERIES}

    for budget in CRAFTAX_BUDGET_LABELS:
        for pol, *_ in SERIES:
            last_vals: list[float] = []
            last_ret_vals: list[float] = []
            steps = CRAFTAX_BUDGET_STEPS[budget]

            for seed in seeds:
                path = os.path.join(REPORT_DIR, f"training_return_by_budget_seed{seed}.json")
                if not os.path.isfile(path):
                    continue
                data = _load(path)
                row = data.get("classic", {}).get(pol, {}).get(budget)
                if not row:
                    continue
                last_vals.append(float(row["last_update"]["return_pct_max"]))
                last_ret_vals.append(float(row["last_update"]["episode_return"]))
                steps = int(row["total_timesteps"])

            last_mean, last_std = _mean_std(last_vals)
            last_ret_mean, _ = _mean_std(last_ret_vals)
            out[pol][budget] = {
                "steps": steps,
                "last_pct": last_mean,
                "last_pct_std": last_std,
                "last_ret": last_ret_mean,
                "n_seeds": len(last_vals),
            }

    return out


def _best_policies(tail: dict[str, float]) -> list[str]:
    best = max(tail.values())
    return [k for k, v in tail.items() if abs(v - best) < 1e-6]


def plot_results(metrics: dict, output_base: str, *, seeds: tuple[int, ...]) -> str:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_max = 0.0
    for pol, label, color, ls, marker in SERIES:
        xs, ys, ystd = [], [], []
        for b in PLOT_BUDGETS:
            row = metrics[pol].get(b)
            if not row or row["n_seeds"] == 0:
                continue
            xs.append(row["steps"])
            ys.append(row["last_pct"])
            ystd.append(row["last_pct_std"])
        if not xs:
            continue
        y_max = max(y_max, max(y + s for y, s in zip(ys, ystd)))

        ax.fill_between(
            xs,
            [y - s for y, s in zip(ys, ystd)],
            [y + s for y, s in zip(ys, ystd)],
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            linestyle=ls,
            linewidth=2.6,
            marker=marker,
            markersize=7,
            color=color,
            label=label,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks([CRAFTAX_BUDGET_STEPS[b] for b, _ in BUDGET_TICKS])
    ax.set_xticklabels([lbl for _, lbl in BUDGET_TICKS])
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Normalized environment return (%)")
    ax.set_ylim(0, max(y_max * 1.08, 1.0))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, which="major", linestyle=":", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    ax.legend(
        loc="upper left",
        framealpha=0.95,
        edgecolor="#E2E8F0",
        fontsize=13,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_base)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            output_base + f".{ext}",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )
    plt.close(fig)
    print(f"Wrote {output_base}.png")
    print(f"Wrote {output_base}.pdf")
    return output_base + ".png"


def _fmt_pct_pm(x: float, std: float, bold: bool = False) -> str:
    s = f"{x:.1f}$\\pm${std:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def write_latex_table(metrics: dict, output_path: str, *, seeds: tuple[int, ...]) -> str:
    seed_lo, seed_hi = min(seeds), max(seeds)
    short = {
        "flat_512": "Flat PPO",
        "meta_no_mask": "Meta-PPO",
        "flat_rnn": "Flat PPO-RNN",
        "meta_rnn_no_mask": "Meta-PPO-RNN",
    }
    budget_labels = {
        "100k": "100k",
        "500k": "500k",
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "10m": "10m",
        "20m": "20m",
    }

    header = " & ".join(short[pol] for pol, *_ in SERIES)
    lines = [
        "% Auto-generated by scripts/generate_thesis_ppo_results.py — do not hand-edit.",
        "% Requires: \\usepackage{booktabs}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"Env. steps & {header} \\\\",
        "\\midrule",
    ]

    for budget in CRAFTAX_BUDGET_LABELS:
        if all(metrics[pol].get(budget, {}).get("n_seeds", 0) == 0 for pol, *_ in SERIES):
            continue
        vals = {
            pol: metrics[pol][budget]["last_pct"]
            for pol, *_ in SERIES
            if metrics[pol].get(budget, {}).get("n_seeds", 0) > 0
        }
        best = _best_policies(vals) if vals else []
        cells = []
        for pol, *_ in SERIES:
            row = metrics[pol].get(budget)
            if not row or row["n_seeds"] == 0:
                cells.append("—")
                continue
            cells.append(
                _fmt_pct_pm(
                    row["last_pct"],
                    row["last_pct_std"],
                    bold=pol in best,
                )
            )
        lines.append(f"{budget_labels.get(budget, budget)} & " + " & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Normalized environment return (\\%); "
            f"mean $\\pm$ one sample standard deviation over seeds ${seed_lo}$--${seed_hi}$.}}",
            "\\label{tab:training_results}",
            "\\end{table}",
            "",
        ]
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path}")
    return output_path


def plot_greedy_boxplot(
    whisker_path: str,
    output_base: str,
    *,
    num_episodes: int,
    master_seed: int = 42,
) -> str:
    """Seed-level boxplots of greedy normalized return (%) at one budget."""
    payload = _load(whisker_path)
    classic = payload["classic"]
    labels, colors, data = [], [], []
    for key, label, color in GREEDY_BOX_SERIES:
        block = classic.get(key)
        if not block:
            continue
        vals = [float(v) for v in block["across_seeds"]["values"]]
        labels.append(label)
        colors.append(color)
        data.append(vals)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150, facecolor="white")
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#0F172A", "linewidth": 1.4},
        whiskerprops={"color": "#334155", "linewidth": 1.1},
        capprops={"color": "#334155", "linewidth": 1.1},
        boxprops={"linewidth": 1.1, "edgecolor": "#334155"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_ylabel("Normalized environment return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_base)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            output_base + f".{ext}",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )
    plt.close(fig)
    print(f"Wrote {output_base}.png")
    print(f"Wrote {output_base}.pdf")
    return output_base + ".png"


def write_figure_tex(output_path: str) -> str:
    text = """\
% Auto-generated snippet for Overleaf — include in Results section.
% Copy ppo_training_vs_budget.pdf into your Overleaf figures/ folder.

\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{figures/ppo_training_vs_budget.pdf}
  \\caption{Normalized environment return on Craftax Classic versus training environment steps (seeds~0--4).
    Flat PPO and Flat PPO-RNN follow the official Craftax baseline protocol.
    Meta-PPO / Meta-PPO-RNN use a 128-d controller with frozen OpenES skills (no action mask).
    Lines show the mean; shaded bands show $\\pm$1 sample standard deviation across seeds.}
  \\label{fig:ppo-training-budget}
\\end{figure}
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis PPO figure and LaTeX table")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated training seeds for mean ± std (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--figure-base",
        default=os.path.join(FIG_DIR, "ppo_training_vs_budget"),
        help="Output path without extension (.png and .pdf written)",
    )
    parser.add_argument(
        "--latex",
        default=os.path.join(LATEX_DIR, "ppo_training_results_table.tex"),
    )
    parser.add_argument(
        "--figure-tex",
        default=os.path.join(LATEX_DIR, "ppo_training_figure.tex"),
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Only write PDF/PNG figures (skip thesis/figures LaTeX snippets).",
    )
    args = parser.parse_args()

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    metrics = _collect(seeds)
    plot_results(metrics, args.figure_base, seeds=seeds)
    for n_ep in (128, 256):
        whisker = os.path.join(
            REPORT_DIR,
            f"greedy_whisker_thesis_quartet_20m_{n_ep}ep.json",
        )
        plot_greedy_boxplot(
            whisker,
            os.path.join(FIG_DIR, f"greedy_boxplot_20m_{n_ep}ep"),
            num_episodes=n_ep,
            master_seed=42,
        )
    if not args.skip_latex:
        write_latex_table(metrics, args.latex, seeds=seeds)
        write_figure_tex(args.figure_tex)


if __name__ == "__main__":
    main()
