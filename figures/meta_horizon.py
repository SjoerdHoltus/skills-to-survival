#!/usr/bin/env python3
"""Thesis figure: fixed meta-action horizon vs normalized environment return.

Reproduces the preliminary single-seed horizon study at the 10M budget.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")

# Fixed preliminary results (seed 42, 10M env steps); do not recompute.
HORIZONS = (1, 2, 4, 6, 10)
RETURNS_PCT = (54.0, 45.9, 36.2, 36.1, 26.5)
BAR_COLOR = "#EA580C"


def plot_horizon(output_base: str) -> str:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = list(range(len(HORIZONS)))
    bars = ax.bar(
        x,
        RETURNS_PCT,
        color=BAR_COLOR,
        edgecolor="white",
        width=0.72,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Meta-action horizon (h)")
    ax.set_ylabel("Normalized environment return (%)")
    ax.set_ylim(0, max(RETURNS_PCT) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, RETURNS_PCT):
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis meta-action horizon figure")
    parser.add_argument(
        "--figure-base",
        default=os.path.join(FIG_DIR, "meta_horizon_last_pct_10m"),
        help="Output path without extension (.png and .pdf written)",
    )
    args = parser.parse_args()
    plot_horizon(args.figure_base)


if __name__ == "__main__":
    main()
