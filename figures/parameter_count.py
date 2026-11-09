#!/usr/bin/env python3
"""Generate the thesis parameter-count comparison figure."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "figures", "output", "parameter_count_comparison")

MODELS = ("PPO", "Meta-PPO + frozen skills")
PARAMETERS = (2_438_162, 415_535)
COLORS = ("#2563EB", "#EA580C")


def main() -> None:
    values_m = [value / 1_000_000 for value in PARAMETERS]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=160)
    bars = ax.barh(MODELS, values_m, color=COLORS, height=0.55)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters (millions)")
    ax.set_xlim(0, max(values_m) * 1.18)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fM"))
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)

    for index, (bar, value, count) in enumerate(zip(bars, values_m, PARAMETERS)):
        label = f"{count / 1_000_000:.2f}M"
        if index == 1:
            label += " (83.0% fewer)"
        ax.text(
            value + max(values_m) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=14,
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            f"{OUTPUT_BASE}.{extension}",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
