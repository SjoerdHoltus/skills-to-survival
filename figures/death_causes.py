#!/usr/bin/env python3
"""Thesis figure: death-cause histogram (meta vs flat baseline 20m).

Reads death-cause JSON from ``scripts/eval_death_causes.py`` (single-seed or
multi-seed aggregate) and writes PNG+PDF under figures/output.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")
DEFAULT_JSON = os.path.join(
    _PROJECT_ROOT,
    "thesis/figures/presentation_rollouts/death_causes_seeds0-4.json",
)

META_COLOR = "#EA580C"
FLAT_COLOR = "#2563EB"

BUCKET_ORDER = ("needs", "combat", "other")
BUCKET_DISPLAY = {
    "needs": "Needs",
    "combat": "Combat",
    "other": "Other",
}
CAUSE_TO_BUCKET = {
    "needs": "needs",
    "zombie": "combat",
    "arrow": "combat",
    "lava": "other",
    "timeout": "other",
    "unknown": "other",
}


def _style_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def _bucket_fractions(by_cause: dict) -> dict[str, float]:
    out = {b: 0.0 for b in BUCKET_ORDER}
    for cause, row in by_cause.items():
        bucket = CAUSE_TO_BUCKET.get(cause)
        if bucket is None:
            continue
        out[bucket] += float(row.get("fraction", 0.0))
    return out


def _policy_bucket_stats(policy_payload: dict) -> tuple[list[float], list[float]]:
    """Return (means_pct, stds_pct) for BUCKET_ORDER."""
    if "aggregate" in policy_payload:
        means, stds = [], []
        for b in BUCKET_ORDER:
            row = policy_payload["aggregate"]["by_bucket"][b]
            means.append(100.0 * float(row["mean"]))
            stds.append(100.0 * float(row["std"]))
        return means, stds
    # Single-seed legacy format
    fracs = _bucket_fractions(policy_payload["summary"]["by_cause"])
    means = [100.0 * fracs[b] for b in BUCKET_ORDER]
    return means, [0.0] * len(BUCKET_ORDER)


def plot_death_causes(payload: dict, output_base: str) -> str:
    _style_rc()
    policies = payload["policies"]
    meta_pct, meta_std = _policy_bucket_stats(policies["meta"])
    flat_pct, flat_std = _policy_bucket_stats(policies["flat"])
    labels = [BUCKET_DISPLAY[b] for b in BUCKET_ORDER]
    multi = any(s > 0 for s in meta_std + flat_std)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = list(range(len(BUCKET_ORDER)))
    width = 0.38
    err_kw = dict(ecolor="#334155", elinewidth=1.2, capsize=4, capthick=1.2)
    bars_m = ax.bar(
        [i - width / 2 for i in x],
        meta_pct,
        width=width,
        yerr=meta_std if multi else None,
        error_kw=err_kw if multi else None,
        color=META_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Meta-PPO",
        zorder=3,
    )
    bars_f = ax.bar(
        [i + width / 2 for i in x],
        flat_pct,
        width=width,
        yerr=flat_std if multi else None,
        error_kw=err_kw if multi else None,
        color=FLAT_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Flat PPO",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Death cause")
    ax.set_ylabel("Share of episodes (%)")
    y_top = max(m + s for m, s in zip(meta_pct + flat_pct, meta_std + flat_std))
    ax.set_ylim(0, y_top * 1.22)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95)

    for bars, stds in ((bars_m, meta_std), (bars_f, flat_std)):
        for bar, std in zip(bars, stds):
            h = bar.get_height()
            if h < 0.05:
                continue
            label = f"{h:.1f}" if not multi else f"{h:.1f}±{std:.1f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, h + (std if multi else 0.0)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10 if multi else 11,
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
    p = argparse.ArgumentParser(description="Generate thesis death-cause histogram")
    p.add_argument("--json", default=DEFAULT_JSON)
    p.add_argument(
        "--figure-base",
        default=os.path.join(FIG_DIR, "death_cause_distribution_20m"),
        help="Output path without extension (.png and .pdf written)",
    )
    args = p.parse_args()
    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)
    plot_death_causes(payload, args.figure_base)


if __name__ == "__main__":
    main()
