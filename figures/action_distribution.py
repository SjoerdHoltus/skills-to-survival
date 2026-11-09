#!/usr/bin/env python3
"""Thesis figures: greedy-eval action histograms (one bar chart per policy).

Reads ``greedy_eval_seed42_128ep.json`` from budget-sweep runs.
Raw action identifiers are preserved for loading/aggregation; display labels
are applied only when rendering.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from configs.survival_meta import SURVIVAL_META_ACTION_LABELS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")
LATEX_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")

# Shared y-axis for flat and meta so thesis figures align.
# Meta skills stay at the top; `do` is inserted after skills so flat's DO is visible.
_ALIGNED_LABELS: tuple[str, ...] = (
    *SURVIVAL_META_ACTION_LABELS[:9],  # noop + 8 skills
    "do",
    *SURVIVAL_META_ACTION_LABELS[9:],  # place/craft/move/sleep
)

# Display-only mapping; internal identifiers stay unchanged for data loading.
_DISPLAY_LABELS: dict[str, str] = {
    "noop": "No-op",
    "wood": "Collect wood",
    "drink": "Find water",
    "food": "Hunt food",
    "fight": "Combat",
    "stone": "Collect stone",
    "coal": "Collect coal",
    "iron": "Collect iron",
    "diamond": "Collect diamond",
    "do": "DO",
    "place_table": "Place table",
    "place_furnace": "Place furnace",
    "place_stone": "Place stone",
    "place_plant": "Place plant",
    "craft_wood_pickaxe": "Craft wooden pickaxe",
    "craft_stone_pickaxe": "Craft stone pickaxe",
    "craft_iron_pickaxe": "Craft iron pickaxe",
    "craft_wood_sword": "Craft wooden sword",
    "craft_stone_sword": "Craft stone sword",
    "craft_iron_sword": "Craft iron sword",
    "up": "Move up",
    "left": "Move left",
    "down": "Move down",
    "right": "Move right",
    "sleep": "Sleep",
}

# Flat eval uses Craftax Action enum names; map onto meta / aligned names.
_FLAT_TO_ALIGNED = {
    "NOOP": "noop",
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "DO": "do",
    "SLEEP": "sleep",
    "PLACE_STONE": "place_stone",
    "PLACE_TABLE": "place_table",
    "PLACE_FURNACE": "place_furnace",
    "PLACE_PLANT": "place_plant",
    "MAKE_WOOD_PICKAXE": "craft_wood_pickaxe",
    "MAKE_STONE_PICKAXE": "craft_stone_pickaxe",
    "MAKE_IRON_PICKAXE": "craft_iron_pickaxe",
    "MAKE_WOOD_SWORD": "craft_wood_sword",
    "MAKE_STONE_SWORD": "craft_stone_sword",
    "MAKE_IRON_SWORD": "craft_iron_sword",
}

DEFAULT_FLAT_EVAL = os.path.join(
    _PROJECT_ROOT,
    "runs/baselines_classic_flat_ppo_20m/run_071626_231351/evals/greedy_eval_seed42_128ep.json",
)
DEFAULT_META_EVAL = os.path.join(
    _PROJECT_ROOT,
    "runs/baselines_classic_meta_ppo_no_mask_20m/run_071726_003850/evals/greedy_eval_seed42_128ep.json",
)
DEFAULT_FLAT_RNN_EVAL = os.path.join(
    _PROJECT_ROOT,
    "runs/baselines_classic_flat_ppo_rnn_20m/run_072926_003025/evals/greedy_eval_seed42_128ep.json",
)
DEFAULT_META_RNN_EVAL = os.path.join(
    _PROJECT_ROOT,
    "runs/baselines_classic_meta_ppo_rnn_no_mask_20m/run_072926_012034/evals/greedy_eval_seed42_128ep.json",
)

POLICIES = (
    ("flat", "Flat PPO", DEFAULT_FLAT_EVAL),
    ("meta", "Meta-PPO", DEFAULT_META_EVAL),
    ("flat_rnn", "Flat PPO-RNN", DEFAULT_FLAT_RNN_EVAL),
    ("meta_rnn", "Meta-PPO-RNN", DEFAULT_META_RNN_EVAL),
)

# Meta skills (OpenES primitives) vs direct meta-actions.
FLAT_COLOR = "#2563EB"
META_SKILL_COLOR = "#93C5FD"  # lighter blue
META_CRAFTAX_COLOR = FLAT_COLOR


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _style_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


META_SKILLS = frozenset({"wood", "drink", "food", "fight", "stone", "coal", "iron", "diamond"})


def _display_label(raw: str) -> str:
    return _DISPLAY_LABELS.get(raw, raw)


def _distribution_fractions(payload: dict, *, policy_key: str) -> tuple[list[str], np.ndarray]:
    """Remap raw eval labels onto the shared meta-aligned y-axis."""
    dist = payload["policy_action_distribution"]
    raw_labels = [str(x) for x in dist["labels"]]
    fracs = np.asarray(dist["fractions"], dtype=np.float64)
    if fracs.size < len(raw_labels):
        fracs = np.pad(fracs, (0, len(raw_labels) - fracs.size))
    elif fracs.size > len(raw_labels):
        fracs = fracs[: len(raw_labels)]

    by_aligned: dict[str, float] = {lb: 0.0 for lb in _ALIGNED_LABELS}
    flat_like = policy_key.startswith("flat")
    for raw, frac in zip(raw_labels, fracs):
        if flat_like:
            aligned = _FLAT_TO_ALIGNED.get(raw, _FLAT_TO_ALIGNED.get(raw.upper()))
        else:
            aligned = raw.lower()
            if aligned.startswith("make_"):
                aligned = "craft_" + aligned[len("make_") :]
        if aligned in by_aligned:
            by_aligned[aligned] += float(frac)

    labels = list(_ALIGNED_LABELS)
    return labels, 100.0 * np.asarray([by_aligned[lb] for lb in labels], dtype=np.float64)


def _bar_colors(policy_key: str, labels: list[str]) -> list[str]:
    """Skills: light blue; direct meta-actions (incl. No-op): same blue as flat."""
    if policy_key.startswith("flat"):
        return [FLAT_COLOR] * len(labels)
    colors = []
    for lb in labels:
        key = lb.lower()
        if key in META_SKILLS:
            colors.append(META_SKILL_COLOR)
        else:
            # noop and Craftax place/craft/move/sleep are direct meta-actions
            colors.append(META_CRAFTAX_COLOR)
    return colors


def _plot_policy_histogram(
    labels: list[str],
    fracs_pct: np.ndarray,
    *,
    policy_key: str,
    output_base: str,
) -> str:
    _style_rc()
    n = len(labels)
    # Horizontal bars grow rightward; wide enough for longer display labels.
    fig_h = max(5.5, 0.34 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(n)
    colors = _bar_colors(policy_key, labels)
    ax.barh(y, fracs_pct, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_display_label(lb) for lb in labels], fontsize=13)
    ax.set_xlabel("Share of steps (%)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax.set_xlim(0, max(float(fracs_pct.max()) * 1.12, 1.0))
    ax.invert_yaxis()  # first distribution entry at the top
    if policy_key.startswith("meta"):
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(
                    facecolor=META_SKILL_COLOR,
                    edgecolor="white",
                    label="Skill-backed meta-actions",
                ),
                Patch(
                    facecolor=META_CRAFTAX_COLOR,
                    edgecolor="white",
                    label="Direct meta-actions",
                ),
            ],
            loc="lower right",
            framealpha=0.95,
            edgecolor="#E2E8F0",
            fontsize=12,
        )

    fig.subplots_adjust(left=0.28, right=0.98, top=0.98, bottom=0.08)
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


def write_latex_figure(output_path: str, *, budget: str, seed: int) -> str:
    lines = [
        "% Auto-generated by scripts/generate_thesis_action_distribution.py — do not hand-edit.",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.92\\linewidth]{{greedy_action_distribution_meta_{budget}.pdf}}\\\\[0.6em]",
        f"\\includegraphics[width=0.92\\linewidth]{{greedy_action_distribution_flat_{budget}.pdf}}",
        "\\caption{Greedy evaluation action distributions on Craftax Classic "
        f"({budget} training budget, seed~{seed}, 128 episodes). "
        "Both panels share the same meta-controller label order (skill-backed "
        "meta-actions at the top, then direct meta-actions; \\texttt{{DO}} is "
        "shown for Flat PPO only). "
        "Top: Meta-PPO modes (light blue~=~skill-backed meta-actions, "
        "blue~=~direct meta-actions). "
        "Bottom: Flat PPO environment actions.}",
        "\\label{fig:greedy-action-distribution}",
        "\\end{figure}",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis action-distribution histograms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", default="20m")
    parser.add_argument("--flat-eval", default=DEFAULT_FLAT_EVAL)
    parser.add_argument("--meta-eval", default=DEFAULT_META_EVAL)
    parser.add_argument("--flat-rnn-eval", default=DEFAULT_FLAT_RNN_EVAL)
    parser.add_argument("--meta-rnn-eval", default=DEFAULT_META_RNN_EVAL)
    parser.add_argument(
        "--policies",
        default="all",
        choices=("all", "ff", "rnn"),
        help="Which policy pair(s) to render (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        default=FIG_DIR,
        help="Directory for output .png / .pdf files",
    )
    parser.add_argument(
        "--latex",
        default=os.path.join(LATEX_DIR, "greedy_action_distribution_figure.tex"),
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Only write PDF/PNG figures (skip thesis/figures LaTeX snippet).",
    )
    args = parser.parse_args()

    eval_paths = {
        "flat": args.flat_eval,
        "meta": args.meta_eval,
        "flat_rnn": args.flat_rnn_eval,
        "meta_rnn": args.meta_rnn_eval,
    }
    if args.policies == "ff":
        selected = ("flat", "meta")
    elif args.policies == "rnn":
        selected = ("flat_rnn", "meta_rnn")
    else:
        selected = tuple(key for key, _, _ in POLICIES)

    for key in selected:
        path = eval_paths[key]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if "policy_action_distribution" not in _load(path):
            raise ValueError(f"{path} lacks policy_action_distribution")

    for key in selected:
        labels, fracs_pct = _distribution_fractions(_load(eval_paths[key]), policy_key=key)
        out_base = os.path.join(args.out_dir, f"greedy_action_distribution_{key}_{args.budget}")
        _plot_policy_histogram(
            labels,
            fracs_pct,
            policy_key=key,
            output_base=out_base,
        )

    if not args.skip_latex and "flat" in selected and "meta" in selected:
        write_latex_figure(args.latex, budget=args.budget, seed=args.seed)


if __name__ == "__main__":
    main()
