#!/usr/bin/env python3
"""Thesis figures + tables: Classic → Craftax Extended skill transfer.

Reads benchmark report JSONs/CSVs and writes:
  figures/output/classic_to_extended_bootstrap.{pdf,png}
  figures/output/classic_to_extended_finetune.{pdf,png}
  figures/output/classic_to_extended_natural.{pdf,png}
  figures/output/skill_natural_eval_rates.{pdf,png}
  figures/output/skill_natural_eval_full_rates.{pdf,png}
  figures/output/skill_natural_eval_full_ft50_rates.{pdf,png}
  figures/output/fight_novel_melee_types.{pdf,png}
  figures/output/classic_to_extended_*.tex

Bootstrap figure/table use dual-metric classic-formula scores from
``classic_to_extended_dual_metrics`` so C@C and C@E share one scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = _PROJECT_ROOT / "figures" / "output"
TABLE_DIR = _PROJECT_ROOT / "figures" / "output"
REPORT = _PROJECT_ROOT / "runs" / "benchmark_reports"

# Thesis display order and labels (shared by Classic + Extended natural rates).
SKILLS = (
    ("find_wood", "Collect wood"),
    ("collect_stone", "Collect stone"),
    ("collect_ore", "Collect ore"),
    ("drink_water", "Find water"),
    ("find_food", "Hunt food"),
    ("monster_fight", "Combat"),
)

TABLE_SKILL_LABELS = {
    "find_wood": "Collect Wood",
    "collect_stone": "Collect Stone",
    "collect_ore": "Collect Ore",
    "drink_water": "Find Water",
    "find_food": "Hunt Food",
    "monster_fight": "Combat",
}

COLORS = {
    "c_c": "#64748B",  # Classic@Classic
    "c_f": "#2563EB",  # Classic@Extended
    "u_f": "#94A3B8",  # Untrained@Extended
    "scratch": "#94A3B8",
    "finetune": "#EA580C",
    "nat_c": "#2563EB",
    "nat_ft": "#EA580C",
    "nat_classic_ref": "#CBD5E1",
}

# Shared across Classic + Extended natural rate histograms (same metric → same colour).
NATURAL_RATE_COLORS = (
    ("Encountered target", "ever", "#64748B"),
    ("Success", "success", "#2563EB"),
    ("Success given encounter", "succ_ever", "#16A34A"),
    ("Death", "death", "#DC2626"),
)


def _rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def _load_json(path: Path):
    return json.loads(path.read_text())


def bootstrap_panel_data() -> dict[str, dict]:
    """Per-skill C@C, C@E, and U@E under the Classic reward formula."""
    dual = _load_json(REPORT / "classic_to_extended_dual_metrics" / "transfer_choice.json")
    by: dict[str, dict] = {}
    for t in dual["transfer"]:
        cf = t["classic_formula"]
        c_c = float(cf["C@C"])
        c_e = float(cf["C@E"])
        u_e = float(cf["U@E"])
        by[t["skill"]] = {
            "metric": "score",
            "c_c": c_c,
            "c_f": c_e,
            "u_f": u_e,
            "retention": c_e / c_c if c_c else float("nan"),
        }
    return by


def finetune_vs_scratch() -> dict[str, dict[str, float]]:
    # Final OpenES fitness from run directory names / manifests (seed 42, 50 gens).
    return {
        "find_wood": {"scratch": 39.40, "finetune": 43.49},
        "collect_stone": {"scratch": 12.69, "finetune": 32.95},
        "collect_ore": {"scratch": 5.46, "finetune": 11.31},
        "drink_water": {"scratch": 3.11, "finetune": 62.70},
        "find_food": {"scratch": 43.25, "finetune": 86.40},
        "monster_fight": {"scratch": 5.93, "finetune": 23.02},
    }


def natural_full() -> dict[str, dict]:
    zs = _load_json(REPORT / "skill_natural_eval_full_classic_zeroshot.json")
    ft = _load_json(REPORT / "skill_natural_eval_full_finetune50.json")
    classic = _load_json(REPORT / "skill_natural_eval.json")
    out = {}
    for r in zs["results"]:
        out[r["skill"]] = {
            "success": float(r["success_rate"]),
            "ever": float(r["ever_saw_target_rate"]),
            "succ_ever": float(r["success_rate_given_ever_saw_target"]),
            "death": float(r["death_rate"]),
            "ft_success": float("nan"),
            "ft_succ_ever": float("nan"),
        }
    for r in ft["results"]:
        out[r["skill"]]["ft_success"] = float(r["success_rate"])
        out[r["skill"]]["ft_succ_ever"] = float(r["success_rate_given_ever_saw_target"])
        out[r["skill"]]["ft_ever"] = float(r["ever_saw_target_rate"])
        out[r["skill"]]["ft_death"] = float(r["death_rate"])
    for r in classic.get("results", []):
        skill = r.get("skill")
        if skill in out:
            out[skill]["classic_natural_success"] = float(r["success_rate"])
    return out


def novel_melee() -> list[dict]:
    return _load_json(REPORT / "fight_novel_melee_types" / "summary.json")


def _savefig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_base.with_suffix('.pdf')}")


def plot_bootstrap_figure(boot: dict, out_base: Path) -> None:
    """Zero-shot bootstrap transfer: C@C, C@E, and U@E on the Classic formula."""
    from matplotlib.patches import Patch

    _rc()
    fig, axes = plt.subplots(
        1,
        6,
        figsize=(11.0, 3.4),
        dpi=150,
        facecolor="white",
        sharey=False,
    )
    for ax, (key, label) in zip(axes, SKILLS):
        d = boot[key]
        vals = [d["c_c"], d["c_f"], d["u_f"]]
        cols = [COLORS["c_c"], COLORS["c_f"], COLORS["u_f"]]
        names = ["C@C", "C@E", "U@E"]
        ax.bar(
            np.arange(3),
            vals,
            color=cols,
            edgecolor="white",
            width=0.78,
            zorder=3,
        )
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_xlabel(label, fontsize=10, labelpad=4)
        ax.axhline(0.0, color="#334155", linewidth=0.8, zorder=2)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        finite_vals = [v for v in vals if np.isfinite(v)]
        ymin = min(0.0, min(finite_vals) * 1.05)
        ymax = max(finite_vals)
        ax.set_ylim(ymin, ymax * 1.18 if ymax > 0 else 1.0)
    axes[0].set_ylabel("Fitness (Classic reward formula)")
    legend = [
        Patch(facecolor=COLORS["c_c"], label="Classic checkpoint on Classic (C@C)"),
        Patch(facecolor=COLORS["c_f"], label="Classic checkpoint on Extended (C@E)"),
        Patch(facecolor=COLORS["u_f"], label="Untrained on Extended (U@E)"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.12, right=0.99, wspace=0.35)
    _savefig(fig, out_base)


def plot_finetune_figure(ft: dict, out_base: Path) -> None:
    """Matched-budget fine-tune vs scratch on Extended bootstrap."""
    _rc()
    fig, ax = plt.subplots(figsize=(9.5, 3.8), dpi=150, facecolor="white")
    keys_b = [k for k, _ in SKILLS]
    labels_b = [lab for _, lab in SKILLS]
    x = np.arange(len(keys_b))
    w = 0.36
    scratch = [ft[k]["scratch"] for k in keys_b]
    fine = [ft[k]["finetune"] for k in keys_b]
    ax.bar(
        x - w / 2,
        scratch,
        width=w,
        color=COLORS["scratch"],
        edgecolor="white",
        label="Extended from scratch (50 generations)",
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        fine,
        width=w,
        color=COLORS["finetune"],
        edgecolor="white",
        label="Classic warm start (50 generations)",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_b, fontsize=8)
    ax.set_ylabel("Final OpenES fitness")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.98,
        "Seed 42",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#475569",
    )
    ymax = max(scratch + fine)
    for i, (s, f) in enumerate(zip(scratch, fine)):
        ax.text(i - w / 2, s + ymax * 0.02, f"{s:.0f}", ha="center", fontsize=7)
        ax.text(i + w / 2, f + ymax * 0.02, f"{f:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    _savefig(fig, out_base)


def plot_natural_success_figure(nat: dict, out_base: Path) -> None:
    """Natural Extended success rates vs Classic natural reference."""
    _rc()
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=150, facecolor="white")
    labels_c = [lab for _, lab in SKILLS]
    keys_c = [k for k, _ in SKILLS]
    x = np.arange(len(keys_c))
    w = 0.28
    succ = [100 * nat[k]["success"] for k in keys_c]
    ref = [100 * nat[k].get("classic_natural_success", float("nan")) for k in keys_c]

    ax.bar(
        x - w,
        succ,
        width=w,
        color=COLORS["nat_c"],
        edgecolor="white",
        label="Classic skill checkpoints (zero-shot)",
        zorder=3,
    )
    ft_x = [i for i, k in enumerate(keys_c) if np.isfinite(nat[k]["ft_success"])]
    ft_y = [100 * nat[keys_c[i]]["ft_success"] for i in ft_x]
    ax.bar(
        np.array(ft_x, dtype=float),
        ft_y,
        width=w,
        color=COLORS["nat_ft"],
        edgecolor="white",
        label="Skill checkpoints after FT50",
        zorder=3,
    )
    ax.scatter(
        x + w,
        ref,
        marker="D",
        s=36,
        color="white",
        edgecolors="#475569",
        linewidths=1.0,
        zorder=5,
        label="Classic natural (reference)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_c, fontsize=8)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_base)


def plot_natural_rate_histogram(
    series_by_skill: dict[str, dict],
    skill_order: tuple[tuple[str, str], ...],
    out_base: Path,
) -> None:
    """Grouped bars: encounter / success / success|encounter / death (shared colours)."""
    _rc()
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=150, facecolor="white")
    labels = [lab for _, lab in skill_order]
    keys = [k for k, _ in skill_order]
    x = np.arange(len(keys))
    w = 0.2
    for i, (name, field, col) in enumerate(NATURAL_RATE_COLORS):
        vals = [100 * float(series_by_skill[k][field]) for k in keys]
        ax.bar(
            x + (i - 1.5) * w,
            vals,
            width=w,
            color=col,
            edgecolor="white",
            label=name,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    _savefig(fig, out_base)


def plot_natural_full_rates(nat: dict, out_base: Path) -> None:
    plot_natural_rate_histogram(
        nat,
        SKILLS,
        out_base,
    )


def plot_natural_ft50_rates(nat: dict, out_base: Path) -> None:
    """Same grouped rates as zero-shot, but for FT50 Extended checkpoints."""
    series = {
        k: {
            "ever": float(d["ft_ever"]),
            "success": float(d["ft_success"]),
            "succ_ever": float(d["ft_succ_ever"]),
            "death": float(d["ft_death"]),
        }
        for k, d in nat.items()
    }
    plot_natural_rate_histogram(
        series,
        SKILLS,
        out_base,
    )


def plot_classic_natural_rates(out_base: Path) -> None:
    classic = _load_json(REPORT / "skill_natural_eval.json")
    by_skill = {}
    for r in classic["results"]:
        by_skill[r["skill"]] = {
            "ever": float(r["ever_saw_target_rate"]),
            "success": float(r["success_rate"]),
            "succ_ever": float(r["success_rate_given_ever_saw_target"]),
            "death": float(r["death_rate"]),
        }
    plot_natural_rate_histogram(
        by_skill,
        SKILLS,
        out_base,
    )


MELEE_DISPLAY = {
    "zombie": "Zombie",
    "gnome_warrior": "Gnome warrior",
    "orc_soldier": "Orc soldier",
}


def plot_novel_melee(rows: list[dict], out_base: Path) -> None:
    _rc()
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150, facecolor="white")
    labels = [
        f"{MELEE_DISPLAY.get(r['melee_type_name'], r['melee_type_name'].replace('_', ' ').title())}"
        f"\n(HP {r['melee_hp']})"
        for r in rows
    ]
    x = np.arange(len(rows))
    w = 0.36
    ck = [100 * r["classic_kill_mean"] for r in rows]
    uk = [100 * r["untrained_kill_mean"] for r in rows]
    ax.bar(
        x - w / 2,
        ck,
        width=w,
        color=COLORS["c_f"],
        edgecolor="white",
        label="Classic combat checkpoint on Extended",
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        uk,
        width=w,
        color=COLORS["u_f"],
        edgecolor="white",
        label="Untrained",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Kill rate (%)")
    ax.set_ylim(0, max(ck) * 1.35)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_base.with_suffix('.pdf')}")


def write_bootstrap_table(boot: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Zero-shot transfer of frozen Classic skills onto bootstrapped "
        r"Craftax Extended environments, scored with the Classic reward formula on "
        r"both environments so C@C and C@E share one scale. "
        r"C@C: Classic checkpoint on Classic; C@E: Classic checkpoint on Extended; "
        r"U@E: untrained policy on Extended. Retention is C@E/C@C. "
        r"Ore row is coal at wall depth~2.}",
        r"\label{tab:classic_to_extended_bootstrap}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Skill & Metric & C@C & C@E & U@E & Retention \\",
        r"\midrule",
    ]
    for key, _ in SKILLS:
        d = boot[key]
        metric = "kill rate" if d["metric"] == "kill_rate" else "score"
        lines.append(
            f"{TABLE_SKILL_LABELS[key]} & {metric} & {d['c_c']:.2f} & "
            f"{d['c_f']:.2f} & "
            f"{d['u_f']:.2f} & {d['c_f'] / d['c_c']:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    (TABLE_DIR / "classic_to_extended_bootstrap.tex").write_text("\n".join(lines))


def main() -> None:
    boot = bootstrap_panel_data()
    ft = finetune_vs_scratch()
    nat = natural_full()
    melee = novel_melee()
    for k, v in {
        "find_wood": 0.948,
        "drink_water": 0.166,
        "find_food": 0.894,
        "monster_fight": 0.643,
        "collect_stone": 0.644,
        "collect_ore": 0.079,
    }.items():
        nat[k].setdefault("classic_natural_success", v)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_bootstrap_figure(boot, FIG_DIR / "classic_to_extended_bootstrap")
    plot_finetune_figure(ft, FIG_DIR / "classic_to_extended_finetune")
    plot_natural_success_figure(nat, FIG_DIR / "classic_to_extended_natural")
    plot_classic_natural_rates(FIG_DIR / "skill_natural_eval_rates")
    plot_natural_full_rates(nat, FIG_DIR / "skill_natural_eval_full_rates")
    plot_natural_ft50_rates(nat, FIG_DIR / "skill_natural_eval_full_ft50_rates")
    plot_novel_melee(melee, FIG_DIR / "fight_novel_melee_types")
    write_bootstrap_table(boot)


if __name__ == "__main__":
    main()
