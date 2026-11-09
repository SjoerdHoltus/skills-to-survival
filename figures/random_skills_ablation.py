#!/usr/bin/env python3
"""Thesis figure: skill-quality ablation (random Flax-init skills vs trained).

Compares greedy 128-ep normalized return (%) at the 20m budget across seeds
0–4 for Flat PPO, Meta-PPO (trained OpenES skills), and Meta-PPO with frozen
random skill weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")
REPORT_DIR = os.path.join(_PROJECT_ROOT, "runs/benchmark_reports/baseline_migration")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

FLAT_COLOR = "#2563EB"
META_COLOR = "#EA580C"
RANDOM_COLOR = "#78716C"  # stone — ablation control

# (key, label, color, runs_policy, random_skills, layer_size)
SERIES = (
    ("flat_512", "Flat PPO", FLAT_COLOR, "baselines_classic_flat_ppo_20m", False, 512),
    (
        "meta_no_mask",
        "Meta-PPO\n(trained skills)",
        META_COLOR,
        "baselines_classic_meta_ppo_no_mask_20m",
        False,
        128,
    ),
    (
        "meta_random",
        "Meta-PPO\n(random skills)",
        RANDOM_COLOR,
        "baselines_classic_meta_ppo_no_mask_random_skills_20m",
        True,
        128,
    ),
)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_run(
    runs_policy: str,
    seed: int,
    *,
    random_skills: bool,
    layer_size: int,
) -> str | None:
    root = os.path.join(_PROJECT_ROOT, "runs", runs_policy)
    cands: list[str] = []
    for d in sorted(glob.glob(os.path.join(root, "run_*"))):
        cfgp = os.path.join(d, "config.json")
        wp = os.path.join(d, "weights", "params.msgpack")
        if not (os.path.isfile(cfgp) and os.path.isfile(wp)):
            continue
        hp = _load(cfgp).get("hyperparameters") or {}
        if int(hp.get("seed", -1)) != seed:
            continue
        if bool(hp.get("random_skills", False)) != random_skills:
            continue
        if int(hp.get("layer_size", -1)) != layer_size:
            continue
        cands.append(d)
    return cands[-1] if cands else None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def collect(seeds: tuple[int, ...]) -> dict:
    out: dict = {
        "budget": "20m",
        "seeds": list(seeds),
        "num_episodes": 128,
        "master_seed": 42,
        "metric": "return_pct_max_achievement_only",
        "policies": {},
    }
    for key, label, _color, runs_policy, random_skills, layer_size in SERIES:
        greedy_vals: list[float] = []
        train_vals: list[float] = []
        per_seed: list[dict] = []
        for seed in seeds:
            run = _find_run(
                runs_policy,
                seed,
                random_skills=random_skills,
                layer_size=layer_size,
            )
            if run is None:
                continue
            gp = os.path.join(run, "evals", "greedy_eval_seed42_128ep.json")
            trp = os.path.join(run, "training_return_metrics.json")
            row: dict = {"seed": seed, "run_dir": run}
            if os.path.isfile(gp):
                g = _load(gp)
                pct = float(g["return_pct_max_achievement_only"]["mean"])
                greedy_vals.append(pct)
                row["greedy_return_pct"] = pct
                row["crafter_score_pct"] = float(g.get("crafter_score_pct", 0.0))
            if os.path.isfile(trp):
                t = _load(trp)
                last = float(t["last_update"]["return_pct_max"])
                train_vals.append(last)
                row["train_last_return_pct"] = last
                row["train_tail_return_pct"] = float(t["tail_mean"]["return_pct_max"])
            per_seed.append(row)
        g_mean, g_std = _mean_std(greedy_vals)
        t_mean, t_std = _mean_std(train_vals)
        out["policies"][key] = {
            "label": label.replace("\n", " "),
            "runs_policy": runs_policy,
            "greedy_values": greedy_vals,
            "greedy_mean": g_mean,
            "greedy_std": g_std,
            "train_last_values": train_vals,
            "train_last_mean": t_mean,
            "train_last_std": t_std,
            "per_seed": per_seed,
        }
    return out


def plot_histogram(payload: dict, output_base: str) -> str:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(SERIES))
    means, stds, colors, tick_labels = [], [], [], []
    for i, (key, label, color, _rp, _rs, _ls) in enumerate(SERIES):
        pol = payload["policies"][key]
        means.append(pol["greedy_mean"])
        stds.append(pol["greedy_std"])
        colors.append(color)
        tick_labels.append(label)
        # seed jitter overlay
        vals = pol["greedy_values"]
        jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2.0) * 0.04
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color="white",
            edgecolors="#334155",
            linewidths=1.0,
            s=36,
            zorder=5,
        )

    err_kw = dict(ecolor="#334155", elinewidth=1.2, capsize=5, capthick=1.2)
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        error_kw=err_kw,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.72,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Normalized environment return (%)")
    y_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, max(y_top * 1.18, 70.0))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    for bar, m, s in zip(bars, means, stds):
        ax.annotate(
            f"{m:.1f}±{s:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, m + s),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
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
    p = argparse.ArgumentParser(description="Generate random-skills ablation histogram")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument(
        "--figure-base",
        default=os.path.join(FIG_DIR, "random_skills_ablation_greedy_20m"),
    )
    p.add_argument(
        "--json-out",
        default=os.path.join(REPORT_DIR, "random_skills_ablation_greedy_20m.json"),
    )
    args = p.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    payload = collect(seeds)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.json_out}")
    plot_histogram(payload, args.figure_base)

    # console summary
    print(f"\n=== Greedy 128ep return % (seeds {list(seeds)}) ===")
    for key, *_ in SERIES:
        pol = payload["policies"][key]
        vals = ", ".join(f"{v:.2f}" for v in pol["greedy_values"])
        print(f"{pol['label']:28s}  {pol['greedy_mean']:.2f} ± {pol['greedy_std']:.2f}  [{vals}]")
    print("\n=== Training last-update return % ===")
    for key, *_ in SERIES:
        pol = payload["policies"][key]
        vals = ", ".join(f"{v:.2f}" for v in pol["train_last_values"])
        print(
            f"{pol['label']:28s}  "
            f"{pol['train_last_mean']:.2f} ± {pol['train_last_std']:.2f}  [{vals}]"
        )


if __name__ == "__main__":
    main()
