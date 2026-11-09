#!/usr/bin/env python3
"""Plot greedy-eval growth vs env-step budget for all baseline models."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppo.configs import CRAFTAX_BUDGET_LABELS, CRAFTAX_BUDGET_STEPS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT = os.path.join(
    _PROJECT_ROOT,
    "runs/benchmark_reports/baseline_migration/greedy_vs_training_by_budget_seed42.json",
)
DEFAULT_OUTPUT = os.path.join(
    _PROJECT_ROOT,
    "runs/benchmark_reports/baseline_migration/growth_by_budget_seed42.png",
)

SERIES = [
    ("extended", "flat_512", "Extended flat 512"),
    ("classic", "flat_128", "Classic flat 128"),
    ("classic", "flat_256", "Classic flat 256"),
    ("classic", "flat_512", "Classic flat 512"),
    ("classic", "meta", "Classic meta (+mask)"),
    ("classic", "meta_no_mask", "Classic meta (no mask)"),
    ("classic", "flat_rnn", "Classic flat RNN"),
    ("classic", "meta_rnn_no_mask", "Classic meta RNN (no mask)"),
]

SERIES_BY_KEY = {key: entry for entry in SERIES for key in (entry[1], f"{entry[0]}.{entry[1]}")}


def _select_series(spec: str | None) -> list[tuple[str, str, str]]:
    """Comma-separated keys: flat_512, meta, meta_no_mask, or classic.flat_512, etc."""
    if not spec:
        return list(SERIES)
    out: list[tuple[str, str, str]] = []
    for token in spec.split(","):
        token = token.strip().lower().replace(" ", "_")
        if not token:
            continue
        aliases = {
            "flat512": "flat_512",
            "meta_mask": "meta",
            "meta+mask": "meta",
            "metamask": "meta",
            "meta_unmasked": "meta_no_mask",
            "meta_nomask": "meta_no_mask",
            "no_mask": "meta_no_mask",
        }
        token = aliases.get(token, token)
        if token in SERIES_BY_KEY:
            entry = SERIES_BY_KEY[token]
            # Bare classic keys prefer classic section.
            if "." not in token:
                for section, key, label in SERIES:
                    if key == token and section == "classic":
                        entry = (section, key, label)
                        break
            out.append(entry)
            continue
        # classic-only preset
        if token in ("classic_trio", "trio"):
            out.extend(
                [
                    ("classic", "flat_512", "Flat 512"),
                    ("classic", "meta", "Meta (+mask)"),
                    ("classic", "meta_no_mask", "Meta (no mask)"),
                ]
            )
            continue
        if token in ("classic_rnn", "rnn_pair", "rnn"):
            out.extend(
                [
                    ("classic", "flat_rnn", "Flat RNN"),
                    ("classic", "meta_rnn_no_mask", "Meta RNN (no mask)"),
                ]
            )
            continue
        raise SystemExit(f"Unknown series key: {token}")
    # de-dupe preserving order
    seen = set()
    uniq = []
    for e in out:
        if e in seen:
            continue
        seen.add(e)
        uniq.append(e)
    return uniq


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_multiseed(data: dict) -> bool:
    return data.get("benchmark") == "greedy_vs_training_by_budget_multiseed"


def _metric_value(row: dict, metric: str) -> float | None:
    if metric not in row:
        return None
    val = row[metric]
    if isinstance(val, dict):
        return float(val["mean"]) if val.get("mean") is not None else None
    return float(val) if val is not None else None


def _metric_std(row: dict, metric: str) -> float | None:
    val = row.get(metric)
    if isinstance(val, dict) and val.get("std") is not None:
        return float(val["std"])
    return None


def _series_points(
    data: dict, section: str, key: str, metric: str
) -> tuple[list[int], list[float], list[float | None]]:
    budgets = data.get("budgets", CRAFTAX_BUDGET_LABELS)
    block = data.get(section, {}).get(key, {})
    xs: list[int] = []
    ys: list[float] = []
    ystd: list[float | None] = []
    for budget in budgets:
        row = block.get(budget)
        if not row:
            continue
        y = _metric_value(row, metric)
        if y is None:
            continue
        xs.append(int(row.get("total_timesteps") or CRAFTAX_BUDGET_STEPS[budget]))
        ys.append(y)
        ystd.append(_metric_std(row, metric))
    return xs, ys, ystd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model growth across env-step budgets")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metric",
        choices=("greedy_eval_pct_max", "training_return_pct_max"),
        default="greedy_eval_pct_max",
        help="Y-axis metric (% of max achievement reward)",
    )
    parser.add_argument(
        "--multiseed",
        action="store_true",
        help="Input is multiseed aggregate JSON; plot mean ± std bands",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated series to plot. Examples: classic_trio | "
            "flat_512,meta,meta_no_mask | classic.flat_512,classic.meta"
        ),
    )
    parser.add_argument("--log-x", action="store_true", default=True)
    parser.add_argument("--no-log-x", action="store_false", dest="log_x")
    args = parser.parse_args()

    data = _load(args.input)
    multiseed = args.multiseed or _is_multiseed(data)
    series = _select_series(args.models)
    metric_label = (
        "Greedy eval (% of max achievement reward)"
        if args.metric == "greedy_eval_pct_max"
        else "Training return (% of max achievement reward)"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (section, key, label) in enumerate(series):
        xs, ys, ystd = _series_points(data, section, key, args.metric)
        if not xs:
            continue
        style = "--" if section == "extended" else "-"
        color = colors[i % len(colors)]
        ax.plot(
            xs,
            ys,
            linestyle=style,
            marker="o",
            linewidth=2,
            markersize=6,
            color=color,
            label=label,
        )
        if multiseed and any(s is not None for s in ystd):
            lower = [y - (s or 0.0) for y, s in zip(ys, ystd)]
            upper = [y + (s or 0.0) for y, s in zip(ys, ystd)]
            ax.fill_between(xs, lower, upper, color=color, alpha=0.15, linewidth=0)

    if args.log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric_label)
    if multiseed:
        seeds = data.get("seeds", [])
        title = f"Classic baselines by budget (seeds {seeds}, mean ± std)"
        if args.models in (None, "classic_trio", "trio"):
            pass
        elif args.models:
            title = f"Growth by budget (seeds {seeds}, mean ± std)"
    else:
        title = f"Baseline growth by budget (seed {data.get('seed', 42)})"
    if args.models in ("classic_trio", "trio") or (
        args.models
        and set(args.models.replace(" ", "").split(","))
        <= {"flat_512", "meta", "meta_no_mask", "meta_mask", "meta_nomask", "meta_unmasked"}
    ):
        title = (
            f"Classic flat 512 vs meta (seeds {data.get('seeds', [data.get('seed')])}, mean ± std)"
            if multiseed
            else f"Classic flat 512 vs meta (seed {data.get('seed', 42)})"
        )
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
