#!/usr/bin/env python3
"""Aggregate per-seed greedy vs training budget JSONs into mean ± std."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from ppo.configs import CRAFTAX_BUDGET_LABELS, CRAFTAX_BUDGET_STEPS, num_updates_for_budget
from training.craftax_classic_metrics import mean_std_across_seeds

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERIES = [
    ("extended", "flat_512"),
    ("classic", "flat_128"),
    ("classic", "flat_256"),
    ("classic", "flat_512"),
    ("classic", "meta"),
    ("classic", "meta_no_mask"),
    ("classic", "flat_rnn"),
    ("classic", "meta_rnn_no_mask"),
]

METRICS = (
    "greedy_eval_pct_max",
    "training_return_pct_max",
    "training_minus_greedy_pp",
)


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    stats = mean_std_across_seeds(values)
    return {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "n": int(stats["n"]),
        "values": values,
    }


def _per_seed_paths(report_dir: str, seeds: list[int]) -> list[str]:
    paths = []
    for seed in seeds:
        path = os.path.join(report_dir, f"greedy_vs_training_by_budget_seed{seed}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing per-seed report: {path}")
        paths.append(path)
    return paths


def aggregate_multiseed(per_seed_data: list[dict]) -> dict:
    seeds = [int(d["seed"]) for d in per_seed_data]
    budgets = per_seed_data[0].get("budgets", list(CRAFTAX_BUDGET_LABELS))
    out: dict = {
        "benchmark": "greedy_vs_training_by_budget_multiseed",
        "seeds": seeds,
        "budgets": budgets,
        "budget_timesteps": {b: CRAFTAX_BUDGET_STEPS.get(b) for b in budgets},
        "budget_num_updates": {
            b: num_updates_for_budget(CRAFTAX_BUDGET_STEPS[b])
            for b in budgets
            if b in CRAFTAX_BUDGET_STEPS
        },
        "greedy_eval": per_seed_data[0].get("greedy_eval", {}),
        "extended": {
            "env_name": per_seed_data[0]["extended"]["env_name"],
            "max_achievement_reward_reference": per_seed_data[0]["extended"][
                "max_achievement_reward_reference"
            ],
        },
        "classic": {
            "env_name": per_seed_data[0]["classic"]["env_name"],
            "max_achievement_reward_reference": per_seed_data[0]["classic"][
                "max_achievement_reward_reference"
            ],
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    for section, key in SERIES:
        section_out = out[section].setdefault(key, {})
        for budget in budgets:
            rows = []
            seed_ids = []
            if section == "extended":
                for d in per_seed_data:
                    row = d.get("extended", {}).get(key, {}).get(budget)
                    if row:
                        rows.append(row)
                        seed_ids.append(int(d["seed"]))
            else:
                for d in per_seed_data:
                    row = d.get("classic", {}).get(key, {}).get(budget)
                    if row:
                        rows.append(row)
                        seed_ids.append(int(d["seed"]))
            if not rows:
                continue
            entry = {
                "budget": budget,
                "total_timesteps": rows[0].get("total_timesteps")
                or CRAFTAX_BUDGET_STEPS.get(budget),
                "num_updates": rows[0].get("num_updates")
                or num_updates_for_budget(CRAFTAX_BUDGET_STEPS[budget])
                if budget in CRAFTAX_BUDGET_STEPS
                else None,
                "seeds_present": seed_ids,
            }
            for metric in METRICS:
                vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
                entry[metric] = _stats(vals)
            section_out[budget] = entry

    return out


def _print_tables(summary: dict) -> None:
    budgets = summary["budgets"]

    def print_section(title: str, block: dict) -> None:
        if not block:
            return
        print(title)
        print(f"{'Budget':<6} {'n':>3} {'Greedy %':>16} {'Train %':>16} {'Train−Greedy':>16}")
        for budget in budgets:
            row = block.get(budget)
            if not row:
                print(f"{budget:<6} {'—':>3} {'—':>16} {'—':>16} {'—':>16}")
                continue
            g = row["greedy_eval_pct_max"]
            t = row["training_return_pct_max"]
            d = row["training_minus_greedy_pp"]
            g_s = f"{g['mean']:.2f}±{g['std']:.2f}" if g["mean"] is not None else "—"
            t_s = f"{t['mean']:.2f}±{t['std']:.2f}" if t["mean"] is not None else "—"
            d_s = f"{d['mean']:+.2f}±{d['std']:.2f}" if d["mean"] is not None else "—"
            print(f"{budget:<6} {g['n']:>3} {g_s:>16} {t_s:>16} {d_s:>16}")
        print()

    print_section(
        "Extended flat 512 (% of 226)",
        summary["extended"].get("flat_512", {}),
    )
    for key in (
        "flat_128",
        "flat_256",
        "flat_512",
        "meta",
        "meta_no_mask",
        "flat_rnn",
        "meta_rnn_no_mask",
    ):
        print_section(
            f"Classic {key.replace('_', ' ')} (% of 22)",
            summary["classic"].get(key, {}),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate greedy vs training budget reports across seeds"
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated training seeds",
    )
    parser.add_argument(
        "--report-dir",
        default="runs/benchmark_reports/baseline_migration",
        help="Directory containing per-seed greedy_vs_training_by_budget_seed*.json",
    )
    parser.add_argument(
        "--output",
        default="runs/benchmark_reports/baseline_migration/greedy_vs_training_multiseed.json",
    )
    parser.add_argument(
        "--inputs",
        default=None,
        help="Optional comma-separated per-seed JSON paths (overrides --seeds/--report-dir)",
    )
    args = parser.parse_args()

    if args.inputs:
        paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        report_dir = os.path.join(_PROJECT_ROOT, args.report_dir)
        paths = _per_seed_paths(report_dir, seeds)

    per_seed_data = [_read_json(p) for p in paths]
    summary = aggregate_multiseed(per_seed_data)

    out_path = os.path.abspath(
        os.path.join(_PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}\n")
    _print_tables(summary)


if __name__ == "__main__":
    main()
