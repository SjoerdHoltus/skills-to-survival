#!/usr/bin/env python3
"""Compare greedy eval vs training-return metrics across env-step budgets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from evaluation.budget_run_discovery import (
    discover_budget_runs,
    greedy_eval_path,
    greedy_pct_from_eval,
)
from ppo.configs import CRAFTAX_BUDGET_LABELS, CRAFTAX_BUDGET_STEPS, num_updates_for_budget

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPARE_1M_CLASSIC_JSON = os.path.join(
    _PROJECT_ROOT,
    "runs/benchmark_reports/baseline_migration/classic_flat_vs_meta_training_return_seed42.json",
)
TRAINING_AGG_TEMPLATE = os.path.join(
    _PROJECT_ROOT,
    "runs/benchmark_reports/baseline_migration/training_return_by_budget_seed{seed}.json",
)


def _training_agg_path(seed: int) -> str:
    return TRAINING_AGG_TEMPLATE.format(seed=seed)


def _read_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _training_pct(training_data: dict, section: str, policy_key: str, budget: str) -> float | None:
    if section == "extended":
        row = training_data.get("extended", {}).get("flat_512", {}).get(budget)
    else:
        row = training_data.get("classic", {}).get(policy_key, {}).get(budget)
    if not row:
        return None
    return float(row["last_update"]["return_pct_max"])


def _inject_1m_training_from_compare(training: dict, seed: int) -> None:
    data = _read_json(COMPARE_1M_CLASSIC_JSON)
    if not data or int(data.get("seed", -1)) != seed:
        return
    classic = training.setdefault("classic", {})
    for key in ("flat_512", "flat_128", "meta", "meta_no_mask"):
        row = data.get(key)
        if not row:
            continue
        classic.setdefault(key, {})["1m"] = {
            "last_update": row.get("last_update"),
            "tail_mean": row.get("tail_mean"),
        }
    ext = data.get("extended_flat_reference")
    if ext:
        training.setdefault("extended", {}).setdefault("flat_512", {})["1m"] = {
            "last_update": ext.get("last_update"),
            "tail_mean": ext.get("tail_mean"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare greedy vs training return by budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", default=",".join(CRAFTAX_BUDGET_LABELS))
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON (default: .../greedy_vs_training_by_budget_seed<seed>.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"runs/benchmark_reports/baseline_migration/"
            f"greedy_vs_training_by_budget_seed{args.seed}.json"
        )

    budgets = [b.strip().lower() for b in args.budgets.split(",") if b.strip()]
    jobs = discover_budget_runs(seed=args.seed, budgets=tuple(budgets))
    training = _read_json(_training_agg_path(args.seed))
    _inject_1m_training_from_compare(training, args.seed)

    extended: dict[str, dict] = {}
    classic: dict[str, dict] = {}

    for job in jobs:
        eval_path = greedy_eval_path(job["run_dir"])
        if not os.path.isfile(eval_path):
            continue
        ev = _read_json(eval_path)
        greedy_pct = greedy_pct_from_eval(ev, ev.get("env_name", ""))
        train_metrics_path = os.path.join(job["run_dir"], "training_return_metrics.json")
        train_row = _read_json(train_metrics_path) if os.path.isfile(train_metrics_path) else {}
        train_pct = (
            float(train_row["last_update"]["return_pct_max"])
            if train_row.get("last_update")
            else _training_pct(
                training,
                "extended"
                if job["policy_key"] == "flat_512" and "extended" in job["run_dir"]
                else "classic",
                job["policy_key"],
                job["budget"],
            )
        )
        entry = {
            "run_dir": job["run_dir"],
            "budget": job["budget"],
            "total_timesteps": CRAFTAX_BUDGET_STEPS.get(job["budget"]),
            "num_updates": num_updates_for_budget(CRAFTAX_BUDGET_STEPS[job["budget"]])
            if job["budget"] in CRAFTAX_BUDGET_STEPS
            else train_row.get("num_updates"),
            "greedy_eval_pct_max": greedy_pct,
            "greedy_eval_std": float(
                ev.get(
                    "return_pct_max_achievement_reward",
                    ev.get("return_pct_max_achievement_only", {}),
                ).get("std", 0.0)
            ),
            "training_return_pct_max": train_pct,
            "training_minus_greedy_pp": (
                float(train_pct) - greedy_pct if train_pct is not None else None
            ),
            "num_episodes": ev.get("num_episodes"),
            "eval_elapsed_sec": ev.get("eval_elapsed_sec"),
        }

        is_extended = "baselines_extended_ppo_" in job["run_dir"]
        if is_extended and job["policy_key"] == "flat_512":
            extended.setdefault(job["budget"], entry)
        else:
            classic.setdefault(job["policy_key"], {})[job["budget"]] = entry

    summary = {
        "benchmark": "greedy_vs_training_by_budget",
        "seed": args.seed,
        "budgets": budgets,
        "budget_timesteps": {b: CRAFTAX_BUDGET_STEPS.get(b) for b in budgets},
        "budget_num_updates": {
            b: num_updates_for_budget(CRAFTAX_BUDGET_STEPS[b])
            for b in budgets
            if b in CRAFTAX_BUDGET_STEPS
        },
        "greedy_eval": {"num_episodes": 128, "master_seed": 42, "policy": "argmax"},
        "extended": {
            "env_name": "Craftax-Symbolic-v1",
            "max_achievement_reward_reference": 226.0,
            "flat_512": extended,
        },
        "classic": {
            "env_name": "Craftax-Classic-Symbolic-v1",
            "max_achievement_reward_reference": 22.0,
            **classic,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}\n")

    def print_section(title: str, series: dict[str, dict], budgets: list[str]) -> None:
        if not series:
            return
        print(title)
        print(f"{'Budget':<6} {'Updates':>8} {'Greedy %':>10} {'Train %':>10} {'Train−Greedy':>14}")
        for budget in budgets:
            row = series.get(budget)
            if not row:
                print(f"{budget:<6} {'—':>8} {'—':>10} {'—':>10} {'—':>14}")
                continue
            g = row["greedy_eval_pct_max"]
            t = row["training_return_pct_max"]
            d = row["training_minus_greedy_pp"]
            d_s = f"{d:+.2f} pp" if d is not None else "—"
            t_s = f"{t:.2f}" if t is not None else "—"
            print(f"{budget:<6} {row.get('num_updates', '—'):>8} {g:>9.2f}% {t_s:>10} {d_s:>14}")
        print()

    print_section("Extended flat 512 (% of 226)", summary["extended"]["flat_512"], budgets)
    for key in sorted(classic.keys()):
        print_section(f"Classic {key.replace('_', ' ')} (% of 22)", classic[key], budgets)


if __name__ == "__main__":
    main()
