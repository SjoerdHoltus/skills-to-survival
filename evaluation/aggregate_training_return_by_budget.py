#!/usr/bin/env python3
"""Aggregate training_return_metrics.json across env-step budgets."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from glob import glob

from ppo.configs import (
    CRAFTAX_BUDGET_LABELS,
    CRAFTAX_BUDGET_STEPS,
    num_updates_for_budget,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS_ROOT = os.path.join(_PROJECT_ROOT, "runs")

_BUDGET_SUFFIX = r"[a-z0-9]+"
POLICY_PATTERNS = {
    "extended_flat_512": re.compile(rf"^baselines_extended_ppo_(?P<budget>{_BUDGET_SUFFIX})$"),
    "classic_flat": re.compile(rf"^baselines_classic_flat_ppo_(?P<budget>{_BUDGET_SUFFIX})$"),
    "classic_meta": re.compile(rf"^baselines_classic_meta_ppo_(?P<budget>{_BUDGET_SUFFIX})$"),
    "classic_meta_no_mask": re.compile(
        rf"^baselines_classic_meta_ppo_no_mask_(?P<budget>{_BUDGET_SUFFIX})$"
    ),
    "classic_flat_rnn": re.compile(
        rf"^baselines_classic_flat_ppo_rnn_(?P<budget>{_BUDGET_SUFFIX})$"
    ),
    "classic_meta_rnn_no_mask": re.compile(
        rf"^baselines_classic_meta_ppo_rnn_no_mask_(?P<budget>{_BUDGET_SUFFIX})$"
    ),
}


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _runs_for_seed(policy_dir: str, seed: int) -> list[str]:
    candidates = sorted(glob(os.path.join(policy_dir, "run_*")))
    matches = []
    for run_dir in candidates:
        cfg_path = os.path.join(run_dir, "config.json")
        metrics_path = os.path.join(run_dir, "training_return_metrics.json")
        if not os.path.isfile(cfg_path) or not os.path.isfile(metrics_path):
            continue
        cfg = _read_json(cfg_path)
        hyp = cfg.get("hyperparameters") or {}
        if int(hyp.get("seed", -1)) != seed:
            continue
        matches.append(run_dir)
    return matches


def _latest_run_for_seed(policy_dir: str, seed: int) -> str | None:
    runs = _runs_for_seed(policy_dir, seed)
    return runs[-1] if runs else None


COMPARE_1M_CLASSIC_JSON = os.path.join(
    RUNS_ROOT,
    "benchmark_reports",
    "baseline_migration",
    "classic_flat_vs_meta_training_return_seed42.json",
)


def _inject_1m_classic_from_compare(found: dict, seed: int) -> None:
    if not os.path.isfile(COMPARE_1M_CLASSIC_JSON):
        return
    data = _read_json(COMPARE_1M_CLASSIC_JSON)
    if int(data.get("seed", -1)) != seed:
        return
    mapping = {
        "flat_512": "flat_512",
        "flat_256": "flat_256",
        "flat_128": "flat_128",
        "meta": "meta",
        "meta_no_mask": "meta_no_mask",
    }
    for src_key, dst_key in mapping.items():
        row = data.get(src_key)
        if not row:
            continue
        found["classic"].setdefault(dst_key, {})
        if "1m" in found["classic"][dst_key]:
            continue
        found["classic"][dst_key]["1m"] = {
            "run_dir": row.get("run_dir"),
            "budget": "1m",
            "total_timesteps": 1_000_000,
            "num_updates": row.get("num_updates"),
            "last_update": row.get("last_update"),
            "tail_mean": row.get("tail_mean"),
            "training_elapsed_sec": row.get("training_elapsed_sec"),
            "layer_size": row.get("layer_size"),
            "use_action_mask": row.get("use_action_mask"),
            "algorithm": "official_ppo" if "flat" in dst_key else "official_meta_ppo",
            "source": "classic_flat_vs_meta_training_return_seed42.json",
        }


def _collect_runs(seed: int, budgets: list[str]) -> dict:
    found: dict[str, dict[str, dict]] = {
        "extended": {},
        "classic": {},
    }

    for name in os.listdir(RUNS_ROOT):
        policy_dir = os.path.join(RUNS_ROOT, name)
        if not os.path.isdir(policy_dir):
            continue

        for key, pattern in POLICY_PATTERNS.items():
            m = pattern.match(name)
            if not m:
                continue
            budget = m.group("budget")
            if budget not in budgets:
                continue
            run_dir = _latest_run_for_seed(policy_dir, seed)
            if not run_dir:
                continue
            metrics = _read_json(os.path.join(run_dir, "training_return_metrics.json"))
            cfg = _read_json(os.path.join(run_dir, "config.json"))
            hyp = cfg.get("hyperparameters") or {}
            entry = {
                "run_dir": run_dir,
                "budget": budget,
                "total_timesteps": int(
                    metrics.get("total_timesteps", hyp.get("total_timesteps", 0))
                ),
                "num_updates": metrics.get("num_updates"),
                "last_update": metrics.get("last_update"),
                "tail_mean": metrics.get("tail_mean"),
                "training_elapsed_sec": metrics.get("training_elapsed_sec"),
                "layer_size": hyp.get("layer_size"),
                "use_action_mask": hyp.get("use_action_mask"),
                "algorithm": cfg.get("algorithm"),
            }

            if key == "extended_flat_512":
                found["extended"][budget] = entry
            elif key == "classic_flat":
                continue
            elif key == "classic_meta":
                found["classic"].setdefault("meta", {})[budget] = entry
            elif key == "classic_meta_no_mask":
                found["classic"].setdefault("meta_no_mask", {})[budget] = entry
            elif key == "classic_flat_rnn":
                found["classic"].setdefault("flat_rnn", {})[budget] = entry
            elif key == "classic_meta_rnn_no_mask":
                found["classic"].setdefault("meta_rnn_no_mask", {})[budget] = entry

    # Classic flat: multiple widths share one policy dir per budget.
    for name in os.listdir(RUNS_ROOT):
        m = POLICY_PATTERNS["classic_flat"].match(name)
        if not m:
            continue
        budget = m.group("budget")
        if budget not in budgets:
            continue
        policy_dir = os.path.join(RUNS_ROOT, name)
        for run_dir in _runs_for_seed(policy_dir, seed):
            metrics = _read_json(os.path.join(run_dir, "training_return_metrics.json"))
            cfg = _read_json(os.path.join(run_dir, "config.json"))
            hyp = cfg.get("hyperparameters") or {}
            width = int(hyp.get("layer_size", 512))
            found["classic"].setdefault(f"flat_{width}", {})[budget] = {
                "run_dir": run_dir,
                "budget": budget,
                "total_timesteps": int(
                    metrics.get("total_timesteps", hyp.get("total_timesteps", 0))
                ),
                "num_updates": metrics.get("num_updates"),
                "last_update": metrics.get("last_update"),
                "tail_mean": metrics.get("tail_mean"),
                "training_elapsed_sec": metrics.get("training_elapsed_sec"),
                "layer_size": width,
                "use_action_mask": hyp.get("use_action_mask"),
                "algorithm": cfg.get("algorithm"),
            }

    _inject_1m_classic_from_compare(found, seed)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate training-return metrics across env-step budgets"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budgets",
        default=",".join(CRAFTAX_BUDGET_LABELS),
        help=f"Comma-separated budgets (default: all {CRAFTAX_BUDGET_LABELS})",
    )
    parser.add_argument(
        "--output",
        default="runs/benchmark_reports/baseline_migration/training_return_by_budget_seed42.json",
    )
    args = parser.parse_args()

    budgets = [b.strip().lower() for b in args.budgets.split(",") if b.strip()]
    data = _collect_runs(args.seed, budgets)

    budget_updates = {
        label: num_updates_for_budget(CRAFTAX_BUDGET_STEPS[label])
        for label in budgets
        if label in CRAFTAX_BUDGET_STEPS
    }

    summary = {
        "benchmark": "training_return_by_budget",
        "seed": args.seed,
        "budgets": budgets,
        "budget_timesteps": {b: CRAFTAX_BUDGET_STEPS.get(b) for b in budgets},
        "budget_num_updates": budget_updates,
        "extended": {
            "env_name": "Craftax-Symbolic-v1",
            "max_achievement_reward_reference": 226.0,
            "flat_512": data["extended"],
        },
        "classic": {
            "env_name": "Craftax-Classic-Symbolic-v1",
            "max_achievement_reward_reference": 22.0,
            **data["classic"],
        },
        "missing": {
            "extended_flat_512": [b for b in budgets if b not in data["extended"]],
            "classic": {
                key: [b for b in budgets if b not in series]
                for key, series in data["classic"].items()
            },
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_path}\n")

    def print_series(title: str, series: dict[str, dict]) -> None:
        if not series:
            print(f"{title}: (no runs found)")
            return
        print(title)
        print(f"{'Budget':<6} {'Steps':>10} {'Updates':>8} {'Last %':>8} {'Tail-10 %':>10}")
        for budget in budgets:
            row = series.get(budget)
            if not row:
                print(f"{budget:<6} {'—':>10} {'—':>8} {'—':>8} {'—':>10}")
                continue
            last = row["last_update"]["return_pct_max"]
            tail = row["tail_mean"]["return_pct_max"]
            print(
                f"{budget:<6} {row['total_timesteps']:>10,} {row['num_updates']:>8} "
                f"{last:>7.2f}% {tail:>9.2f}%"
            )
        print()

    print_series("Extended flat PPO 512 (% of 226)", data["extended"])
    for key in sorted(data["classic"]):
        label = key.replace("_", " ")
        print_series(f"Classic {label} (% of 22)", data["classic"][key])


if __name__ == "__main__":
    main()
