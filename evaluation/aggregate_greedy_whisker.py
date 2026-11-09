#!/usr/bin/env python3
"""Build thesis whisker/boxplot payloads from 256-ep (or N-ep) greedy evals.

Produces:
  - per training-seed mean % (for seed-level boxplots)
  - pooled per-episode % across seeds (for episode-level boxplots)
  - Tukey boxplot summaries
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from evaluation.budget_run_discovery import (
    discover_budget_runs,
    greedy_eval_path,
    greedy_pct_from_eval,
)
from training.craftax_classic_metrics import mean_std_across_seeds

CLASSIC_TRIO = ("flat_512", "meta", "meta_no_mask")
THESIS_QUARTET = ("flat_512", "meta_no_mask", "flat_rnn", "meta_rnn_no_mask")


def _boxplot(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    q1, med, q3 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    iqr = q3 - q1
    return {
        "min": float(arr.min()),
        "q1": q1,
        "median": med,
        "q3": q3,
        "max": float(arr.max()),
        "iqr": float(iqr),
        "whisker_low": float(max(arr.min(), q1 - 1.5 * iqr)),
        "whisker_high": float(min(arr.max(), q3 + 1.5 * iqr)),
        "n": int(arr.size),
    }


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate greedy evals for whisker plots")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--budget", default="20m")
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument(
        "--policy-keys",
        default="thesis_quartet",
        help="Comma-separated keys, or classic_trio / thesis_quartet",
    )
    parser.add_argument(
        "--output",
        default="runs/benchmark_reports/baseline_migration/greedy_whisker_thesis_quartet_20m_256ep.json",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    wanted: set[str] = set()
    for tok in args.policy_keys.split(","):
        tok = tok.strip().lower()
        if tok in ("classic_trio", "trio"):
            wanted.update(CLASSIC_TRIO)
        elif tok in ("thesis_quartet", "quartet"):
            wanted.update(THESIS_QUARTET)
        elif tok:
            wanted.add(tok)

    key_order = [k for k in THESIS_QUARTET if k in wanted] + [
        k for k in sorted(wanted) if k not in THESIS_QUARTET
    ]

    out: dict = {
        "benchmark": "greedy_eval_whisker",
        "budget": args.budget,
        "seeds": seeds,
        "num_episodes": args.num_episodes,
        "master_seed": args.master_seed,
        "policy": "argmax",
        "classic": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    for key in key_order:
        seed_means: list[float] = []
        seed_rows: list[dict] = []
        pooled_pct: list[float] = []
        for seed in seeds:
            jobs = [
                j
                for j in discover_budget_runs(seed=seed, budgets=(args.budget,))
                if j["policy_key"] == key and "baselines_classic" in j["run_dir"]
            ]
            if not jobs:
                continue
            job = jobs[-1]
            path = greedy_eval_path(
                job["run_dir"],
                master_seed=args.master_seed,
                num_episodes=args.num_episodes,
            )
            if not os.path.isfile(path):
                print(f"MISSING {key} seed={seed}: {path}", flush=True)
                continue
            ev = _read_json(path)
            mean_pct = greedy_pct_from_eval(ev, ev.get("env_name", ""))
            seed_means.append(float(mean_pct))
            ep_pct = ev.get("episode_return_pct_max") or []
            pooled_pct.extend(float(x) for x in ep_pct)
            seed_rows.append(
                {
                    "seed": seed,
                    "run_dir": job["run_dir"],
                    "eval_path": path,
                    "greedy_eval_pct_max": float(mean_pct),
                    "episode_return_pct_max": ep_pct,
                    "episode_return_pct_max_boxplot": ev.get("episode_return_pct_max_boxplot"),
                    "num_episodes": ev.get("num_episodes"),
                }
            )

        stats = mean_std_across_seeds(seed_means)
        out["classic"][key] = {
            "per_seed_mean_pct": seed_rows,
            "across_seeds": {
                "mean": float(stats["mean"]) if seed_means else None,
                "std": float(stats["std"]) if seed_means else None,
                "n": int(stats["n"]),
                "values": seed_means,
                "boxplot": _boxplot(seed_means),
            },
            "pooled_episodes_pct": {
                "n": len(pooled_pct),
                "mean": float(np.mean(pooled_pct)) if pooled_pct else None,
                "std": float(np.std(pooled_pct, ddof=1))
                if len(pooled_pct) > 1
                else (0.0 if pooled_pct else None),
                "boxplot": _boxplot(pooled_pct),
                # Full series for custom plotting (can be large: 5*256).
                "values": pooled_pct,
            },
        }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    for key, block in out["classic"].items():
        a = block["across_seeds"]
        p = block["pooled_episodes_pct"]["boxplot"]
        print(
            f"{key}: seeds n={a['n']} mean={a['mean']:.2f}±{a['std']:.2f} "
            f"| pooled episodes box med={p.get('median')} "
            f"[{p.get('whisker_low')}, {p.get('whisker_high')}]"
        )


if __name__ == "__main__":
    main()
