#!/usr/bin/env python3
"""Run greedy eval on all budget-sweep checkpoints."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from evaluation.budget_run_discovery import (
    discover_budget_runs,
    greedy_eval_filename,
    greedy_eval_matches_train,
    greedy_eval_path,
    train_use_action_mask,
)
from ppo.configs import CRAFTAX_BUDGET_LABELS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSIC_TRIO = {"flat_512", "meta", "meta_no_mask"}
# Thesis comparison set (no masked meta).
THESIS_QUARTET = {"flat_512", "meta_no_mask", "flat_rnn", "meta_rnn_no_mask"}


def _is_classic_job(job: dict) -> bool:
    return "baselines_classic" in job["run_dir"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy-eval all budget-sweep checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budgets",
        default=",".join(CRAFTAX_BUDGET_LABELS),
        help="Comma-separated budget labels",
    )
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument(
        "--policy-keys",
        default="",
        help="Comma-separated policy keys, or classic_trio / thesis_quartet "
        "(flat_512, meta_no_mask, flat_rnn, meta_rnn_no_mask).",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if eval JSON exists")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    budgets = tuple(b.strip().lower() for b in args.budgets.split(",") if b.strip())
    jobs = discover_budget_runs(seed=args.seed, budgets=budgets)
    if args.policy_keys.strip():
        wanted: set[str] = set()
        classic_only = False
        for tok in args.policy_keys.split(","):
            tok = tok.strip().lower()
            if not tok:
                continue
            if tok in ("classic_trio", "trio"):
                wanted |= CLASSIC_TRIO
                classic_only = True
            elif tok in ("thesis_quartet", "quartet"):
                wanted |= THESIS_QUARTET
                classic_only = True
            else:
                wanted.add(tok)
        jobs = [j for j in jobs if j["policy_key"] in wanted]
        if classic_only:
            jobs = [j for j in jobs if _is_classic_job(j)]

    print(
        f"Found {len(jobs)} checkpoint runs for seed {args.seed} "
        f"({args.num_episodes} ep, master_seed={args.master_seed})",
        flush=True,
    )

    env = os.environ.copy()
    if not args.cpu:
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    for i, job in enumerate(jobs, 1):
        out_path = greedy_eval_path(
            job["run_dir"],
            master_seed=args.master_seed,
            num_episodes=args.num_episodes,
        )
        if not args.force and greedy_eval_matches_train(
            job["run_dir"],
            job["eval_policy"],
            master_seed=args.master_seed,
            num_episodes=args.num_episodes,
        ):
            print(
                f"[{i}/{len(jobs)}] skip {job['policy_key']} {job['budget']} (exists)",
                flush=True,
            )
            continue

        mask_note = ""
        if job["eval_policy"] in ("meta", "meta_rnn"):
            train_mask = train_use_action_mask(job["run_dir"])
            mask_note = f", eval_mask={train_mask}"

        print(
            f"\n[{i}/{len(jobs)}] {job['policy_key']} @ {job['budget']} "
            f"-> {job['run_dir']}{mask_note}",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-m",
            "evaluation.benchmark_eval_official_ppo",
            "--run-dir",
            job["run_dir"],
            "--policy",
            job["eval_policy"],
            "--num-episodes",
            str(args.num_episodes),
            "--master-seed",
            str(args.master_seed),
            "--output",
            out_path,
            "--eval-action-mask",
            "auto",
        ]
        if args.cpu:
            cmd.append("--cpu")
        elif job["eval_policy"] in ("flat", "meta", "flat_rnn", "meta_rnn"):
            cmd.append("--fast")
        subprocess.run(cmd, check=True, env=env, cwd=_PROJECT_ROOT)

    print(f"\nDone. Eval files: evals/{greedy_eval_filename(args.master_seed, args.num_episodes)}")


if __name__ == "__main__":
    main()
