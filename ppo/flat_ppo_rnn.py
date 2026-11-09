#!/usr/bin/env python3
"""Run official Craftax flat PPO-RNN baseline with local checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import os

if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from ppo.common import (
    add_shared_cli,
    get_run_dir,
    namespace_from_preset,
    preset_overrides,
    run_official_train,
    save_run_config,
)
from ppo.configs import (
    CRAFTAX_1M_FLAT_RNN_CLASSIC_PRESET,
    CRAFTAX_1M_FLAT_RNN_PRESET,
    CRAFTAX_BUDGET_LABELS,
    flat_rnn_preset,
)


def main() -> None:
    from ppo.craftax_baselines.ppo_rnn import make_train

    parser = argparse.ArgumentParser(description="Official Craftax flat PPO-RNN baseline")
    parser.add_argument(
        "--preset",
        choices=("extended", "classic"),
        default="classic",
        help="extended=Craftax-Symbolic-v1, classic=Craftax-Classic-Symbolic-v1",
    )
    parser.add_argument(
        "--chunked-budgets",
        action="store_true",
        help="Train once to the final budget, checkpointing at each --budgets label",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(CRAFTAX_BUDGET_LABELS),
        help="Comma-separated budget labels for --chunked-budgets (default: all)",
    )
    add_shared_cli(parser)
    args = parser.parse_args()

    if args.chunked_budgets:
        budgets = tuple(b.strip().lower() for b in args.budgets.split(",") if b.strip())
        final = budgets[-1]
        preset = flat_rnn_preset(
            env=args.preset,
            budget=final,
            total_timesteps=args.total_timesteps,
        )
        ns = namespace_from_preset(preset, **preset_overrides(args, preset))
        if args.log_dir:
            run_dir = os.path.abspath(args.log_dir)
            start_time = os.path.basename(run_dir).replace("run_", "")
        else:
            start_time = dt.datetime.now().strftime("%m%d%y_%H%M%S")
        print(f"Chunked flat RNN: budgets={list(budgets)} seed={ns.seed}")
        from ppo.chunked_train import run_chunked_budget_train

        run_chunked_budget_train(
            make_train,
            ns,
            start_time=start_time,
            budgets=budgets,
            algorithm="official_ppo_rnn",
        )
        return

    preset = (
        flat_rnn_preset(
            env=args.preset,
            budget=args.budget,
            total_timesteps=args.total_timesteps,
        )
        if args.budget is not None or args.total_timesteps is not None
        else (
            CRAFTAX_1M_FLAT_RNN_PRESET
            if args.preset == "extended"
            else CRAFTAX_1M_FLAT_RNN_CLASSIC_PRESET
        )
    )
    ns = namespace_from_preset(preset, **preset_overrides(args, preset))

    if args.log_dir:
        run_dir = os.path.abspath(args.log_dir)
        start_time = os.path.basename(run_dir).replace("run_", "")
    else:
        run_dir, start_time = get_run_dir(ns.runs_policy)

    os.makedirs(run_dir, exist_ok=True)
    save_run_config(
        run_dir,
        {
            "task": ns.runs_policy,
            "algorithm": "official_ppo_rnn",
            "preset": preset,
            "hyperparameters": dict(ns.__dict__),
            "start_time": start_time,
        },
    )

    print(f"Run dir: {run_dir}")
    print(f"Env: {ns.env_name}, layer_size={ns.layer_size}, steps={ns.total_timesteps}")
    run_official_train(make_train, ns, run_dir)


if __name__ == "__main__":
    main()
