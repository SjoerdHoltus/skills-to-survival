#!/usr/bin/env python3
"""Run official Craftax flat PPO baseline with local checkpoints."""

from __future__ import annotations

import argparse
import os

if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from ppo.common import (
    add_shared_cli,
    get_run_dir,
    namespace_from_preset,
    preset_overrides,
    run_official_train,
    save_run_config,
)
from ppo.configs import flat_ppo_preset


def main() -> None:
    from ppo.craftax_baselines.ppo import make_train

    parser = argparse.ArgumentParser(description="Official Craftax flat PPO baseline")
    parser.add_argument(
        "--preset",
        choices=("extended", "classic"),
        default="extended",
        help="extended=Craftax-Symbolic-v1, classic=Craftax-Classic-Symbolic-v1",
    )
    add_shared_cli(parser)
    args = parser.parse_args()

    preset = flat_ppo_preset(
        env=args.preset,
        budget=args.budget,
        total_timesteps=args.total_timesteps,
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
            "algorithm": "official_ppo",
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
