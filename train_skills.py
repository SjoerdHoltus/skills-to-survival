"""Stage 1: train the six frozen skills with OpenES.

Hunt Food and Combat use a two-phase curriculum: static mobs first, then moving
mobs warm-started from the phase-1 weights. The other skills train in one phase.

Examples::

  python train_skills.py
  python train_skills.py --only find_wood,drink_water
  python train_skills.py --variant craftax          # Craftax Extended (transfer study)
  python train_skills.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import time

from skills.collect_ore import train_collect_ore
from skills.collect_stone import train_collect_stone
from skills.drink_water import train_drink_water
from skills.find_food import train_find_food
from skills.find_wood import train_find_wood
from skills.monster_fight import train_monster_fight
from training.run_utils import RUNS_DIR
from training.skill_variant import normalize_variant, runs_policy_folder

# skill -> (train fn, OpenES defaults, (static-phase kwargs, moving-phase kwargs) or None)
SKILLS = {
    "find_wood": (train_find_wood, dict(pop_size=50, generations=100, num_rollouts=32), None),
    "drink_water": (train_drink_water, dict(pop_size=100, generations=100, num_rollouts=64), None),
    "find_food": (
        train_find_food,
        dict(pop_size=100, num_rollouts=128),
        ({"static_cow": True}, {"static_cow": False}),
    ),
    "monster_fight": (
        train_monster_fight,
        dict(pop_size=100, num_rollouts=128),
        (
            {"static_zombie": True, "static_skeleton": True},
            {"static_zombie": False, "static_skeleton": False},
        ),
    ),
    "collect_stone": (
        train_collect_stone,
        dict(pop_size=100, generations=100, num_rollouts=64),
        None,
    ),
    "collect_ore": (train_collect_ore, dict(pop_size=100, generations=200, num_rollouts=64), None),
}


def _latest_final_weights(policy_folder: str) -> str:
    paths = glob.glob(os.path.join(RUNS_DIR, policy_folder, "run_*", "weights", "final_*.npy"))
    if not paths:
        raise FileNotFoundError(f"No final_*.npy under {RUNS_DIR}/{policy_folder}/")
    return max(paths, key=os.path.getmtime)


def _run(label: str, train_fn, folder: str, dry_run: bool, **kwargs) -> str | None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    for key, value in sorted(kwargs.items()):
        print(f"  {key}: {value}")
    if dry_run:
        return None
    t0 = time.time()
    train_fn(**kwargs)
    weights = _latest_final_weights(folder)
    print(f"  done in {(time.time() - t0) / 60:.1f} min -> {weights}")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="Comma-separated subset of: " + ",".join(SKILLS))
    parser.add_argument("--pop-size", type=int, help="Override for all skills")
    parser.add_argument("--num-rollouts", type=int, help="Override for all skills")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--static-gens", type=int, default=100, help="Static-mob phase (food, combat)"
    )
    parser.add_argument(
        "--moving-gens", type=int, default=100, help="Moving-mob phase (food, combat)"
    )
    parser.add_argument("--variant", choices=("classic", "craftax"), default="classic")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only")
    args = parser.parse_args()

    selected = [s.strip() for s in args.only.split(",")] if args.only else list(SKILLS)
    unknown = set(selected) - set(SKILLS)
    if unknown:
        raise SystemExit(f"Unknown skills: {sorted(unknown)}. Choose from {list(SKILLS)}")

    variant = normalize_variant(args.variant)
    overrides = {"variant": variant, "seed": args.seed}
    if args.pop_size is not None:
        overrides["pop_size"] = args.pop_size
    if args.num_rollouts is not None:
        overrides["num_rollouts"] = args.num_rollouts

    t0 = time.time()
    for skill in selected:
        train_fn, defaults, curriculum = SKILLS[skill]
        folder = runs_policy_folder(skill, variant)
        kw = {**defaults, **overrides}
        if curriculum is None:
            _run(skill, train_fn, folder, args.dry_run, **kw)
            continue
        static_kw, moving_kw = curriculum
        phase1 = _run(
            f"{skill} - phase 1/2: static mobs",
            train_fn,
            folder,
            args.dry_run,
            **kw,
            **static_kw,
            generations=args.static_gens,
        )
        _run(
            f"{skill} - phase 2/2: moving mobs",
            train_fn,
            folder,
            args.dry_run,
            **kw,
            **moving_kw,
            generations=args.moving_gens,
            resume_from=phase1,
        )
    print(f"\nAll skills finished in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
