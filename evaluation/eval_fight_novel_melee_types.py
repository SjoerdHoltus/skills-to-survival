"""Zero-shot Classic monster_fight on Full Craftax across melee type_ids.

Tests combat-controller transfer under novel Extended dynamics (HP/damage).
Obs does not expose type_id (melee channels are summed).

Types: 0=zombie (Classic), 1=gnome_warrior, 2=orc_soldier.

Usage:
  python -m evaluation.eval_fight_novel_melee_types --num-rollouts 256 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.constants import MOB_TYPE_HEALTH_MAPPING, MobType

from configs import FIND_WOOD_NUM_ACTIONS
from configs.monster_fight import (
    MONSTER_FIGHT_HIDDEN_SIZE,
    MONSTER_FIGHT_OBS_SIZE,
    MONSTER_FIGHT_ROLLOUT_STEPS,
)
from policies.monster_fight import MonsterFightPolicy
from ppo.configs import DEFAULT_SKILL_WEIGHTS, PROJECT_ROOT
from training.fitness_monster_fight import make_env, make_fitness_fn

MELEE_TYPE_NAMES = {
    0: "zombie",
    1: "gnome_warrior",
    2: "orc_soldier",
}


def _abs(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)


def _policy_shapes():
    policy = MonsterFightPolicy(
        hidden_size=MONSTER_FIGHT_HIDDEN_SIZE, num_actions=FIND_WOOD_NUM_ACTIONS
    )
    init = policy.init(jax.random.PRNGKey(0), jnp.zeros(MONSTER_FIGHT_OBS_SIZE))["params"]
    leaves, tree_def = jax.tree_util.tree_flatten(init)
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])
    return [leaf.shape for leaf in leaves], tree_def, flat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-rollouts", type=int, default=256)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--types", type=str, default="0,1,2")
    p.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_SKILL_WEIGHTS["monster_fight"],
        help="Classic fight .npy (default: hierarchy canonical).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="runs/benchmark_reports/fight_novel_melee_types",
    )
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    type_ids = [int(t) for t in args.types.split(",") if t.strip() != ""]
    weights_path = _abs(args.weights)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = jnp.asarray(np.load(weights_path), dtype=jnp.float32)
    shapes, treedef, template = _policy_shapes()
    if int(flat.shape[0]) != int(template.shape[0]):
        raise ValueError(f"Weight dim {flat.shape[0]} != policy {template.shape[0]}")
    untrained = jnp.zeros_like(flat)

    env, env_params = make_env(variant="craftax")

    rows: list[dict] = []
    for tid in type_ids:
        name = MELEE_TYPE_NAMES.get(tid, f"type_{tid}")
        hp = int(np.asarray(MOB_TYPE_HEALTH_MAPPING)[tid, int(MobType.MELEE.value)])
        print(f"\n[compile] melee_type={tid} ({name}) hp={hp}", flush=True)
        # Fresh JIT state per type — avoids PJRT arity glitches across closed-over type_ids.
        jax.clear_caches()
        fit = make_fitness_fn(
            env,
            env_params,
            shapes,
            treedef,
            rollout_steps=MONSTER_FIGHT_ROLLOUT_STEPS,
            num_rollouts=args.num_rollouts,
            hidden_size=MONSTER_FIGHT_HIDDEN_SIZE,
            num_actions=FIND_WOOD_NUM_ACTIONS,
            stochastic_actions=False,
            variant="craftax",
            melee_type_id=tid,
        )
        _ = fit(flat, jax.random.PRNGKey(999), jnp.int32(args.num_rollouts), jnp.float32(1.0))
        for weight_source, vec in (("classic", flat), ("untrained", untrained)):
            for seed in seeds:
                score, kill = fit(
                    vec,
                    jax.random.PRNGKey(seed),
                    jnp.int32(args.num_rollouts),
                    jnp.float32(1.0),
                )
                row = {
                    "melee_type_id": tid,
                    "melee_type_name": name,
                    "melee_hp": hp,
                    "weight_source": weight_source,
                    "seed": seed,
                    "score": float(score),
                    "kill_rate": float(kill),
                    "weights": weights_path if weight_source == "classic" else "",
                    "num_rollouts": args.num_rollouts,
                }
                rows.append(row)
                print(
                    f"  {weight_source:10s} type={tid} seed={seed} "
                    f"kill={float(kill):.3f} score={float(score):.2f}",
                    flush=True,
                )

    csv_path = out_dir / "raw_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for tid in type_ids:
        name = MELEE_TYPE_NAMES.get(tid, f"type_{tid}")
        hp = int(MOB_TYPE_HEALTH_MAPPING[tid, MobType.MELEE.value])
        entry = {"melee_type_id": tid, "melee_type_name": name, "melee_hp": hp}
        for src in ("classic", "untrained"):
            ks = [
                r["kill_rate"]
                for r in rows
                if r["melee_type_id"] == tid and r["weight_source"] == src
            ]
            ss = [
                r["score"] for r in rows if r["melee_type_id"] == tid and r["weight_source"] == src
            ]
            entry[f"{src}_kill_mean"] = float(np.mean(ks))
            entry[f"{src}_kill_std"] = float(np.std(ks))
            entry[f"{src}_score_mean"] = float(np.mean(ss))
            entry[f"{src}_score_std"] = float(np.std(ss))
        summary.append(entry)

    z0_row = next((s for s in summary if s["melee_type_id"] == 0), None)
    z0 = float(z0_row["classic_kill_mean"]) if z0_row is not None else float("nan")
    for s in summary:
        s["retention_vs_zombie"] = (
            float(s["classic_kill_mean"] / z0) if z0 == z0 and z0 > 1e-8 else float("nan")
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Classic fight → Full novel melee types (zero-shot)",
        "",
        "Classic hierarchy `monster_fight` weights, greedy Full bootstrap,",
        "moving melee, sword equipped. Obs sums melee channels (no type identity).",
        "",
        f"Weights: `{weights_path}`",
        f"Rollouts×seeds: {args.num_rollouts} × {seeds}",
        "",
        "| type_id | name | HP | Classic kill | Untrained kill | vs zombie |",
        "|--------:|------|---:|-------------:|---------------:|----------:|",
    ]
    for s in summary:
        report.append(
            f"| {s['melee_type_id']} | {s['melee_type_name']} | {s['melee_hp']} | "
            f"{s['classic_kill_mean']:.3f}±{s['classic_kill_std']:.3f} | "
            f"{s['untrained_kill_mean']:.3f}±{s['untrained_kill_std']:.3f} | "
            f"{s['retention_vs_zombie']:.2f} |"
        )
    report.append("")
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report), flush=True)
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
