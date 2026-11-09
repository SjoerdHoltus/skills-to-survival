#!/usr/bin/env python3
"""Classic → Extended transfer eval with dual metrics.

For every (skill, weights, env, seed) records:

- ``score_native``: fitness used natively on that environment variant
- ``score_classic_formula``: Classic reward formula on the same rollouts/env
- ``behavioural``: shared behavioural metric (kill_rate for food/combat;
  resource amount for gather skills)
- ``behavioural_name``: which behavioural field was used

On Classic envs, native and classic-formula scores coincide for all skills.
On Extended food/combat they differ (damage-scaled native vs kill/binary Classic).

Writes ``runs/benchmark_reports/classic_to_extended_dual_metrics/``.
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

from configs import (
    COLLECT_ORE_HIDDEN_SIZE,
    COLLECT_ORE_NUM_ACTIONS,
    COLLECT_ORE_OBS_SIZE,
    COLLECT_ORE_ROLLOUT_STEPS,
    COLLECT_STONE_HIDDEN_SIZE,
    COLLECT_STONE_NUM_ACTIONS,
    COLLECT_STONE_OBS_SIZE,
    COLLECT_STONE_ROLLOUT_STEPS,
    DRINK_WATER_HIDDEN_SIZE,
    DRINK_WATER_OBS_SIZE,
    DRINK_WATER_ROLLOUT_STEPS,
    FIND_FOOD_HIDDEN_SIZE,
    FIND_FOOD_OBS_SIZE,
    FIND_FOOD_ROLLOUT_STEPS,
    FIND_WOOD_HIDDEN_SIZE,
    FIND_WOOD_NUM_ACTIONS,
    FIND_WOOD_OBS_SIZE,
    FIND_WOOD_ROLLOUT_STEPS,
    MONSTER_FIGHT_HIDDEN_SIZE,
    MONSTER_FIGHT_OBS_SIZE,
    MONSTER_FIGHT_ROLLOUT_STEPS,
)
from policies.collect_ore import CollectOrePolicy
from policies.collect_stone import CollectStonePolicy
from policies.drink_water import DrinkWaterPolicy
from policies.find_food import FindFoodPolicy
from policies.find_wood import FindWoodPolicy
from policies.monster_fight import MonsterFightPolicy
from ppo.configs import DEFAULT_SKILL_WEIGHTS
from training.bootstrap_ore import target_id_from_name
from training.fitness_collect_ore import (
    make_env as make_ore_env,
)
from training.fitness_collect_ore import (
    make_single_rollout_fn as make_ore_single_rollout,
)
from training.fitness_collect_stone import make_env as make_stone_env
from training.fitness_collect_stone import make_fitness_fn as make_stone_fitness
from training.fitness_drink_water import make_env as make_drink_env
from training.fitness_drink_water import make_fitness_fn as make_drink_fitness
from training.fitness_find_food import make_env as make_food_env
from training.fitness_find_food import make_fitness_fn as make_food_fitness
from training.fitness_find_wood import make_env as make_wood_env
from training.fitness_find_wood import make_fitness_fn as make_wood_fitness
from training.fitness_monster_fight import make_env as make_fight_env
from training.fitness_monster_fight import make_fitness_fn as make_fight_fitness

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FULL_CEILING = {
    "find_wood": "runs/find_wood_craftax/run_062026_140559_fitness_59.28/weights/final_062026_140559_59.28.npy",
    "drink_water": "runs/drink_water_craftax/run_062026_141048_fitness_58.77/weights/final_062026_141048_58.77.npy",
    "collect_stone": "runs/collect_stone_craftax/run_062026_141707_fitness_29.02/weights/final_062026_141707_29.02.npy",
    "find_food": "runs/find_food_craftax/run_080626_170026_fitness_86.40/weights/final_080626_170026_86.40.npy",
    "monster_fight": "runs/monster_fight_craftax/run_080626_171357_fitness_23.02/weights/final_080626_171357_23.02.npy",
}


def _abs(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)


def _shapes(policy_cls, obs_size: int, hidden: int, n_act: int):
    init = policy_cls(hidden_size=hidden, num_actions=n_act).init(
        jax.random.PRNGKey(0), jnp.zeros(obs_size)
    )["params"]
    leaves, tree_def = jax.tree_util.tree_flatten(init)
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in leaves])
    return [leaf.shape for leaf in leaves], tree_def, flat


def _mean_mask(vals, n, n_max):
    m = (jnp.arange(n_max) < n).astype(jnp.float32)
    d = jnp.maximum(m.sum(), 1.0)
    return (vals * m).sum() / d


def _run_gather(fitness_fn, flat, seed: int, num_rollouts: int) -> dict:
    out = fitness_fn(
        flat,
        jax.random.PRNGKey(seed),
        jnp.int32(num_rollouts),
        jnp.float32(1.0),
    )
    vals = [float(x) for x in jax.device_get(out)]
    # (score, success, reach, resource[, mined])
    score = vals[0]
    resource = vals[3]
    return {
        "score_native": score,
        "score_classic_formula": score,
        "behavioural": resource,
        "behavioural_name": "resource",
        "success": vals[1],
        "reach": vals[2],
        "resource": resource,
        **({"mined": vals[4]} if len(vals) > 4 else {}),
    }


def _run_combat_classic(fitness_fn, flat, seed: int, num_rollouts: int) -> dict:
    score, kill = [
        float(x)
        for x in jax.device_get(
            fitness_fn(
                flat,
                jax.random.PRNGKey(seed),
                jnp.int32(num_rollouts),
                jnp.float32(1.0),
            )
        )
    ]
    return {
        "score_native": score,
        "score_classic_formula": score,
        "behavioural": kill,
        "behavioural_name": "kill_rate",
        "kill_rate": kill,
    }


def _run_combat_dual(dual_fn, flat, seed: int, num_rollouts: int) -> dict:
    score_n, score_c, kill = [
        float(x)
        for x in jax.device_get(
            dual_fn(
                flat,
                jax.random.PRNGKey(seed),
                jnp.int32(num_rollouts),
                jnp.float32(1.0),
            )
        )
    ]
    return {
        "score_native": score_n,
        "score_classic_formula": score_c,
        "behavioural": kill,
        "behavioural_name": "kill_rate",
        "kill_rate": kill,
    }


def _build_gather(sk, variant: str, num_rollouts: int):
    shapes, tree_def, init_flat = _shapes(
        sk["policy_cls"], sk["obs_size"], sk["hidden"], sk["n_act"]
    )
    env, env_params = sk["make_env"](variant=variant)
    fitness_fn = sk["make_fitness"](
        env,
        env_params,
        shapes,
        tree_def,
        rollout_steps=sk["rollout_steps"],
        num_rollouts=num_rollouts,
        hidden_size=sk["hidden"],
        num_actions=sk["n_act"],
        stochastic_actions=False,
        variant=variant,
    )
    return fitness_fn, init_flat


def _build_combat(name: str, variant: str, num_rollouts: int):
    if name == "find_food":
        shapes, tree_def, init_flat = _shapes(
            FindFoodPolicy, FIND_FOOD_OBS_SIZE, FIND_FOOD_HIDDEN_SIZE, FIND_WOOD_NUM_ACTIONS
        )
        env, env_params = make_food_env(variant=variant)
        if variant == "craftax":
            from training.craftax.fitness_find_food import make_dual_fitness_fn

            dual = make_dual_fitness_fn(
                env,
                env_params,
                shapes,
                tree_def,
                rollout_steps=FIND_FOOD_ROLLOUT_STEPS,
                num_rollouts=num_rollouts,
                hidden_size=FIND_FOOD_HIDDEN_SIZE,
                num_actions=FIND_WOOD_NUM_ACTIONS,
                stochastic_actions=False,
                variant=variant,
            )
            return ("dual", dual, init_flat)
        fit = make_food_fitness(
            env,
            env_params,
            shapes,
            tree_def,
            rollout_steps=FIND_FOOD_ROLLOUT_STEPS,
            num_rollouts=num_rollouts,
            hidden_size=FIND_FOOD_HIDDEN_SIZE,
            num_actions=FIND_WOOD_NUM_ACTIONS,
            stochastic_actions=False,
            variant=variant,
        )
        return ("classic", fit, init_flat)

    shapes, tree_def, init_flat = _shapes(
        MonsterFightPolicy,
        MONSTER_FIGHT_OBS_SIZE,
        MONSTER_FIGHT_HIDDEN_SIZE,
        FIND_WOOD_NUM_ACTIONS,
    )
    env, env_params = make_fight_env(variant=variant)
    if variant == "craftax":
        from training.craftax.fitness_monster_fight import make_dual_fitness_fn

        dual = make_dual_fitness_fn(
            env,
            env_params,
            shapes,
            tree_def,
            rollout_steps=MONSTER_FIGHT_ROLLOUT_STEPS,
            num_rollouts=num_rollouts,
            hidden_size=MONSTER_FIGHT_HIDDEN_SIZE,
            num_actions=FIND_WOOD_NUM_ACTIONS,
            stochastic_actions=False,
            variant=variant,
        )
        return ("dual", dual, init_flat)
    fit = make_fight_fitness(
        env,
        env_params,
        shapes,
        tree_def,
        rollout_steps=MONSTER_FIGHT_ROLLOUT_STEPS,
        num_rollouts=num_rollouts,
        hidden_size=MONSTER_FIGHT_HIDDEN_SIZE,
        num_actions=FIND_WOOD_NUM_ACTIONS,
        stochastic_actions=False,
        variant=variant,
    )
    return ("classic", fit, init_flat)


def _build_ore(variant: str, target: str, wall: int, num_rollouts: int):
    shapes, tree_def, init_flat = _shapes(
        CollectOrePolicy,
        COLLECT_ORE_OBS_SIZE,
        COLLECT_ORE_HIDDEN_SIZE,
        COLLECT_ORE_NUM_ACTIONS,
    )
    env, env_params = make_ore_env(variant=variant)
    single = make_ore_single_rollout(
        env,
        env_params,
        shapes,
        tree_def,
        rollout_steps=COLLECT_ORE_ROLLOUT_STEPS,
        hidden_size=COLLECT_ORE_HIDDEN_SIZE,
        num_actions=COLLECT_ORE_NUM_ACTIONS,
        stochastic_actions=False,
        fixed_target_id=jnp.int32(target_id_from_name(target)),
        fixed_wall_count=jnp.int32(wall),
        variant=variant,
    )
    vmapped = jax.vmap(single, in_axes=(None, 0, None, None, None))

    @jax.jit
    def fitness_fn(flat_params, rng, active_num_rollouts=jnp.int32(num_rollouts)):
        keys = jax.random.split(rng, num_rollouts)
        out = vmapped(
            flat_params,
            keys,
            jnp.float32(1.0),
            jnp.int32(0),
            jnp.int32(1),
        )
        scores, success, _steps, amounts, reached, mined, shells, _fov = out
        n = jnp.clip(active_num_rollouts.astype(jnp.int32), 1, jnp.int32(num_rollouts))
        return {
            "score": _mean_mask(scores, n, num_rollouts),
            "success": _mean_mask(success, n, num_rollouts),
            "reach": _mean_mask(reached, n, num_rollouts),
            "resource": _mean_mask(amounts, n, num_rollouts),
            "mined": _mean_mask(mined, n, num_rollouts),
            "shell": _mean_mask(shells, n, num_rollouts),
        }

    return fitness_fn, init_flat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-rollouts", type=int, default=256)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument(
        "--out-dir",
        type=str,
        default="runs/benchmark_reports/classic_to_extended_dual_metrics",
    )
    p.add_argument("--only", type=str, default="")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    out_dir = Path(_abs(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_rows.csv"

    gather = [
        dict(
            name="find_wood",
            make_env=make_wood_env,
            make_fitness=make_wood_fitness,
            policy_cls=FindWoodPolicy,
            obs_size=FIND_WOOD_OBS_SIZE,
            hidden=FIND_WOOD_HIDDEN_SIZE,
            n_act=FIND_WOOD_NUM_ACTIONS,
            rollout_steps=FIND_WOOD_ROLLOUT_STEPS,
        ),
        dict(
            name="drink_water",
            make_env=make_drink_env,
            make_fitness=make_drink_fitness,
            policy_cls=DrinkWaterPolicy,
            obs_size=DRINK_WATER_OBS_SIZE,
            hidden=DRINK_WATER_HIDDEN_SIZE,
            n_act=FIND_WOOD_NUM_ACTIONS,
            rollout_steps=DRINK_WATER_ROLLOUT_STEPS,
        ),
        dict(
            name="collect_stone",
            make_env=make_stone_env,
            make_fitness=make_stone_fitness,
            policy_cls=CollectStonePolicy,
            obs_size=COLLECT_STONE_OBS_SIZE,
            hidden=COLLECT_STONE_HIDDEN_SIZE,
            n_act=COLLECT_STONE_NUM_ACTIONS,
            rollout_steps=COLLECT_STONE_ROLLOUT_STEPS,
        ),
    ]
    combat_names = ["find_food", "monster_fight"]

    rows: list[dict] = []
    done: set[tuple] = set()

    def _key(r: dict) -> tuple:
        return (
            r["skill"],
            r["weight_source"],
            r["variant"],
            str(r["seed"]),
            str(r.get("target") or ""),
            str(r.get("wall") or ""),
        )

    if args.resume and csv_path.is_file():
        rows = list(csv.DictReader(csv_path.open()))
        done = {_key(r) for r in rows}
        print(f"[resume] {len(rows)} rows", flush=True)

    def _flush() -> None:
        if not rows:
            return
        fields = sorted({k for r in rows for k in r})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _resolve_full(name: str) -> str | None:
        rel = _FULL_CEILING.get(name)
        if rel and os.path.isfile(_abs(rel)):
            return _abs(rel)
        return None

    # --- gather ---
    for sk in gather:
        if only and sk["name"] not in only:
            continue
        name = sk["name"]
        classic_w = _abs(DEFAULT_SKILL_WEIGHTS[name])
        full_w = _resolve_full(name)
        classic_flat = jnp.load(classic_w)
        full_flat = jnp.load(full_w) if full_w else None

        for variant in ("classic", "craftax"):
            print(f"\n[compile] {name} on {variant}", flush=True)
            fitness_fn, init_flat = _build_gather(sk, variant, args.num_rollouts)
            if variant == "classic":
                conditions = [("classic", classic_flat)]
            else:
                conditions = [("classic", classic_flat), ("untrained", init_flat)]
                if full_flat is not None:
                    conditions.append(("full", full_flat))

            for weight_source, flat in conditions:
                for seed in seeds:
                    key = (name, weight_source, variant, str(seed), "", "")
                    if key in done:
                        continue
                    print(
                        f"=== {name} | {weight_source}@{variant} | seed={seed} ===",
                        flush=True,
                    )
                    metrics = _run_gather(fitness_fn, flat, seed, args.num_rollouts)
                    row = {
                        "skill": name,
                        "weight_source": weight_source,
                        "variant": variant,
                        "seed": seed,
                        "target": "",
                        "wall": "",
                        "classic_weights": classic_w,
                        "full_weights": full_w or "",
                        **metrics,
                    }
                    rows.append(row)
                    done.add(key)
                    print(
                        f"  classic={metrics['score_classic_formula']:.3f}  "
                        f"native={metrics['score_native']:.3f}  "
                        f"{metrics['behavioural_name']}={metrics['behavioural']:.3f}",
                        flush=True,
                    )
            del fitness_fn
            jax.clear_caches()
            _flush()

    # --- combat ---
    for name in combat_names:
        if only and name not in only:
            continue
        classic_w = _abs(DEFAULT_SKILL_WEIGHTS[name])
        full_w = _resolve_full(name)
        classic_flat = jnp.load(classic_w)
        full_flat = jnp.load(full_w) if full_w else None

        for variant in ("classic", "craftax"):
            print(f"\n[compile] {name} on {variant}", flush=True)
            mode, fit, init_flat = _build_combat(name, variant, args.num_rollouts)
            if variant == "classic":
                conditions = [("classic", classic_flat)]
            else:
                conditions = [("classic", classic_flat), ("untrained", init_flat)]
                if full_flat is not None:
                    conditions.append(("full", full_flat))

            for weight_source, flat in conditions:
                for seed in seeds:
                    key = (name, weight_source, variant, str(seed), "", "")
                    if key in done:
                        continue
                    print(
                        f"=== {name} | {weight_source}@{variant} | seed={seed} ===",
                        flush=True,
                    )
                    if mode == "dual":
                        metrics = _run_combat_dual(fit, flat, seed, args.num_rollouts)
                    else:
                        metrics = _run_combat_classic(fit, flat, seed, args.num_rollouts)
                    row = {
                        "skill": name,
                        "weight_source": weight_source,
                        "variant": variant,
                        "seed": seed,
                        "target": "",
                        "wall": "",
                        "classic_weights": classic_w,
                        "full_weights": full_w or "",
                        **metrics,
                    }
                    rows.append(row)
                    done.add(key)
                    print(
                        f"  classic={metrics['score_classic_formula']:.3f}  "
                        f"native={metrics['score_native']:.3f}  "
                        f"kill_rate={metrics['behavioural']:.3f}",
                        flush=True,
                    )
                    _flush()
            del fit
            jax.clear_caches()
            _flush()

    # --- ore (coal wall=2 representative) ---
    if not only or "collect_ore" in only:
        name = "collect_ore"
        classic_w = _abs(DEFAULT_SKILL_WEIGHTS[name])
        classic_flat = jnp.load(classic_w)
        target, wall = "coal", 2
        for variant in ("classic", "craftax"):
            print(f"\n[compile] {name}/{target}/wall{wall} on {variant}", flush=True)
            fitness_fn, init_flat = _build_ore(variant, target, wall, args.num_rollouts)
            if variant == "classic":
                conditions = [("classic", classic_flat)]
            else:
                conditions = [("classic", classic_flat), ("untrained", init_flat)]

            for weight_source, flat in conditions:
                for seed in seeds:
                    key = (name, weight_source, variant, str(seed), target, str(wall))
                    if key in done:
                        continue
                    print(
                        f"=== {name} | {weight_source}@{variant} | "
                        f"{target}/w{wall} | seed={seed} ===",
                        flush=True,
                    )
                    raw = {
                        k: float(v)
                        for k, v in jax.device_get(
                            fitness_fn(
                                flat,
                                jax.random.PRNGKey(seed),
                                jnp.int32(args.num_rollouts),
                            )
                        ).items()
                    }
                    metrics = {
                        "score_native": raw["score"],
                        "score_classic_formula": raw["score"],
                        "behavioural": raw["resource"],
                        "behavioural_name": "resource",
                        **raw,
                    }
                    row = {
                        "skill": name,
                        "weight_source": weight_source,
                        "variant": variant,
                        "seed": seed,
                        "target": target,
                        "wall": wall,
                        "classic_weights": classic_w,
                        "full_weights": "",
                        **metrics,
                    }
                    rows.append(row)
                    done.add(key)
                    print(
                        f"  classic={metrics['score_classic_formula']:.3f}  "
                        f"resource={metrics['behavioural']:.3f}",
                        flush=True,
                    )
            del fitness_fn
            jax.clear_caches()
            _flush()

    # --- summary table ---
    by: dict[tuple, list] = {}
    for r in rows:
        key = (
            r["skill"],
            r["weight_source"],
            r["variant"],
            str(r.get("target") or ""),
            str(r.get("wall") or ""),
        )
        by.setdefault(key, []).append(r)

    summary_rows = []
    for key, rs in sorted(by.items()):
        skill, wsrc, variant, target, wall = key
        summary_rows.append(
            {
                "skill": skill,
                "weight_source": wsrc,
                "variant": variant,
                "target": target,
                "wall": wall,
                "n": len(rs),
                "score_native_mean": float(np.mean([float(r["score_native"]) for r in rs])),
                "score_classic_formula_mean": float(
                    np.mean([float(r["score_classic_formula"]) for r in rs])
                ),
                "behavioural_mean": float(np.mean([float(r["behavioural"]) for r in rs])),
                "behavioural_name": rs[0]["behavioural_name"],
            }
        )

    summary = {"conditions": summary_rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Transfer-oriented view for figure choice
    transfer = []
    skills = sorted({r["skill"] for r in summary_rows})
    for skill in skills:

        def _get(wsrc, variant):
            for r in summary_rows:
                if (
                    r["skill"] == skill
                    and r["weight_source"] == wsrc
                    and r["variant"] == variant
                    and (skill != "collect_ore" or (r["target"] == "coal" and r["wall"] == "2"))
                ):
                    return r
            return None

        cc = _get("classic", "classic")
        ce = _get("classic", "craftax")
        ue = _get("untrained", "craftax")
        ee = _get("full", "craftax")
        if not (cc and ce):
            continue
        transfer.append(
            {
                "skill": skill,
                "behavioural_name": ce["behavioural_name"],
                "classic_formula": {
                    "C@C": cc["score_classic_formula_mean"],
                    "C@E": ce["score_classic_formula_mean"],
                    "U@E": None if ue is None else ue["score_classic_formula_mean"],
                    "E@E": None if ee is None else ee["score_classic_formula_mean"],
                },
                "behavioural": {
                    "C@C": cc["behavioural_mean"],
                    "C@E": ce["behavioural_mean"],
                    "U@E": None if ue is None else ue["behavioural_mean"],
                    "E@E": None if ee is None else ee["behavioural_mean"],
                },
                "native_score": {
                    "C@C": cc["score_native_mean"],
                    "C@E": ce["score_native_mean"],
                    "U@E": None if ue is None else ue["score_native_mean"],
                    "E@E": None if ee is None else ee["score_native_mean"],
                },
            }
        )

    (out_dir / "transfer_choice.json").write_text(json.dumps({"transfer": transfer}, indent=2))
    report = [
        "# Classic → Extended dual-metric transfer",
        "",
        "Each skill has three comparable columns families:",
        "- `classic_formula`: Classic reward formula on that environment",
        "- `behavioural`: kill_rate (food/combat) or resource (gather)",
        "- `native_score`: environment-native fitness (differs for Extended food/combat)",
        "",
        "| Skill | Metric family | C@C | C@E | U@E | E@E |",
        "|-------|---------------|----:|----:|----:|----:|",
    ]
    for t in transfer:
        for family in ("classic_formula", "behavioural", "native_score"):
            d = t[family]

            def fmt(x):
                return "—" if x is None else f"{x:.3f}"

            label = family if family != "behavioural" else f"behavioural ({t['behavioural_name']})"
            report.append(
                f"| {t['skill']} | {label} | {fmt(d['C@C'])} | {fmt(d['C@E'])} | "
                f"{fmt(d['U@E'])} | {fmt(d['E@E'])} |"
            )
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {out_dir / 'transfer_choice.json'}")
    print(f"Wrote {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
