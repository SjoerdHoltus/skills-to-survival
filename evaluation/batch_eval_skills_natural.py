#!/usr/bin/env python3
"""Batch-eval frozen skills on unmodified Craftax Classic maps (no bootstrap spawners).

Greedy skill-only rollouts. Reports unconditional and FOV-conditional success rates.

Example:
  JAX_PLATFORMS=cpu python -m evaluation.batch_eval_skills_natural \\
    --num-seeds 1024 --batch-size 64 --steps 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

if any(a == "--cpu" for a in sys.argv):
    os.environ["JAX_PLATFORMS"] = "cpu"


import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from configs import (
    COLLECT_ORE_FEAT_HAS_TARGET,
    COLLECT_ORE_HIDDEN_SIZE,
    COLLECT_ORE_NUM_ACTIONS,
    COLLECT_ORE_OBS_SIZE,
    COLLECT_ORE_TRAINING_TARGET_ID,
    COLLECT_STONE_HIDDEN_SIZE,
    COLLECT_STONE_NUM_ACTIONS,
    COLLECT_STONE_OBS_SIZE,
    DRINK_WATER_HIDDEN_SIZE,
    DRINK_WATER_OBS_SIZE,
    FIND_FOOD_HIDDEN_SIZE,
    FIND_FOOD_OBS_SIZE,
    FIND_WOOD_HIDDEN_SIZE,
    FIND_WOOD_NUM_ACTIONS,
    FIND_WOOD_OBS_SIZE,
    MONSTER_FIGHT_HIDDEN_SIZE,
    MONSTER_FIGHT_OBS_SIZE,
)
from policies.collect_ore import CollectOrePolicy
from policies.collect_stone import CollectStonePolicy
from policies.drink_water import DrinkWaterPolicy
from policies.find_food import FindFoodPolicy
from policies.find_wood import FindWoodPolicy
from policies.monster_fight import MonsterFightPolicy
from ppo.configs import DEFAULT_SKILL_WEIGHTS as DEFAULT_WEIGHTS
from training.bootstrap_ore import make_collect_ore_obs
from training.bootstrap_stone import make_collect_stone_obs
from training.bootstrap_water import make_drink_water_obs
from training.bootstrap_wood import make_find_wood_obs
from training.monster_world_bootstrap import make_find_food_obs, make_monster_fight_obs
from training.natural_mining_setup import (
    setup_natural_collect_ore_state,
    setup_natural_collect_stone_state,
    setup_natural_drink_water_state,
    setup_natural_find_food_state,
    setup_natural_find_wood_state,
    setup_natural_monster_fight_state,
)
from training.params import pytree_to_flat_params

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKILL_ORDER = (
    "find_wood",
    "drink_water",
    "find_food",
    "monster_fight",
    "collect_stone",
    "collect_ore",
)


def _make_natural_env():
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    return env, env.default_params


def _load_flat(path: str, policy, obs_size: int):
    flat = jnp.load(path)
    init = policy.init(jax.random.PRNGKey(0), jnp.zeros(obs_size))["params"]
    flat_init, shapes, tree_def = pytree_to_flat_params(init)
    if int(flat.shape[0]) != int(flat_init.shape[0]):
        raise ValueError(
            f"Weights length {flat.shape[0]} != policy {flat_init.shape[0]} for {path}"
        )
    return flat, shapes, tree_def


def _cow_kills(prev_cows, post_cows) -> jnp.ndarray:
    flip = jnp.logical_and(prev_cows.mask, jnp.logical_not(post_cows.mask))
    dead = post_cows.health.astype(jnp.int32) <= 0
    return jnp.logical_and(flip, dead).astype(jnp.int32).sum().astype(jnp.float32)


def _hostile_kills(prev_z, post_z, prev_s, post_s) -> jnp.ndarray:
    z_flip = jnp.logical_and(prev_z.mask, jnp.logical_not(post_z.mask))
    z_dead = post_z.health.astype(jnp.int32) <= 0
    s_flip = jnp.logical_and(prev_s.mask, jnp.logical_not(post_s.mask))
    s_dead = post_s.health.astype(jnp.int32) <= 0
    z_k = jnp.logical_and(z_flip, z_dead).astype(jnp.int32).sum()
    s_k = jnp.logical_and(s_flip, s_dead).astype(jnp.int32).sum()
    return (z_k + s_k).astype(jnp.float32)


def _make_rollout_fn(
    *,
    skill: str,
    env,
    env_params,
    shapes,
    tree_def,
    policy_ctor,
    hidden_size: int,
    num_actions: int,
    make_obs,
    setup_state,
    target_idx: int,
    max_steps: int,
    ore_target_id: int = 0,
):
    from training.params import flat_params_to_pytree

    @jax.jit
    def single_rollout(flat_params, rollout_key):
        params = flat_params_to_pytree(flat_params, shapes, tree_def)
        carry_key, reset_key = jax.random.split(rollout_key)
        _, state = env.reset(reset_key, env_params)
        if skill == "collect_ore":
            state = setup_state(state, target_id=jnp.int32(ore_target_id))
        else:
            state = setup_state(state)

        if skill == "collect_ore":
            obs = make_obs(env, state, jnp.int32(ore_target_id))
        else:
            obs = make_obs(env, state)
        target_in_fov0 = (obs[target_idx] >= jnp.float32(0.5)).astype(jnp.float32)

        init_wood = state.inventory.wood.astype(jnp.float32)
        init_drink = state.player_drink.astype(jnp.float32)
        init_food = state.player_food.astype(jnp.float32)
        init_stone = state.inventory.stone.astype(jnp.float32)
        init_coal = state.inventory.coal.astype(jnp.float32)
        init_iron = state.inventory.iron.astype(jnp.float32)

        def step_fn(carry, _):
            obs, state, rng, episode_done, kills, saw_target = carry
            rng, env_rng = jax.random.split(rng)
            logits = policy_ctor(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )
            action = jnp.argmax(logits) + 1

            prev_c = state.cows
            prev_z = state.zombies
            prev_s = state.skeletons
            _new_obs_raw, new_state, _r, done, _info = env.step(env_rng, state, action, env_params)
            active = jnp.logical_not(episode_done).astype(jnp.float32)
            step_kills = jnp.float32(0.0)
            if skill == "find_food":
                step_kills = _cow_kills(prev_c, new_state.cows) * active
            elif skill == "monster_fight":
                step_kills = (
                    _hostile_kills(prev_z, new_state.zombies, prev_s, new_state.skeletons) * active
                )

            if skill == "collect_ore":
                new_obs = make_obs(env, new_state, jnp.int32(ore_target_id))
            else:
                new_obs = make_obs(env, new_state)
            saw_now = (new_obs[target_idx] >= jnp.float32(0.5)).astype(jnp.float32)
            episode_done = jnp.logical_or(episode_done, done)
            return (
                new_obs,
                new_state,
                rng,
                episode_done,
                kills + step_kills,
                jnp.maximum(saw_target, saw_now),
            ), None

        (_, final_state, _, _done, total_kills, ever_saw), _ = jax.lax.scan(
            step_fn,
            (
                obs,
                state,
                carry_key,
                jnp.array(False),
                jnp.float32(0.0),
                target_in_fov0,
            ),
            None,
            length=max_steps,
        )

        wood_gain = jnp.maximum(final_state.inventory.wood.astype(jnp.float32) - init_wood, 0.0)
        drink_gain = jnp.maximum(final_state.player_drink.astype(jnp.float32) - init_drink, 0.0)
        food_gain = jnp.maximum(final_state.player_food.astype(jnp.float32) - init_food, 0.0)
        stone_gain = jnp.maximum(final_state.inventory.stone.astype(jnp.float32) - init_stone, 0.0)
        coal_gain = jnp.maximum(final_state.inventory.coal.astype(jnp.float32) - init_coal, 0.0)
        iron_gain = jnp.maximum(final_state.inventory.iron.astype(jnp.float32) - init_iron, 0.0)
        died = (final_state.player_health.astype(jnp.float32) <= 0).astype(jnp.float32)

        if skill == "find_wood":
            success = (wood_gain >= 1.0).astype(jnp.float32)
            amount = wood_gain
        elif skill == "drink_water":
            success = (drink_gain > 0.0).astype(jnp.float32)
            amount = drink_gain
        elif skill == "find_food":
            success = jnp.logical_or(total_kills >= 1.0, food_gain > 0.0).astype(jnp.float32)
            amount = total_kills
        elif skill == "monster_fight":
            success = (total_kills >= 1.0).astype(jnp.float32)
            amount = total_kills
        elif skill == "collect_stone":
            success = (stone_gain >= 1.0).astype(jnp.float32)
            amount = stone_gain
        elif skill == "collect_ore":
            # Training target is coal; also count iron as a secondary ore success.
            success = jnp.logical_or(coal_gain >= 1.0, iron_gain >= 1.0).astype(jnp.float32)
            amount = coal_gain + iron_gain
        else:
            raise ValueError(skill)

        return target_in_fov0, ever_saw, success, amount, died

    return single_rollout


def _skill_spec(skill: str):
    if skill == "find_wood":
        return dict(
            policy_ctor=FindWoodPolicy,
            hidden=FIND_WOOD_HIDDEN_SIZE,
            actions=FIND_WOOD_NUM_ACTIONS,
            obs_size=FIND_WOOD_OBS_SIZE,
            make_obs=make_find_wood_obs,
            setup=setup_natural_find_wood_state,
            target_idx=2,
            success_name="Δwood ≥ 1",
        )
    if skill == "drink_water":
        return dict(
            policy_ctor=DrinkWaterPolicy,
            hidden=DRINK_WATER_HIDDEN_SIZE,
            actions=FIND_WOOD_NUM_ACTIONS,
            obs_size=DRINK_WATER_OBS_SIZE,
            make_obs=make_drink_water_obs,
            setup=setup_natural_drink_water_state,
            target_idx=2,
            success_name="Δdrink > 0",
        )
    if skill == "find_food":
        return dict(
            policy_ctor=FindFoodPolicy,
            hidden=FIND_FOOD_HIDDEN_SIZE,
            actions=FIND_WOOD_NUM_ACTIONS,
            obs_size=FIND_FOOD_OBS_SIZE,
            make_obs=make_find_food_obs,
            setup=setup_natural_find_food_state,
            target_idx=2,
            success_name="≥1 cow kill or Δfood > 0",
        )
    if skill == "monster_fight":
        return dict(
            policy_ctor=MonsterFightPolicy,
            hidden=MONSTER_FIGHT_HIDDEN_SIZE,
            actions=FIND_WOOD_NUM_ACTIONS,
            obs_size=MONSTER_FIGHT_OBS_SIZE,
            make_obs=make_monster_fight_obs,
            setup=setup_natural_monster_fight_state,
            target_idx=3,
            success_name="≥1 hostile kill",
        )
    if skill == "collect_stone":
        return dict(
            policy_ctor=CollectStonePolicy,
            hidden=COLLECT_STONE_HIDDEN_SIZE,
            actions=COLLECT_STONE_NUM_ACTIONS,
            obs_size=COLLECT_STONE_OBS_SIZE,
            make_obs=make_collect_stone_obs,
            setup=setup_natural_collect_stone_state,
            target_idx=2,
            success_name="Δstone ≥ 1",
        )
    if skill == "collect_ore":
        return dict(
            policy_ctor=CollectOrePolicy,
            hidden=COLLECT_ORE_HIDDEN_SIZE,
            actions=COLLECT_ORE_NUM_ACTIONS,
            obs_size=COLLECT_ORE_OBS_SIZE,
            make_obs=make_collect_ore_obs,
            setup=setup_natural_collect_ore_state,
            target_idx=COLLECT_ORE_FEAT_HAS_TARGET,
            success_name="Δcoal or Δiron ≥ 1",
        )
    raise ValueError(skill)


def eval_skill(
    skill: str,
    weights_path: str,
    *,
    num_seeds: int,
    batch_size: int,
    max_steps: int,
    master_seed: int,
) -> dict:
    spec = _skill_spec(skill)
    env, env_params = _make_natural_env()
    policy = spec["policy_ctor"](hidden_size=spec["hidden"], num_actions=spec["actions"])
    flat, shapes, tree_def = _load_flat(weights_path, policy, spec["obs_size"])
    single = _make_rollout_fn(
        skill=skill,
        env=env,
        env_params=env_params,
        shapes=shapes,
        tree_def=tree_def,
        policy_ctor=spec["policy_ctor"],
        hidden_size=spec["hidden"],
        num_actions=spec["actions"],
        make_obs=spec["make_obs"],
        setup_state=spec["setup"],
        target_idx=spec["target_idx"],
        max_steps=max_steps,
        ore_target_id=int(COLLECT_ORE_TRAINING_TARGET_ID),
    )

    @jax.jit
    def eval_batch(keys):
        return jax.vmap(single, in_axes=(None, 0))(flat, keys)

    master = jax.random.PRNGKey(master_seed)
    all_keys = jax.random.split(master, num_seeds)

    fov0_parts, ever_parts, succ_parts, amt_parts, died_parts = [], [], [], [], []
    t0 = time.perf_counter()
    i = 0
    while i < num_seeds:
        bs = min(batch_size, num_seeds - i)
        chunk = all_keys[i : i + bs]
        if bs == batch_size:
            fov0, ever, succ, amt, died = eval_batch(chunk)
        else:
            fov0, ever, succ, amt, died = jax.vmap(single, in_axes=(None, 0))(flat, chunk)
        fov0_parts.append(np.asarray(fov0))
        ever_parts.append(np.asarray(ever))
        succ_parts.append(np.asarray(succ))
        amt_parts.append(np.asarray(amt))
        died_parts.append(np.asarray(died))
        i += bs

    elapsed = time.perf_counter() - t0
    fov0 = np.concatenate(fov0_parts) > 0.5
    ever = np.concatenate(ever_parts) > 0.5
    succ = np.concatenate(succ_parts) > 0.5
    amt = np.concatenate(amt_parts)
    died = np.concatenate(died_parts) > 0.5
    n = len(fov0)
    n_fov0 = int(fov0.sum())
    n_ever = int(ever.sum())
    succ_given_fov0 = float(succ[fov0].mean()) if n_fov0 else 0.0
    succ_given_ever = float(succ[ever].mean()) if n_ever else 0.0

    return {
        "skill": skill,
        "weights": os.path.abspath(weights_path),
        "success_definition": spec["success_name"],
        "num_seeds": n,
        "max_steps": int(max_steps),
        "master_seed": int(master_seed),
        "elapsed_sec": float(elapsed),
        "target_in_fov_at_reset_rate": float(fov0.mean()),
        "target_in_fov_at_reset_count": n_fov0,
        "ever_saw_target_rate": float(ever.mean()),
        "ever_saw_target_count": n_ever,
        "success_rate": float(succ.mean()),
        "success_count": int(succ.sum()),
        "success_rate_given_target_in_fov_at_reset": succ_given_fov0,
        "success_rate_given_ever_saw_target": succ_given_ever,
        "mean_amount": float(amt.mean()),
        "mean_amount_given_success": float(amt[succ].mean()) if succ.any() else 0.0,
        "death_rate": float(died.mean()),
    }


def main():
    p = argparse.ArgumentParser(description="Natural-map batch eval for frozen Meta-PPO skills")
    p.add_argument(
        "--skills",
        default=",".join(SKILL_ORDER),
        help="Comma-separated skill names",
    )
    p.add_argument("--num-seeds", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Greedy skill-only horizon on the natural map",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-json",
        default=None,
        help="Write aggregate JSON (default: runs/benchmark_reports/skill_natural_eval_*.json)",
    )
    p.add_argument("--cpu", action="store_true", help="Force JAX CPU")
    for skill in SKILL_ORDER:
        p.add_argument(f"--{skill.replace('_', '-')}-weights", default=None)
    args = p.parse_args()

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    for s in skills:
        if s not in DEFAULT_WEIGHTS:
            p.error(f"Unknown skill {s}; choose from {list(DEFAULT_WEIGHTS)}")

    results = []
    print(f"jax devices: {jax.devices()}")
    print(
        f"natural Craftax Classic · seeds={args.num_seeds} · steps={args.steps} · "
        f"batch={args.batch_size}"
    )
    print()

    for skill in skills:
        # argparse converts to find_wood_weights etc.
        key = skill + "_weights"
        path = getattr(args, key, None) or os.path.join(_PROJECT_ROOT, DEFAULT_WEIGHTS[skill])
        if not os.path.isfile(path):
            raise SystemExit(f"Missing weights for {skill}: {path}")
        print(f"=== {skill} ===")
        print(f"weights: {path}")
        row = eval_skill(
            skill,
            path,
            num_seeds=args.num_seeds,
            batch_size=args.batch_size,
            max_steps=args.steps,
            master_seed=args.seed,
        )
        results.append(row)
        print(f"success def:     {row['success_definition']}")
        print(
            f"FOV at reset:    {row['target_in_fov_at_reset_rate']:.3f} "
            f"({row['target_in_fov_at_reset_count']}/{row['num_seeds']})"
        )
        print(
            f"ever saw target: {row['ever_saw_target_rate']:.3f} "
            f"({row['ever_saw_target_count']}/{row['num_seeds']})"
        )
        print(
            f"success rate:    {row['success_rate']:.3f} "
            f"({row['success_count']}/{row['num_seeds']})"
        )
        print(f"success | FOV0:  {row['success_rate_given_target_in_fov_at_reset']:.3f}")
        print(f"success | ever:  {row['success_rate_given_ever_saw_target']:.3f}")
        print(f"mean amount:     {row['mean_amount']:.3f}")
        print(f"death rate:      {row['death_rate']:.3f}")
        print(f"elapsed:         {row['elapsed_sec']:.1f}s")
        print()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "Unmodified Craftax-Classic-Symbolic-v1 resets; no bootstrap spawners. "
            "Light inventory/meter staging only. Greedy skill-only rollouts."
        ),
        "num_seeds": int(args.num_seeds),
        "max_steps": int(args.steps),
        "master_seed": int(args.seed),
        "results": results,
    }

    if args.output_json:
        out = os.path.abspath(args.output_json)
    else:
        stamp = datetime.now(timezone.utc).strftime("%d%m%Y_%H%M%S")
        out_dir = os.path.join(_PROJECT_ROOT, "runs", "benchmark_reports")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"skill_natural_eval_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")

    print()
    print("--- Summary ---")
    print(f"{'skill':<16} {'FOV0%':>7} {'ever%':>7} {'succ%':>7} {'succ|ever%':>11} {'death%':>7}")
    for r in results:
        print(
            f"{r['skill']:<16} "
            f"{100 * r['target_in_fov_at_reset_rate']:6.1f}% "
            f"{100 * r['ever_saw_target_rate']:6.1f}% "
            f"{100 * r['success_rate']:6.1f}% "
            f"{100 * r['success_rate_given_ever_saw_target']:10.1f}% "
            f"{100 * r['death_rate']:6.1f}%"
        )


if __name__ == "__main__":
    main()
