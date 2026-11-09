#!/usr/bin/env python3
"""Batch-eval frozen skills on unmodified full Craftax (Extended) maps.

Mirrors ``batch_eval_skills_natural.py`` (Classic) but uses Craftax-Symbolic-v1,
Full handcrafted obs, and Extended inventory/mob fields.

Example:
  JAX_PLATFORMS=cpu python -m evaluation.batch_eval_skills_natural_craftax \\
    --weight-set classic --num-seeds 1024 --batch-size 16 --steps 128 --cpu
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
from configs.craftax_base import CRAFTAX_ENV_NAME
from policies.collect_ore import CollectOrePolicy
from policies.collect_stone import CollectStonePolicy
from policies.drink_water import DrinkWaterPolicy
from policies.find_food import FindFoodPolicy
from policies.find_wood import FindWoodPolicy
from policies.monster_fight import MonsterFightPolicy
from ppo.configs import DEFAULT_SKILL_WEIGHTS as CLASSIC_WEIGHTS
from training.craftax.bootstrap import (
    make_collect_ore_obs,
    make_collect_stone_obs,
    make_drink_water_obs,
    make_find_food_obs,
    make_find_wood_obs,
    make_monster_fight_obs,
)
from training.craftax.natural_setup import (
    setup_natural_collect_ore_state,
    setup_natural_collect_stone_state,
    setup_natural_drink_water_state,
    setup_natural_find_food_state,
    setup_natural_find_wood_state,
    setup_natural_monster_fight_state,
)
from training.params import pytree_to_flat_params

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 50-gen Classic→Full fine-tunes (all six skills).
FINETUNE50_WEIGHTS = {
    "find_wood": (
        "runs/find_wood_craftax/run_080626_221027_fitness_43.49/"
        "weights/final_080626_221027_43.49.npy"
    ),
    "drink_water": (
        "runs/drink_water_craftax/run_080626_161543_fitness_62.70/"
        "weights/final_080626_161543_62.70.npy"
    ),
    "find_food": (
        "runs/find_food_craftax/run_080626_170026_fitness_86.40/"
        "weights/final_080626_170026_86.40.npy"
    ),
    "monster_fight": (
        "runs/monster_fight_craftax/run_080626_171357_fitness_23.02/"
        "weights/final_080626_171357_23.02.npy"
    ),
    "collect_stone": (
        "runs/collect_stone_craftax/run_080626_222021_fitness_32.95/"
        "weights/final_080626_222021_32.95.npy"
    ),
    "collect_ore": (
        "runs/collect_ore_craftax/run_080626_222651_fitness_11.31/"
        "weights/final_080626_222651_11.31.npy"
    ),
}

SKILL_ORDER = (
    "find_wood",
    "drink_water",
    "find_food",
    "monster_fight",
    "collect_stone",
    "collect_ore",
)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _make_natural_env():
    env = make_craftax_env_from_name(CRAFTAX_ENV_NAME, auto_reset=False)
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


def _lvl_passive(state):
    lvl = state.player_level.astype(jnp.int32)
    p = state.passive_mobs
    return p.position[lvl], p.mask[lvl], p.health[lvl]


def _lvl_melee(state):
    lvl = state.player_level.astype(jnp.int32)
    m = state.melee_mobs
    return m.mask[lvl], m.health[lvl]


def _lvl_ranged(state):
    lvl = state.player_level.astype(jnp.int32)
    r = state.ranged_mobs
    return r.mask[lvl], r.health[lvl]


def _passive_kills(prev_mask, prev_health, post_mask, post_health) -> jnp.ndarray:
    flip = jnp.logical_and(prev_mask, jnp.logical_not(post_mask))
    dead = post_health.astype(jnp.int32) <= 0
    return jnp.logical_and(flip, dead).astype(jnp.int32).sum().astype(jnp.float32)


def _hostile_kills(
    prev_m_mask,
    prev_m_hp,
    post_m_mask,
    post_m_hp,
    prev_r_mask,
    prev_r_hp,
    post_r_mask,
    post_r_hp,
) -> jnp.ndarray:
    """Melee + ranged body kills (Classic zombies + skeletons analogue)."""
    m_flip = jnp.logical_and(prev_m_mask, jnp.logical_not(post_m_mask))
    m_dead = post_m_hp.astype(jnp.int32) <= 0
    r_flip = jnp.logical_and(prev_r_mask, jnp.logical_not(post_r_mask))
    r_dead = post_r_hp.astype(jnp.int32) <= 0
    m_k = jnp.logical_and(m_flip, m_dead).astype(jnp.int32).sum()
    r_k = jnp.logical_and(r_flip, r_dead).astype(jnp.int32).sum()
    return (m_k + r_k).astype(jnp.float32)


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

            prev_p_pos, prev_p_mask, prev_p_hp = _lvl_passive(state)
            prev_m_mask, prev_m_hp = _lvl_melee(state)
            prev_r_mask, prev_r_hp = _lvl_ranged(state)
            _raw, new_state, _r, done, _info = env.step(env_rng, state, action, env_params)
            active = jnp.logical_not(episode_done).astype(jnp.float32)
            step_kills = jnp.float32(0.0)
            if skill == "find_food":
                _pp, post_p_mask, post_p_hp = _lvl_passive(new_state)
                step_kills = _passive_kills(prev_p_mask, prev_p_hp, post_p_mask, post_p_hp) * active
            elif skill == "monster_fight":
                post_m_mask, post_m_hp = _lvl_melee(new_state)
                post_r_mask, post_r_hp = _lvl_ranged(new_state)
                step_kills = (
                    _hostile_kills(
                        prev_m_mask,
                        prev_m_hp,
                        post_m_mask,
                        post_m_hp,
                        prev_r_mask,
                        prev_r_hp,
                        post_r_mask,
                        post_r_hp,
                    )
                    * active
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
            success_name="≥1 passive kill or Δfood > 0",
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
            success_name="≥1 melee/ranged kill",
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
            target_idx=2,
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
        print(f"  {skill}: {i}/{num_seeds}", flush=True)

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weight-set",
        choices=("classic", "finetune50"),
        default="classic",
        help="classic=DEFAULT Classic hierarchy; finetune50=50-gen Full FT (all six skills).",
    )
    p.add_argument("--skills", default="", help="Comma list (default: all in weight-set).")
    p.add_argument("--num-seeds", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", default=None)
    p.add_argument("--cpu", action="store_true")
    for skill in SKILL_ORDER:
        p.add_argument(f"--{skill.replace('_', '-')}-weights", default=None)
    args = p.parse_args()

    weight_map = CLASSIC_WEIGHTS if args.weight_set == "classic" else FINETUNE50_WEIGHTS
    if args.skills.strip():
        skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    else:
        skills = list(weight_map.keys())

    results = []
    print(f"jax devices: {jax.devices()}", flush=True)
    print(
        f"natural full Craftax · weight_set={args.weight_set} · "
        f"seeds={args.num_seeds} · steps={args.steps}",
        flush=True,
    )

    for skill in skills:
        if skill not in SKILL_ORDER:
            p.error(f"Unknown skill {skill}")
        override = getattr(args, f"{skill}_weights", None)
        rel = override or weight_map.get(skill)
        if rel is None:
            print(f"skip {skill}: no weights in {args.weight_set}", flush=True)
            continue
        path = _abs(rel)
        print(f"\n=== {skill} ===\n  weights={path}", flush=True)
        row = eval_skill(
            skill,
            path,
            num_seeds=args.num_seeds,
            batch_size=args.batch_size,
            max_steps=args.steps,
            master_seed=args.seed,
        )
        row["weight_set"] = args.weight_set
        results.append(row)
        print(
            f"  success={row['success_rate']:.3f}  "
            f"ever_saw={row['ever_saw_target_rate']:.3f}  "
            f"succ|ever={row['success_rate_given_ever_saw_target']:.3f}  "
            f"death={row['death_rate']:.3f}",
            flush=True,
        )

    out = args.output_json
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = f"runs/benchmark_reports/skill_natural_eval_full_{args.weight_set}_{stamp}.json"
    out_path = _abs(out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "protocol": "natural_full_craftax",
        "env": CRAFTAX_ENV_NAME,
        "weight_set": args.weight_set,
        "num_seeds": args.num_seeds,
        "steps": args.steps,
        "master_seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
