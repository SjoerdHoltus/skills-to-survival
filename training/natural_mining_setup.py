"""Post-reset setup for skill policies on natural Craftax Classic maps.

Keeps the procedurally generated world; only applies light inventory / meter staging
so a frozen skill can be tested without the training bootstrap spawners.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from configs import COLLECT_ORE_FEAT_HAS_TARGET
from training.bootstrap_ore import make_collect_ore_obs
from training.bootstrap_stone import _clear_hostiles, make_collect_stone_obs


def setup_natural_find_wood_state(state):
    """Natural map; zero wood inventory so gains are attributable to this rollout."""
    inventory = state.inventory.replace(wood=jnp.int32(0))
    return state.replace(inventory=inventory)


def setup_natural_drink_water_state(state):
    """Natural map; empty drink meter so drinking is possible."""
    return state.replace(
        player_drink=jnp.int32(0),
        player_thirst=jnp.float32(0.0),
    )


def setup_natural_find_food_state(state):
    """Natural map + wood sword; leave natural cows; empty food meter."""
    inventory = state.inventory.replace(wood_sword=jnp.int32(1))
    return state.replace(
        inventory=inventory,
        player_food=jnp.int32(0),
        player_hunger=jnp.float32(0.0),
    )


def setup_natural_monster_fight_state(state):
    """Natural map + wood sword; leave natural hostiles."""
    inventory = state.inventory.replace(wood_sword=jnp.int32(1))
    return state.replace(inventory=inventory)


def setup_natural_collect_stone_state(state):
    """Natural map + wood pickaxe; no artificial stone spawn."""
    staged = _clear_hostiles(state)
    inventory = staged.inventory.replace(
        wood_pickaxe=jnp.int32(1),
        stone_pickaxe=jnp.int32(0),
        stone=jnp.int32(0),
    )
    return staged.replace(inventory=inventory)


def setup_natural_collect_ore_state(state, target_id: jnp.ndarray | int | None = None):
    """Natural map + pickaxes; ore counts start at zero. Diamond needs iron pickaxe."""
    staged = _clear_hostiles(state)
    is_diamond = jnp.int32(target_id) == jnp.int32(2) if target_id is not None else jnp.array(False)
    inventory = staged.inventory.replace(
        wood_pickaxe=jnp.int32(1),
        stone_pickaxe=jnp.int32(1),
        iron_pickaxe=jnp.where(is_diamond, jnp.int32(1), jnp.int32(0)),
        coal=jnp.int32(0),
        iron=jnp.int32(0),
        diamond=jnp.int32(0),
    )
    return staged.replace(inventory=inventory)


def target_visible_in_obs(obs: jnp.ndarray, has_target_index: int = 3) -> jnp.ndarray:
    """True when the policy obs reports a target block in symbolic FOV."""
    return (obs[..., has_target_index] >= jnp.float32(0.5)).astype(jnp.float32)


def natural_stone_target_in_fov(sym_env, sym_params, seed: int) -> bool:
    rng = jax.random.PRNGKey(seed)
    _, state = sym_env.reset(rng, sym_params)
    state = setup_natural_collect_stone_state(state)
    obs = make_collect_stone_obs(sym_env, state)
    return bool(float(target_visible_in_obs(obs)) >= 0.5)


def natural_ore_target_in_fov(sym_env, sym_params, seed: int, target_id: int) -> bool:
    rng = jax.random.PRNGKey(seed)
    _, state = sym_env.reset(rng, sym_params)
    state = setup_natural_collect_ore_state(state, target_id=target_id)
    obs = make_collect_ore_obs(sym_env, state, jnp.int32(target_id))
    return bool(float(target_visible_in_obs(obs, COLLECT_ORE_FEAT_HAS_TARGET)) >= 0.5)


def resolve_natural_seed(
    probe,
    start_seed: int,
    max_scan: int,
    require_target_in_fov: bool,
) -> int:
    """Return start_seed, or the first seed in [start, start+max_scan) passing probe."""
    if not require_target_in_fov:
        return start_seed
    for offset in range(max_scan):
        seed = start_seed + offset
        if probe(seed):
            if offset:
                print(
                    f"Natural seed scan: using seed={seed} (+{offset} from start seed {start_seed})"
                )
            return seed
    raise SystemExit(
        f"No natural seed in [{start_seed}, {start_seed + max_scan}) "
        f"with target visible in FOV. Increase --seed-scan-max or change --seed."
    )
