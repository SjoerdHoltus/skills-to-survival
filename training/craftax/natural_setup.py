"""Post-reset setup for skill policies on natural full Craftax maps.

Keeps the procedurally generated world; only light inventory / meter staging
(no bootstrap spawners). Inventory fields follow Extended (``sword`` / ``pickaxe``
levels), not Classic ``wood_sword`` / ``wood_pickaxe``.
"""

from __future__ import annotations

import jax.numpy as jnp

from training.craftax.state_helpers import clear_mobs_on_level


def setup_natural_find_wood_state(state):
    inventory = state.inventory.replace(wood=jnp.int32(0))
    return state.replace(inventory=inventory)


def setup_natural_drink_water_state(state):
    return state.replace(
        player_drink=jnp.int32(0),
        player_thirst=jnp.float32(0.0),
    )


def setup_natural_find_food_state(state):
    inventory = state.inventory.replace(sword=jnp.int32(1))
    return state.replace(
        inventory=inventory,
        player_food=jnp.int32(0),
        player_hunger=jnp.float32(0.0),
    )


def setup_natural_monster_fight_state(state):
    inventory = state.inventory.replace(sword=jnp.int32(1))
    return state.replace(inventory=inventory)


def setup_natural_collect_stone_state(state):
    staged = clear_mobs_on_level(state)
    inventory = staged.inventory.replace(pickaxe=jnp.int32(1), stone=jnp.int32(0))
    return staged.replace(inventory=inventory)


def setup_natural_collect_ore_state(state, target_id: jnp.ndarray | int | None = None):
    staged = clear_mobs_on_level(state)
    is_diamond = jnp.int32(target_id) == jnp.int32(2) if target_id is not None else jnp.array(False)
    inventory = staged.inventory.replace(
        pickaxe=jnp.where(is_diamond, jnp.int32(3), jnp.int32(2)),
        coal=jnp.int32(0),
        iron=jnp.int32(0),
        diamond=jnp.int32(0),
    )
    return staged.replace(inventory=inventory)
