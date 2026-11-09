"""Helpers for mutating full Craftax ``EnvState`` during skill bootstraps."""

from __future__ import annotations

import jax.numpy as jnp


def player_level(state) -> jnp.ndarray:
    return state.player_level.astype(jnp.int32)


def level_map(state) -> jnp.ndarray:
    lvl = player_level(state)
    return state.map[lvl]


def set_level_map(state, new_map: jnp.ndarray):
    lvl = player_level(state)
    return state.replace(map=state.map.at[lvl].set(new_map))


def clear_mobs_on_level(state):
    """Zero mob masks on the player's current level."""
    lvl = player_level(state)

    def _clear_mobs(mobs):
        zmask = jnp.zeros_like(mobs.mask[lvl])
        return mobs.replace(
            mask=mobs.mask.at[lvl].set(zmask),
            health=mobs.health.at[lvl].set(jnp.zeros_like(mobs.health[lvl])),
        )

    state = state.replace(
        melee_mobs=_clear_mobs(state.melee_mobs),
        passive_mobs=_clear_mobs(state.passive_mobs),
        ranged_mobs=_clear_mobs(state.ranged_mobs),
        mob_projectiles=_clear_mobs(state.mob_projectiles),
    )
    return state
