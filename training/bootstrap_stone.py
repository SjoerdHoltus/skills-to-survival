"""Collect-stone bootstrap and 11-D features (stone target, wood pickaxe)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BlockType

import configs as _cfg
from training.bootstrap_target import (
    INV_IDX_STONE,
    _sample_block_positions_in_fov,
    _strip_natural_ores,
    collect_target_features_from_symbolic_obs,
)
from training.monster_world_bootstrap import rebuild_mob_map_from_state

_STONE_V = jnp.int32(BlockType.STONE.value)
_CAP = jnp.float32(_cfg.COLLECT_STONE_MAX_STONE)


def collect_stone_features_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    return collect_target_features_from_symbolic_obs(
        symbolic_obs,
        _STONE_V,
        INV_IDX_STONE,
        _CAP,
        float(_cfg.COLLECT_STONE_FOV_CHEBYSHEV_R),
        include_health=False,
    )


def make_collect_stone_obs(env, state) -> jnp.ndarray:
    return collect_stone_features_from_symbolic_obs(env.get_obs(state))


def _clear_hostiles(state):
    zombies = state.zombies.replace(
        position=jnp.zeros_like(state.zombies.position),
        health=jnp.zeros_like(state.zombies.health),
        mask=jnp.zeros_like(state.zombies.mask),
        attack_cooldown=jnp.zeros_like(state.zombies.attack_cooldown),
    )
    skeletons = state.skeletons.replace(
        position=jnp.zeros_like(state.skeletons.position),
        health=jnp.zeros_like(state.skeletons.health),
        mask=jnp.zeros_like(state.skeletons.mask),
        attack_cooldown=jnp.zeros_like(state.skeletons.attack_cooldown),
    )
    cows = state.cows.replace(
        position=jnp.zeros_like(state.cows.position),
        health=jnp.zeros_like(state.cows.health),
        mask=jnp.zeros_like(state.cows.mask),
        attack_cooldown=jnp.zeros_like(state.cows.attack_cooldown),
    )
    arrows = state.arrows.replace(
        position=jnp.zeros_like(state.arrows.position),
        health=jnp.zeros_like(state.arrows.health),
        mask=jnp.zeros_like(state.arrows.mask),
    )
    return state.replace(zombies=zombies, skeletons=skeletons, cows=cows, arrows=arrows)


def bootstrap_stone_state(state, boot_rng: jax.Array, _params) -> object:
    """Clear hostiles, strip ores, equip wood pickaxe, spawn stone in FOV."""
    boot_rng_count, boot_rng_blocks = jax.random.split(boot_rng)
    staged = _clear_hostiles(state)
    player = staged.player_position.astype(jnp.int32)

    block_count = jax.random.randint(
        boot_rng_count,
        (),
        minval=jnp.int32(_cfg.COLLECT_STONE_NUM_BLOCKS_MIN),
        maxval=jnp.int32(_cfg.COLLECT_STONE_NUM_BLOCKS_MAX + 1),
    )
    cleared_map = _strip_natural_ores(staged.map)
    mob_positions = jnp.zeros((1, 2), dtype=jnp.int32)
    mob_valids = jnp.zeros((1,), dtype=bool)
    new_map = _sample_block_positions_in_fov(
        cleared_map,
        player,
        mob_positions,
        mob_valids,
        boot_rng_blocks,
        block_count,
        _STONE_V,
        fov_r=int(_cfg.COLLECT_STONE_FOV_CHEBYSHEV_R),
    )
    staged_map = staged.replace(map=new_map)
    mob_map = rebuild_mob_map_from_state(staged_map)
    inventory = staged_map.inventory.replace(
        wood_pickaxe=jnp.int32(1),
        stone_pickaxe=jnp.int32(0),
        stone=jnp.int32(0),
    )
    return staged_map.replace(mob_map=mob_map, inventory=inventory)
