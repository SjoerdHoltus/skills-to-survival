"""Training bootstraps for full Craftax OpenES skills (gather + utility)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax.constants import OBS_DIM, BlockType, ItemType

import configs as _cfg
from training.craftax.bootstrap_target import (
    _sample_block_positions_in_fov,
    strip_natural_ores,
)
from training.craftax.state_helpers import clear_mobs_on_level, level_map, set_level_map

_TREE_V = jnp.int32(BlockType.TREE.value)
_WATER_V = jnp.int32(BlockType.WATER.value)
_STONE_V = jnp.int32(BlockType.STONE.value)
_GRASS_V = jnp.int32(BlockType.GRASS.value)
_LADDER_V = jnp.int32(ItemType.LADDER_DOWN.value)

_TARGET_BLOCKS = jnp.asarray(
    [BlockType.COAL.value, BlockType.IRON.value, BlockType.DIAMOND.value],
    dtype=jnp.int32,
)
_CARDINAL_STEPS = jnp.asarray([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)


def bootstrap_wood_state(state, boot_rng: jax.Array, _params) -> object:
    state = clear_mobs_on_level(state)
    boot_rng_count, boot_rng_trees = jax.random.split(boot_rng)
    player = state.player_position.astype(jnp.int32)
    mmap = level_map(state)
    tree_count = jax.random.randint(
        boot_rng_count,
        (),
        minval=jnp.int32(_cfg.FIND_WOOD_NUM_TREES_MIN),
        maxval=jnp.int32(_cfg.FIND_WOOD_NUM_TREES_MAX + 1),
    )
    mob_positions = jnp.zeros((1, 2), dtype=jnp.int32)
    mob_valids = jnp.zeros((1,), dtype=bool)
    new_map = _sample_block_positions_in_fov(
        mmap,
        player,
        mob_positions,
        mob_valids,
        boot_rng_trees,
        tree_count,
        _TREE_V,
        fov_r=int(_cfg.CHEBYSHEV_R),
    )
    state = set_level_map(state, new_map)
    inventory = state.inventory.replace(wood=jnp.int32(0))
    return state.replace(inventory=inventory)


def _strip_water(mmap: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mmap == _WATER_V, _GRASS_V, mmap)


def _place_water_patch(mmap: jnp.ndarray, player_position: jnp.ndarray, patch_rng: jax.Array):
    side = int(_cfg.DRINK_WATER_WATER_PATCH_SIDE)
    half_w = side // 2
    max_r_scan = max(int(_cfg.DRINK_WATER_SPAWN_MAX_CHEBYSHEV_R), half_w)
    rr = jnp.arange(-max_r_scan, max_r_scan + 1, dtype=jnp.int32)
    cc = jnp.arange(-max_r_scan, max_r_scan + 1, dtype=jnp.int32)
    grid_r, grid_c = jnp.meshgrid(rr, cc, indexing="ij")
    dr = grid_r.reshape(-1)
    dc = grid_c.reshape(-1)
    cheb = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
    lo = jnp.int32(_cfg.DRINK_WATER_SPAWN_MIN_CHEBYSHEV_R)
    hi = jnp.int32(_cfg.DRINK_WATER_SPAWN_MAX_CHEBYSHEV_R)
    in_symbolic = (jnp.abs(dr) <= jnp.int32(OBS_DIM[0] // 2)) & (
        jnp.abs(dc) <= jnp.int32(OBS_DIM[1] // 2)
    )
    valid = (cheb >= lo) & (cheb <= hi) & in_symbolic
    fallback = in_symbolic & (cheb >= jnp.int32(1))
    valid = jnp.where(jnp.any(valid), valid, fallback)
    noise = jax.random.uniform(patch_rng, shape=valid.shape)
    pick_i = jnp.argmax(noise + valid.astype(jnp.float32))
    pp = player_position.astype(jnp.int32)
    cr = jnp.clip(pp[0] + dr[pick_i], 1, mmap.shape[0] - 2)
    cl = jnp.clip(pp[1] + dc[pick_i], 1, mmap.shape[1] - 2)
    ij = jnp.arange(-half_w, half_w + 1, dtype=jnp.int32)
    ddi, ddj = jnp.meshgrid(ij, ij, indexing="ij")
    tr = jnp.clip(cr + ddi, 1, mmap.shape[0] - 2).ravel()
    tc = jnp.clip(cl + ddj, 1, mmap.shape[1] - 2).ravel()
    return mmap.at[tr, tc].set(_WATER_V)


def bootstrap_water_state(state, boot_rng: jax.Array, _params) -> object:
    state = clear_mobs_on_level(state)
    mmap = _strip_water(level_map(state))
    new_map = _place_water_patch(mmap, state.player_position.astype(jnp.int32), boot_rng)
    state = set_level_map(state, new_map)
    return state.replace(
        player_drink=jnp.int32(_cfg.DRINK_WATER_PLAYER_START_DRINK),
        player_thirst=jnp.asarray(float(_cfg.DRINK_WATER_PLAYER_START_THIRST), dtype=jnp.float32),
        is_sleeping=jnp.asarray(False, dtype=state.is_sleeping.dtype),
    )


def bootstrap_stone_state(state, boot_rng: jax.Array, _params) -> object:
    state = clear_mobs_on_level(state)
    boot_rng_count, boot_rng_blocks = jax.random.split(boot_rng)
    player = state.player_position.astype(jnp.int32)
    block_count = jax.random.randint(
        boot_rng_count,
        (),
        minval=jnp.int32(_cfg.COLLECT_STONE_NUM_BLOCKS_MIN),
        maxval=jnp.int32(_cfg.COLLECT_STONE_NUM_BLOCKS_MAX + 1),
    )
    cleared = strip_natural_ores(level_map(state))
    mob_positions = jnp.zeros((1, 2), dtype=jnp.int32)
    mob_valids = jnp.zeros((1,), dtype=bool)
    new_map = _sample_block_positions_in_fov(
        cleared,
        player,
        mob_positions,
        mob_valids,
        boot_rng_blocks,
        block_count,
        _STONE_V,
        fov_r=int(_cfg.COLLECT_STONE_FOV_CHEBYSHEV_R),
    )
    state = set_level_map(state, new_map)
    inventory = state.inventory.replace(pickaxe=jnp.int32(1), stone=jnp.int32(0))
    return state.replace(inventory=inventory)


def _delta_to_direction(delta: jnp.ndarray) -> jnp.ndarray:
    dr, dc = delta[0], delta[1]
    return jnp.where(
        dr < 0,
        jnp.int32(3),
        jnp.where(dr > 0, jnp.int32(4), jnp.where(dc < 0, jnp.int32(1), jnp.int32(2))),
    )


def _place_layered_ore(mmap, player, rng, ore_value, shell_count):
    rng_dir, rng_dist = jax.random.split(rng)
    dir_idx = jax.random.randint(rng_dir, (), 0, 2)
    delta = _CARDINAL_STEPS[dir_idx]
    layers = jnp.clip(shell_count.astype(jnp.int32), 1, 3)
    h, w = mmap.shape
    max_row = jnp.int32((OBS_DIM[0] - 1) // 2)
    max_col = jnp.int32((OBS_DIM[1] - 1) // 2)
    is_vertical = jnp.abs(delta[0]) > 0
    fov_cap = jnp.where(is_vertical, max_row, max_col)
    map_cap = jnp.where(
        delta[0] < 0,
        player[0],
        jnp.where(
            delta[0] > 0,
            h - 1 - player[0],
            jnp.where(delta[1] < 0, player[1], w - 1 - player[1]),
        ),
    )
    usable_cap = jnp.minimum(fov_cap, map_cap)
    dist_lo = jnp.minimum(jnp.int32(_cfg.COLLECT_ORE_ORE_DISTANCE_MIN), usable_cap)
    dist_hi = jnp.minimum(jnp.int32(_cfg.COLLECT_ORE_ORE_DISTANCE_MAX), usable_cap)
    dist = jax.random.randint(rng_dist, (), minval=dist_lo, maxval=dist_hi + 1)
    ore_r = player[0] + delta[0] * dist
    ore_c = player[1] + delta[1] * dist
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    cheb = jnp.maximum(jnp.abs(rr - ore_r), jnp.abs(cc - ore_c))
    shell = (cheb > 0) & (cheb <= layers)
    player_cell = (rr == player[0]) & (cc == player[1])
    shell = shell & jnp.logical_not(player_cell)
    new_map = jnp.where(shell, _STONE_V, mmap)
    new_map = new_map.at[ore_r, ore_c].set(ore_value.astype(new_map.dtype))
    return new_map, _delta_to_direction(delta)


def bootstrap_ore_state(
    state, boot_rng: jax.Array, _params, target_id: jnp.ndarray, wall_count=None
):
    state = clear_mobs_on_level(state)
    boot_rng_layout, boot_rng_wall = jax.random.split(boot_rng)
    if wall_count is None:
        wall_count = jnp.int32(2)
    player = state.player_position.astype(jnp.int32)
    ore_value = _TARGET_BLOCKS[target_id.astype(jnp.int32)]
    cleared = strip_natural_ores(level_map(state))
    cleared = jnp.where(cleared == _STONE_V, _GRASS_V, cleared)
    new_map, face_dir = _place_layered_ore(cleared, player, boot_rng_layout, ore_value, wall_count)
    state = set_level_map(state, new_map)
    tid = target_id.astype(jnp.int32)
    is_diamond = tid == jnp.int32(2)
    inventory = state.inventory.replace(
        pickaxe=jnp.where(is_diamond, jnp.int32(3), jnp.int32(2)),
        coal=jnp.int32(0),
        iron=jnp.int32(0),
        diamond=jnp.int32(0),
    )
    return state.replace(inventory=inventory, player_direction=face_dir)


def bootstrap_monster_fight_state(
    state,
    boot_rng: jax.Array,
    env_params,
    *,
    static_melee: bool = False,
    melee_type_id: int = 0,
):
    from training.craftax.bootstrap_mobs import apply_melee_bootstrap

    return apply_melee_bootstrap(
        state,
        boot_rng,
        env_params,
        static_positions=static_melee,
        equip_sword=True,
        melee_type_id=melee_type_id,
    )


def bootstrap_find_food_state(state, boot_rng: jax.Array, env_params, *, static_cow: bool = False):
    from training.craftax.bootstrap_mobs import apply_passive_bootstrap

    return apply_passive_bootstrap(
        state,
        boot_rng,
        env_params,
        static_positions=static_cow,
        equip_sword=True,
    )
