"""Symbolic observation helpers for hierarchical survival meta-policy.

The meta actor/critic use the full Craftax Classic symbolic vector via
``survival_meta_policy_obs``. Handcrafted skill slices from this module are still
used for frozen skill dispatch, dashboards, and diagnostics.
"""

from __future__ import annotations

import jax.numpy as jnp
from craftax.craftax_classic.constants import (
    OBS_DIM,
    Achievement,
    BlockType,
)
from craftax.craftax_classic.game_logic import is_near_block

from configs import (
    COLLECT_ORE_MAX_COAL,
    COLLECT_ORE_MAX_DIAMOND,
    COLLECT_ORE_MAX_IRON,
    COLLECT_STONE_MAX_STONE,
    OBS_SIZE,
    SURVIVAL_META_CRAFT_FEATURES,
    SURVIVAL_META_FIND_FOOD_SLICE_FEATURES,
    SURVIVAL_META_GATHER_SLICE_FEATURES,
    SURVIVAL_META_HANDCRAFTED_OBS_SIZE,
    SURVIVAL_META_MONSTER_SLICE_FEATURES,
    SURVIVAL_META_OBS_SIZE,
    SURVIVAL_META_ORE_SLICE_FEATURES,
)
from training.bootstrap_ore import (
    DIAMOND_TARGET_ID,
    collect_ore_features_from_symbolic_obs,
)
from training.bootstrap_stone import collect_stone_features_from_symbolic_obs
from training.bootstrap_target import (
    cardinal_blocked_from_symbolic_obs,
    gather_obs_meta_slice,
    ore_obs_meta_slice,
)
from training.bootstrap_water import drink_water_features_from_symbolic_obs
from training.bootstrap_wood import find_wood_features_from_symbolic_obs
from training.monster_world_bootstrap import (
    find_food_features_from_symbolic_obs,
    monster_fight_features_from_symbolic_obs,
    monster_obs_meta_slice,
)

_NUM_BLOCK_TYPES = int(len(BlockType))
_NUM_MOB_CHANNELS = 4
_SYMBOLIC_CHANNELS_PER_CELL = _NUM_BLOCK_TYPES + _NUM_MOB_CHANNELS
_SYMBOLIC_MAP_FLAT = int(OBS_DIM[0] * OBS_DIM[1] * _SYMBOLIC_CHANNELS_PER_CELL)
assert OBS_SIZE - _SYMBOLIC_MAP_FLAT == 22, (_SYMBOLIC_MAP_FLAT, OBS_SIZE)
_SYMBOLIC_FOOD_IDX = _SYMBOLIC_MAP_FLAT + 13
_SYMBOLIC_DRINK_IDX = _SYMBOLIC_MAP_FLAT + 14
_SYMBOLIC_ENERGY_IDX = _SYMBOLIC_MAP_FLAT + 15
_CLASSIC_NEED_MAX = 9.0
_CLASSIC_WOOD_MAX = 9.0


def survival_need_features_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """Food/drink/energy levels and deficits, normalized to roughly [0, 1]."""
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    food = jnp.clip(sym[_SYMBOLIC_FOOD_IDX] * 10.0 / _CLASSIC_NEED_MAX, 0.0, 1.0)
    drink = jnp.clip(sym[_SYMBOLIC_DRINK_IDX] * 10.0 / _CLASSIC_NEED_MAX, 0.0, 1.0)
    energy = jnp.clip(sym[_SYMBOLIC_ENERGY_IDX] * 10.0 / _CLASSIC_NEED_MAX, 0.0, 1.0)
    return jnp.stack(
        [
            food,
            drink,
            energy,
            1.0 - food,
            1.0 - drink,
            1.0 - energy,
        ],
        axis=0,
    )


def survival_craft_progress_features_from_state(state) -> jnp.ndarray:
    """Craft chain + pickaxes + mining resource norms for meta mode selection."""
    wood_norm = jnp.clip(
        state.inventory.wood.astype(jnp.float32) / jnp.float32(_CLASSIC_WOOD_MAX),
        0.0,
        1.0,
    )
    ach = state.achievements
    table_done = ach[Achievement.PLACE_TABLE.value].astype(jnp.float32)
    furnace_done = ach[Achievement.PLACE_FURNACE.value].astype(jnp.float32)
    diamond_done = ach[Achievement.COLLECT_DIAMOND.value].astype(jnp.float32)
    can_place = (state.inventory.wood >= 2).astype(jnp.float32)
    near_table = is_near_block(state, BlockType.CRAFTING_TABLE.value).astype(jnp.float32)
    near_furnace = is_near_block(state, BlockType.FURNACE.value).astype(jnp.float32)
    has_wood_pick = (state.inventory.wood_pickaxe >= 1).astype(jnp.float32)
    has_stone_pick = (state.inventory.stone_pickaxe >= 1).astype(jnp.float32)
    has_iron_pick = (state.inventory.iron_pickaxe >= 1).astype(jnp.float32)
    has_wood_sword = (state.inventory.wood_sword >= 1).astype(jnp.float32)
    has_stone_sword = (state.inventory.stone_sword >= 1).astype(jnp.float32)
    has_iron_sword = (state.inventory.iron_sword >= 1).astype(jnp.float32)
    stone_norm = jnp.clip(
        state.inventory.stone.astype(jnp.float32) / jnp.float32(COLLECT_STONE_MAX_STONE),
        0.0,
        1.0,
    )
    coal_norm = jnp.clip(
        state.inventory.coal.astype(jnp.float32) / jnp.float32(COLLECT_ORE_MAX_COAL),
        0.0,
        1.0,
    )
    iron_norm = jnp.clip(
        state.inventory.iron.astype(jnp.float32) / jnp.float32(COLLECT_ORE_MAX_IRON),
        0.0,
        1.0,
    )
    diamond_norm = jnp.clip(
        state.inventory.diamond.astype(jnp.float32) / jnp.float32(COLLECT_ORE_MAX_DIAMOND),
        0.0,
        1.0,
    )
    can_place_furnace = (state.inventory.stone >= 4).astype(jnp.float32)
    near_table_and_furnace = ((near_table > 0.5) & (near_furnace > 0.5)).astype(jnp.float32)
    has_iron_tool_materials = (
        (state.inventory.wood >= 1)
        & (state.inventory.stone >= 1)
        & (state.inventory.coal >= 1)
        & (state.inventory.iron >= 1)
    ).astype(jnp.float32)
    can_craft_wood_pick = ((state.inventory.wood >= 1) & (near_table > 0.5)).astype(jnp.float32)
    can_craft_stone_pick = (
        (state.inventory.stone >= 1) & (state.inventory.wood >= 1) & (near_table > 0.5)
    ).astype(jnp.float32)
    can_craft_iron_pick = ((has_iron_tool_materials > 0.5) & (near_table_and_furnace > 0.5)).astype(
        jnp.float32
    )
    can_craft_iron_sword = (
        (has_iron_tool_materials > 0.5) & (near_table_and_furnace > 0.5)
    ).astype(jnp.float32)
    return jnp.stack(
        [
            wood_norm,
            table_done,
            near_table,
            can_place,
            has_wood_pick,
            has_stone_pick,
            has_iron_pick,
            has_wood_sword,
            has_stone_sword,
            has_iron_sword,
            stone_norm,
            coal_norm,
            iron_norm,
            diamond_norm,
            near_furnace,
            furnace_done,
            can_place_furnace,
            can_craft_wood_pick,
            diamond_done,
            can_craft_stone_pick,
            can_craft_iron_pick,
            can_craft_iron_sword,
            near_table_and_furnace,
            has_iron_tool_materials,
        ],
        axis=0,
    )


assert SURVIVAL_META_CRAFT_FEATURES == 24


def survival_meta_policy_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """Full Craftax Classic symbolic vector for meta actor/critic."""
    return symbolic_obs.astype(jnp.float32).reshape(-1)


def survival_meta_features_from_symbolic_obs(symbolic_obs: jnp.ndarray, state=None) -> jnp.ndarray:
    """Stack deduplicated skill slices + needs + craft progress for the meta actor/critic."""
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    blocked = cardinal_blocked_from_symbolic_obs(sym)
    w = gather_obs_meta_slice(find_wood_features_from_symbolic_obs(sym))
    d = gather_obs_meta_slice(drink_water_features_from_symbolic_obs(sym))
    f = gather_obs_meta_slice(find_food_features_from_symbolic_obs(sym))
    m = monster_obs_meta_slice(monster_fight_features_from_symbolic_obs(sym))
    stone = gather_obs_meta_slice(collect_stone_features_from_symbolic_obs(sym))
    coal = ore_obs_meta_slice(collect_ore_features_from_symbolic_obs(sym, jnp.int32(0)))
    iron = ore_obs_meta_slice(collect_ore_features_from_symbolic_obs(sym, jnp.int32(1)))
    diamond = ore_obs_meta_slice(collect_ore_features_from_symbolic_obs(sym, DIAMOND_TARGET_ID))
    n = survival_need_features_from_symbolic_obs(sym)
    if state is None:
        c = jnp.zeros(SURVIVAL_META_CRAFT_FEATURES, dtype=jnp.float32)
    else:
        c = survival_craft_progress_features_from_state(state)
    return jnp.concatenate([blocked, w, d, f, m, stone, coal, iron, diamond, n, c], axis=0)


_assert_policy = survival_meta_policy_obs(jnp.zeros(OBS_SIZE, dtype=jnp.float32))
assert int(_assert_policy.shape[0]) == SURVIVAL_META_OBS_SIZE, (
    f"SURVIVAL_META_OBS_SIZE={SURVIVAL_META_OBS_SIZE} vs policy obs {_assert_policy.shape[0]}"
)
_assert_vec = survival_meta_features_from_symbolic_obs(jnp.zeros(OBS_SIZE, dtype=jnp.float32))
assert int(_assert_vec.shape[0]) == SURVIVAL_META_HANDCRAFTED_OBS_SIZE, (
    f"SURVIVAL_META_HANDCRAFTED_OBS_SIZE={SURVIVAL_META_HANDCRAFTED_OBS_SIZE} "
    f"vs built length {_assert_vec.shape[0]}"
)
assert int(gather_obs_meta_slice(jnp.zeros(11)).shape[0]) == SURVIVAL_META_GATHER_SLICE_FEATURES
assert int(gather_obs_meta_slice(jnp.zeros(10)).shape[0]) == SURVIVAL_META_FIND_FOOD_SLICE_FEATURES
assert int(monster_obs_meta_slice(jnp.zeros(18)).shape[0]) == SURVIVAL_META_MONSTER_SLICE_FEATURES
assert int(ore_obs_meta_slice(jnp.zeros(15)).shape[0]) == SURVIVAL_META_ORE_SLICE_FEATURES
