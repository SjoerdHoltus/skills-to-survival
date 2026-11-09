"""Valid meta-mode mask for survival hierarchical PPO (craft/place/material gating)."""

from __future__ import annotations

import jax.numpy as jnp
from craftax.craftax_classic.constants import DIRECTIONS, BlockType
from craftax.craftax_classic.game_logic import in_bounds, is_in_wall, is_near_block

from configs import SURVIVAL_META_NUM_MODES
from training.survival_meta_obs import survival_craft_progress_features_from_state

# Indices within the 24-D craft progress slice appended to meta obs.
_CRAFT_CAN_PLACE_TABLE = 3
_CRAFT_CAN_PLACE_FURNACE = 16
_CRAFT_CAN_CRAFT_WOOD_PICK = 17
_CRAFT_CAN_CRAFT_STONE_PICK = 19
_CRAFT_CAN_CRAFT_IRON_PICK = 20
_CRAFT_CAN_CRAFT_IRON_SWORD = 21
_CRAFT_NEAR_TABLE = 2


def survival_meta_action_mask(state, *, num_modes: int = SURVIVAL_META_NUM_MODES) -> jnp.ndarray:
    """Boolean mask ``(num_modes,)``: True when the meta mode is valid in ``state``."""
    craft = survival_craft_progress_features_from_state(state)
    placing_pos = state.player_position + DIRECTIONS[state.player_direction]
    can_place_here = jnp.logical_not(is_in_wall(state, placing_pos))
    near_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)

    has_wood = state.inventory.wood >= jnp.int32(1)
    has_stone = state.inventory.stone >= jnp.int32(1)
    has_sapling = state.inventory.sapling >= jnp.int32(1)
    has_wood_pick = state.inventory.wood_pickaxe >= jnp.int32(1)
    has_stone_pick = state.inventory.stone_pickaxe >= jnp.int32(1)
    has_iron_pick = state.inventory.iron_pickaxe >= jnp.int32(1)
    has_wood_sword = state.inventory.wood_sword >= jnp.int32(1)

    face_is_grass = jnp.logical_and(
        in_bounds(state, placing_pos),
        state.map[placing_pos[0], placing_pos[1]] == jnp.int32(BlockType.GRASS.value),
    )

    can_place_table = jnp.logical_and(state.inventory.wood >= jnp.int32(2), can_place_here)
    can_place_furnace = jnp.logical_and(state.inventory.stone >= jnp.int32(4), can_place_here)
    can_place_stone = jnp.logical_and(has_stone, can_place_here)
    can_place_plant = jnp.logical_and(has_sapling, face_is_grass)

    can_craft_wood_pick = craft[_CRAFT_CAN_CRAFT_WOOD_PICK] >= jnp.float32(0.5)
    can_craft_stone_pick = craft[_CRAFT_CAN_CRAFT_STONE_PICK] >= jnp.float32(0.5)
    can_craft_iron_pick = craft[_CRAFT_CAN_CRAFT_IRON_PICK] >= jnp.float32(0.5)
    can_craft_iron_sword = craft[_CRAFT_CAN_CRAFT_IRON_SWORD] >= jnp.float32(0.5)
    can_craft_wood_sword = jnp.logical_and(has_wood, near_table)
    can_craft_stone_sword = jnp.logical_and(jnp.logical_and(has_wood, has_stone), near_table)

    # Modes 0..23 — always allow noop, skills, movement, sleep; gate place/craft.
    valid = jnp.ones((num_modes,), dtype=jnp.bool_)

    valid = valid.at[9].set(can_place_table)
    valid = valid.at[10].set(can_place_furnace)
    valid = valid.at[11].set(can_place_stone)
    valid = valid.at[12].set(can_place_plant)
    valid = valid.at[13].set(can_craft_wood_pick)
    valid = valid.at[14].set(can_craft_stone_pick)
    valid = valid.at[15].set(can_craft_iron_pick)
    valid = valid.at[16].set(can_craft_wood_sword)
    valid = valid.at[17].set(can_craft_stone_sword)
    valid = valid.at[18].set(can_craft_iron_sword)

    # Mining modes require prior pickaxes (same penalties as training rollout).
    valid = valid.at[5].set(has_wood_pick)
    valid = valid.at[6].set(has_stone_pick)
    valid = valid.at[7].set(has_stone_pick)
    valid = valid.at[8].set(has_iron_pick)
    valid = valid.at[4].set(has_wood_sword)

    return valid


def survival_meta_greedy_mode(
    logits: jnp.ndarray, state, *, num_modes: int = SURVIVAL_META_NUM_MODES
) -> jnp.ndarray:
    """Argmax over meta logits with the same invalid-mode masking used in PPO training."""
    valid_mask = survival_meta_action_mask(state, num_modes=num_modes)
    masked = jnp.where(valid_mask, logits, jnp.float32(-1e9))
    return jnp.argmax(masked).astype(jnp.int32)
