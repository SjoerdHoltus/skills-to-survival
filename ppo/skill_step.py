"""Map meta modes to Craftax Classic env actions for baseline meta PPO."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Action

from configs import (
    SURVIVAL_META_MODE_CRAFT_PLACE_FIRST,
    SURVIVAL_META_MODE_CRAFT_PLACE_LAST,
    SURVIVAL_META_MODE_PRIMITIVE_FIRST,
    SURVIVAL_META_MODE_SLEEP,
    SURVIVAL_META_PRIMITIVE_ACTIONS,
)
from training.meta_skill_executor import skill_greedy_action

_ACTION_NOOP = jnp.int32(Action.NOOP.value)
_ACTION_SLEEP = jnp.int32(Action.SLEEP.value)
_ACTION_PLACE_TABLE = jnp.int32(Action.PLACE_TABLE.value)
_ACTION_PLACE_FURNACE = jnp.int32(Action.PLACE_FURNACE.value)
_ACTION_PLACE_STONE = jnp.int32(Action.PLACE_STONE.value)
_ACTION_PLACE_PLANT = jnp.int32(Action.PLACE_PLANT.value)
_ACTION_MAKE_WOOD_PICKAXE = jnp.int32(Action.MAKE_WOOD_PICKAXE.value)
_ACTION_MAKE_STONE_PICKAXE = jnp.int32(Action.MAKE_STONE_PICKAXE.value)
_ACTION_MAKE_IRON_PICKAXE = jnp.int32(Action.MAKE_IRON_PICKAXE.value)
_ACTION_MAKE_WOOD_SWORD = jnp.int32(Action.MAKE_WOOD_SWORD.value)
_ACTION_MAKE_STONE_SWORD = jnp.int32(Action.MAKE_STONE_SWORD.value)
_ACTION_MAKE_IRON_SWORD = jnp.int32(Action.MAKE_IRON_SWORD.value)
_PRIM = jnp.asarray(SURVIVAL_META_PRIMITIVE_ACTIONS, dtype=jnp.int32)


def unwrap_craftax_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def craftax_states_from_runner(env_state):
    """Extract batched Craftax states from LogWrapper / optimistic-reset states."""
    if hasattr(env_state, "env_state"):
        return env_state.env_state
    return env_state


def _craft_action_for_mode(mode: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(
        mode == jnp.int32(9),
        _ACTION_PLACE_TABLE,
        jnp.where(
            mode == jnp.int32(10),
            _ACTION_PLACE_FURNACE,
            jnp.where(
                mode == jnp.int32(11),
                _ACTION_PLACE_STONE,
                jnp.where(
                    mode == jnp.int32(12),
                    _ACTION_PLACE_PLANT,
                    jnp.where(
                        mode == jnp.int32(13),
                        _ACTION_MAKE_WOOD_PICKAXE,
                        jnp.where(
                            mode == jnp.int32(14),
                            _ACTION_MAKE_STONE_PICKAXE,
                            jnp.where(
                                mode == jnp.int32(15),
                                _ACTION_MAKE_IRON_PICKAXE,
                                jnp.where(
                                    mode == jnp.int32(16),
                                    _ACTION_MAKE_WOOD_SWORD,
                                    jnp.where(
                                        mode == jnp.int32(17),
                                        _ACTION_MAKE_STONE_SWORD,
                                        jnp.where(
                                            mode == jnp.int32(18),
                                            _ACTION_MAKE_IRON_SWORD,
                                            _ACTION_NOOP,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def meta_mode_to_env_action(
    mode: jnp.ndarray,
    craftax_env,
    state,
    *,
    wp,
    dp,
    fp,
    mp,
    sp,
    op,
    h_wood: int,
    h_drink: int,
    h_ff: int,
    h_mf: int,
    h_stone: int,
    h_ore: int,
) -> jnp.ndarray:
    """Single-env meta mode -> native Craftax action."""
    mode = mode.astype(jnp.int32)
    skill_a = skill_greedy_action(
        mode,
        craftax_env,
        state,
        wp=wp,
        dp=dp,
        fp=fp,
        mp=mp,
        sp=sp,
        op=op,
        h_wood=h_wood,
        h_drink=h_drink,
        h_ff=h_ff,
        h_mf=h_mf,
        h_stone=h_stone,
        h_ore=h_ore,
    )
    craft_a = _craft_action_for_mode(mode)
    prim_a = _PRIM[
        jnp.clip(mode - jnp.int32(SURVIVAL_META_MODE_PRIMITIVE_FIRST), 0, 3).astype(jnp.int32)
    ]
    return jnp.where(
        mode == jnp.int32(0),
        _ACTION_NOOP,
        jnp.where(
            jnp.logical_and(mode >= jnp.int32(1), mode <= jnp.int32(8)),
            skill_a,
            jnp.where(
                jnp.logical_and(
                    mode >= jnp.int32(SURVIVAL_META_MODE_CRAFT_PLACE_FIRST),
                    mode <= jnp.int32(SURVIVAL_META_MODE_CRAFT_PLACE_LAST),
                ),
                craft_a,
                jnp.where(
                    mode >= jnp.int32(SURVIVAL_META_MODE_SLEEP),
                    _ACTION_SLEEP,
                    prim_a,
                ),
            ),
        ),
    )


def batched_meta_to_env_action(
    modes,
    craftax_env,
    states,
    *,
    skill_bundle: dict,
) -> jnp.ndarray:
    fn = lambda m, s: meta_mode_to_env_action(
        m,
        craftax_env,
        s,
        wp=skill_bundle["wp"],
        dp=skill_bundle["dp"],
        fp=skill_bundle["fp"],
        mp=skill_bundle["mp"],
        sp=skill_bundle["sp"],
        op=skill_bundle["op"],
        h_wood=skill_bundle["h_wood"],
        h_drink=skill_bundle["h_drink"],
        h_ff=skill_bundle["h_ff"],
        h_mf=skill_bundle["h_mf"],
        h_stone=skill_bundle["h_stone"],
        h_ore=skill_bundle["h_ore"],
    )
    return jax.vmap(fn)(modes, states)
