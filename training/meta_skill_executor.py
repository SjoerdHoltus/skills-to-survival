"""Greedy one-step actions from frozen OpenES skills for the survival meta-controller."""

from __future__ import annotations

import jax.numpy as jnp

from configs import (
    FIND_WOOD_NUM_ACTIONS,
    SURVIVAL_META_MODE_COAL,
    SURVIVAL_META_MODE_DIAMOND,
    SURVIVAL_META_MODE_DRINK,
    SURVIVAL_META_MODE_FIGHT,
    SURVIVAL_META_MODE_FOOD,
    SURVIVAL_META_MODE_IRON,
    SURVIVAL_META_MODE_STONE,
    SURVIVAL_META_MODE_WOOD,
)
from policies.collect_ore import CollectOrePolicy
from policies.collect_stone import CollectStonePolicy
from policies.drink_water import DrinkWaterPolicy
from policies.find_food import FindFoodPolicy
from policies.find_wood import FindWoodPolicy
from policies.monster_fight import MonsterFightPolicy
from training.bootstrap_ore import DIAMOND_TARGET_ID, make_collect_ore_obs
from training.bootstrap_stone import make_collect_stone_obs
from training.bootstrap_water import make_drink_water_obs
from training.bootstrap_wood import make_find_wood_obs
from training.monster_world_bootstrap import make_find_food_obs, make_monster_fight_obs


def skill_greedy_action(
    mode: jnp.ndarray,
    env,
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
    """Greedy low-level action for frozen skill meta modes 1..8."""
    wood_obs = make_find_wood_obs(env, state)
    drink_obs = make_drink_water_obs(env, state)
    food_obs = make_find_food_obs(env, state)
    mf_obs = make_monster_fight_obs(env, state)
    stone_obs = make_collect_stone_obs(env, state)
    coal_obs = make_collect_ore_obs(env, state, jnp.int32(0))
    iron_obs = make_collect_ore_obs(env, state, jnp.int32(1))
    diamond_obs = make_collect_ore_obs(env, state, DIAMOND_TARGET_ID)

    fw_logits = FindWoodPolicy(hidden_size=h_wood, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": wp}, wood_obs
    )
    dw_logits = DrinkWaterPolicy(hidden_size=h_drink, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": dp}, drink_obs
    )
    ff_logits = FindFoodPolicy(hidden_size=h_ff, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": fp}, food_obs
    )
    mf_logits = MonsterFightPolicy(hidden_size=h_mf, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": mp}, mf_obs
    )
    stone_logits = CollectStonePolicy(hidden_size=h_stone, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": sp}, stone_obs
    )
    coal_logits = CollectOrePolicy(hidden_size=h_ore, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": op}, coal_obs
    )
    iron_logits = CollectOrePolicy(hidden_size=h_ore, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": op}, iron_obs
    )
    diamond_logits = CollectOrePolicy(hidden_size=h_ore, num_actions=FIND_WOOD_NUM_ACTIONS).apply(
        {"params": op}, diamond_obs
    )

    fw_a = jnp.argmax(fw_logits).astype(jnp.int32) + jnp.int32(1)
    dw_a = jnp.argmax(dw_logits).astype(jnp.int32) + jnp.int32(1)
    ff_a = jnp.argmax(ff_logits).astype(jnp.int32) + jnp.int32(1)
    mf_a = jnp.argmax(mf_logits).astype(jnp.int32) + jnp.int32(1)
    stone_a = jnp.argmax(stone_logits).astype(jnp.int32) + jnp.int32(1)
    coal_a = jnp.argmax(coal_logits).astype(jnp.int32) + jnp.int32(1)
    iron_a = jnp.argmax(iron_logits).astype(jnp.int32) + jnp.int32(1)
    diamond_a = jnp.argmax(diamond_logits).astype(jnp.int32) + jnp.int32(1)

    m = mode.astype(jnp.int32)
    return jnp.where(
        m == jnp.int32(SURVIVAL_META_MODE_WOOD),
        fw_a,
        jnp.where(
            m == jnp.int32(SURVIVAL_META_MODE_DRINK),
            dw_a,
            jnp.where(
                m == jnp.int32(SURVIVAL_META_MODE_FOOD),
                ff_a,
                jnp.where(
                    m == jnp.int32(SURVIVAL_META_MODE_FIGHT),
                    mf_a,
                    jnp.where(
                        m == jnp.int32(SURVIVAL_META_MODE_STONE),
                        stone_a,
                        jnp.where(
                            m == jnp.int32(SURVIVAL_META_MODE_COAL),
                            coal_a,
                            jnp.where(
                                m == jnp.int32(SURVIVAL_META_MODE_IRON),
                                iron_a,
                                jnp.where(
                                    m == jnp.int32(SURVIVAL_META_MODE_DIAMOND),
                                    diamond_a,
                                    jnp.int32(0),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
