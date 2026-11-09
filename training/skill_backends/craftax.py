"""Craftax Extended skill-training backend (same interface as ``classic``)."""

from __future__ import annotations

import jax

from training.bootstrap_ore import wall_count_for_curriculum
from training.craftax.bootstrap import (
    make_collect_ore_obs,
    make_collect_stone_obs,
    make_drink_water_obs,
    make_find_food_obs,
    make_find_wood_obs,
    make_monster_fight_obs,
)
from training.craftax.bootstrap_training import (
    bootstrap_find_food_state,
    bootstrap_monster_fight_state,
    bootstrap_stone_state,
    bootstrap_water_state,
    bootstrap_wood_state,
)
from training.craftax.bootstrap_training import bootstrap_ore_state as _bootstrap_ore_state
from training.craftax.env import make_skill_env as make_env
from training.craftax.state_helpers import level_map

__all__ = [
    "bootstrap_find_food_state",
    "bootstrap_monster_fight_state",
    "bootstrap_ore_state",
    "bootstrap_stone_state",
    "bootstrap_water_state",
    "bootstrap_wood_state",
    "level_map",
    "make_collect_ore_obs",
    "make_collect_stone_obs",
    "make_drink_water_obs",
    "make_env",
    "make_find_food_obs",
    "make_find_wood_obs",
    "make_monster_fight_obs",
]


def bootstrap_ore_state(
    state,
    boot_rng,
    env_params,
    target_id,
    wall_count=None,
    curriculum_gen=None,
    curriculum_num_gens=None,
):
    if wall_count is None:
        _, boot_rng_wall = jax.random.split(boot_rng)
        wall_count = wall_count_for_curriculum(boot_rng_wall, curriculum_gen, curriculum_num_gens)
    return _bootstrap_ore_state(state, boot_rng, env_params, target_id, wall_count=wall_count)
