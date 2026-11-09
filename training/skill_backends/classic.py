"""Craftax-Classic skill-training backend: env factory, bootstraps and feature filters."""

from __future__ import annotations

from craftax.craftax_env import make_craftax_env_from_name

from training.bootstrap_ore import bootstrap_ore_state, make_collect_ore_obs
from training.bootstrap_stone import bootstrap_stone_state, make_collect_stone_obs
from training.bootstrap_water import bootstrap_water_state, make_drink_water_obs
from training.bootstrap_wood import bootstrap_wood_state, make_find_wood_obs
from training.monster_world_bootstrap import (
    bootstrap_find_food_state,
    bootstrap_monster_fight_state,
    make_find_food_obs,
    make_monster_fight_obs,
)

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


def make_env():
    """Classic env with natural mob spawning disabled; bootstraps place mobs explicitly."""
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
        mob_despawn_distance=10_000,
    )
    return env, env_params


def level_map(state):
    return state.map
