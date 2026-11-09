"""Find-food training diagnostics (action distribution probes)."""

from __future__ import annotations

from policies.find_food import FindFoodPolicy
from training.diagnostics_mob_skills import (
    ACTION_LABELS,
    _freeze_find_food_cows,
    action_fractions_and_entropy,
)
from training.diagnostics_mob_skills import (
    make_action_count_fn as _make_action_count_fn,
)
from training.monster_world_bootstrap import bootstrap_find_food_state, make_find_food_obs

__all__ = [
    "ACTION_LABELS",
    "action_fractions_and_entropy",
    "make_action_count_fn",
]


def make_action_count_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    *,
    rollout_steps: int,
    hidden_size: int,
    num_actions: int,
    stochastic: bool,
    temperature: float,
    num_rollouts: int,
):
    return _make_action_count_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        policy_cls=FindFoodPolicy,
        bootstrap_state_fn=bootstrap_find_food_state,
        make_obs_fn=make_find_food_obs,
        apply_static_freeze_fn=_freeze_find_food_cows,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic=stochastic,
        temperature=temperature,
        num_rollouts=num_rollouts,
    )
