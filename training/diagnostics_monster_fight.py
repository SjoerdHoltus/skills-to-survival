"""Monster-fight training diagnostics (action distribution probes)."""

from __future__ import annotations

from policies.monster_fight import MonsterFightPolicy
from training.diagnostics_mob_skills import (
    ACTION_LABELS,
    _freeze_monster_hostiles,
    action_fractions_and_entropy,
)
from training.diagnostics_mob_skills import (
    make_action_count_fn as _make_action_count_fn,
)
from training.monster_world_bootstrap import (
    bootstrap_monster_fight_state,
    make_monster_fight_obs,
)

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
        policy_cls=MonsterFightPolicy,
        bootstrap_state_fn=bootstrap_monster_fight_state,
        make_obs_fn=make_monster_fight_obs,
        apply_static_freeze_fn=_freeze_monster_hostiles,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic=stochastic,
        temperature=temperature,
        num_rollouts=num_rollouts,
    )
