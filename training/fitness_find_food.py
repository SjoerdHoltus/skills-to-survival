"""
Fitness for the find-food task: hunt bootstrap-spawned cows with a wooden sword.

Mirrors ``fitness_monster_fight``: same rollout shape/reward scaffolding, but
kill detection and proximity shaping target ``state.cows`` only.

Returned per rollout: ``(score_total, kill_rate)`` where kill_rate counts cow
deaths attributed to combat (mask True->False and health<=0).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import configs as ff_cfg
from configs import (
    FIND_FOOD_DO_FACING_COW_BONUS,
    FIND_FOOD_HIDDEN_SIZE,
    FIND_FOOD_KILL_REWARD,
    FIND_FOOD_POLICY_TEMPERATURE,
    FIND_FOOD_PROGRESS_ALPHA,
    FIND_FOOD_ROLLOUT_STEPS,
    FIND_FOOD_STEP_PENALTY,
    FIND_FOOD_WASTED_DO_PENALTY,
    FIND_WOOD_NUM_ACTIONS,
)
from policies.find_food import FindFoodPolicy
from training.monster_world_bootstrap import (
    bootstrap_find_food_state,
    make_find_food_obs,
    rebuild_mob_map_from_state,
)
from training.params import flat_params_to_pytree
from training.skill_variant import VARIANT_CRAFTAX, normalize_variant

_BIG_DIST = jnp.asarray(1_000_000, dtype=jnp.int32)
_DO_ACTION_VALUE = jnp.int32(5)


def _min_manhattan_to_active_cows(
    player_pos: jnp.ndarray, c_pos: jnp.ndarray, c_mask: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Min Manhattan distance and whether any cow is alive."""
    p = player_pos.astype(jnp.int32)
    cd = jnp.abs(c_pos.astype(jnp.int32) - p).sum(axis=-1)
    cd = jnp.where(c_mask, cd, _BIG_DIST)
    any_cow = jnp.any(c_mask)
    return jnp.min(cd), any_cow


def make_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = FIND_FOOD_ROLLOUT_STEPS,
    hidden_size: int = FIND_FOOD_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "classic",
):
    """JIT-compiled single rollout. Returns ``(score_total, kill_rate)``."""
    if normalize_variant(variant) == VARIANT_CRAFTAX:
        from training.craftax import fitness_find_food as cf

        return cf.make_single_rollout_fn(
            env,
            env_params,
            param_shapes,
            param_tree_def,
            rollout_steps=rollout_steps,
            hidden_size=hidden_size,
            num_actions=num_actions,
            stochastic_actions=stochastic_actions,
            variant=variant,
            static_cow=bool(ff_cfg.FIND_FOOD_STATIC_COW),
        )

    @jax.jit
    def single_rollout(flat_params: jnp.ndarray, rng: jax.Array, policy_temperature: jnp.ndarray):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)

        rng, reset_rng, boot_rng = jax.random.split(rng, 3)
        _, state = env.reset(reset_rng, env_params)
        state = bootstrap_find_food_state(state, boot_rng, env_params)

        initial_alive = jnp.sum(state.cows.mask).astype(jnp.float32)
        kill_rate_denom = jnp.maximum(initial_alive, jnp.float32(1.0))

        obs = make_find_food_obs(env, state)
        episode_done = jnp.array(False)

        def step_fn(carry, _):
            obs, state, rng, episode_done = carry
            rng, policy_rng, env_step_rng = jax.random.split(rng, 3)

            logits = FindFoodPolicy(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )
            if stochastic_actions:
                temp = jnp.maximum(jnp.float32(policy_temperature), jnp.float32(1e-6))
                scaled_logits = logits / temp
                local_action = jax.random.categorical(policy_rng, scaled_logits)
            else:
                local_action = jnp.argmax(logits)
            action = local_action + 1

            prev_c = state.cows
            prev_player_pos = state.player_position.astype(jnp.int32)
            prev_min_d, prev_any_cow = _min_manhattan_to_active_cows(
                prev_player_pos, prev_c.position, prev_c.mask
            )
            prev_health_sum = prev_c.health.astype(jnp.int32).sum()

            new_obs, new_state, _reward, done, _info = env.step(
                env_step_rng, state, action, env_params
            )

            post_c = new_state.cows
            c_mask_flip = jnp.logical_and(prev_c.mask, jnp.logical_not(post_c.mask))
            c_dead = post_c.health.astype(jnp.int32) <= 0
            c_kills = jnp.logical_and(c_mask_flip, c_dead).astype(jnp.int32).sum()
            num_kills_this_step = c_kills.astype(jnp.float32)

            new_health_sum = post_c.health.astype(jnp.int32).sum()
            damage_dealt = jnp.maximum(prev_health_sum - new_health_sum, 0).astype(jnp.float32)
            did_damage = damage_dealt > 0

            if ff_cfg.FIND_FOOD_STATIC_COW:
                lock_c = prev_c.mask
                frozen_c = post_c.replace(
                    position=jnp.where(
                        lock_c[:, None],
                        prev_c.position.astype(post_c.position.dtype),
                        post_c.position,
                    ),
                )
                new_state = new_state.replace(cows=frozen_c)
                new_state = new_state.replace(mob_map=rebuild_mob_map_from_state(new_state))

            new_obs = make_find_food_obs(env, new_state)

            active = jnp.logical_not(episode_done)
            active_f = active.astype(jnp.float32)

            new_player_pos = new_state.player_position.astype(jnp.int32)
            new_min_d, any_cow_now = _min_manhattan_to_active_cows(
                new_player_pos,
                new_state.cows.position,
                new_state.cows.mask,
            )
            prev_phi = 1.0 / (prev_min_d.astype(jnp.float32) + 1.0)
            new_phi = 1.0 / (new_min_d.astype(jnp.float32) + 1.0)
            no_kill_this_step = jnp.logical_not(num_kills_this_step > 0)
            both_have_cow = (prev_any_cow & any_cow_now & no_kill_this_step).astype(jnp.float32)
            progress = (new_phi - prev_phi) * both_have_cow

            chose_do = action == _DO_ACTION_VALUE
            wasted_do = jnp.logical_and(chose_do, jnp.logical_not(did_damage))
            do_bonus = jnp.logical_and(chose_do, did_damage).astype(jnp.float32)
            wasted_do_f = wasted_do.astype(jnp.float32)

            any_alive_f = any_cow_now.astype(jnp.float32)

            step_score = active_f * (
                FIND_FOOD_KILL_REWARD * num_kills_this_step
                + FIND_FOOD_DO_FACING_COW_BONUS * do_bonus
                + FIND_FOOD_PROGRESS_ALPHA * progress
                - FIND_FOOD_WASTED_DO_PENALTY * wasted_do_f
                - FIND_FOOD_STEP_PENALTY * any_alive_f
            )

            kills_for_metric = num_kills_this_step * active_f
            episode_done = jnp.logical_or(episode_done, done)
            return (new_obs, new_state, rng, episode_done), (
                step_score,
                kills_for_metric,
            )

        (_, _, _, _), step_outs = jax.lax.scan(
            step_fn,
            (obs, state, rng, episode_done),
            None,
            length=rollout_steps,
        )

        step_scores, step_kills = step_outs
        total_kills = step_kills.sum()
        kill_rate = jnp.clip(total_kills / kill_rate_denom, 0.0, 1.0)
        return step_scores.sum(), kill_rate

    return single_rollout


def make_fitness_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = FIND_FOOD_ROLLOUT_STEPS,
    num_rollouts: int = 256,
    hidden_size: int = FIND_FOOD_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "classic",
):
    """Returns ``(score_mean, kill_rate_mean)`` over ``num_rollouts``."""
    if normalize_variant(variant) == VARIANT_CRAFTAX:
        from training.craftax import fitness_find_food as cf

        return cf.make_fitness_fn(
            env,
            env_params,
            param_shapes,
            param_tree_def,
            rollout_steps=rollout_steps,
            num_rollouts=num_rollouts,
            hidden_size=hidden_size,
            num_actions=num_actions,
            stochastic_actions=stochastic_actions,
            variant=variant,
            static_cow=bool(ff_cfg.FIND_FOOD_STATIC_COW),
        )

    single_rollout = make_single_rollout_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic_actions=stochastic_actions,
        variant=variant,
    )
    vmapped_rollouts = jax.vmap(single_rollout, in_axes=(None, 0, None))

    @jax.jit
    def fitness_fn(
        flat_params: jnp.ndarray,
        rng: jax.Array,
        active_num_rollouts: jnp.ndarray = jnp.int32(num_rollouts),
        policy_temperature: jnp.ndarray = jnp.float32(FIND_FOOD_POLICY_TEMPERATURE),
    ):
        rollout_rngs = jax.random.split(rng, num_rollouts)
        scores, kill_rates = vmapped_rollouts(flat_params, rollout_rngs, policy_temperature)
        n = jnp.clip(active_num_rollouts.astype(jnp.int32), 1, jnp.int32(num_rollouts))
        idx = jnp.arange(num_rollouts, dtype=jnp.int32)
        mask = (idx < n).astype(jnp.float32)
        denom = jnp.maximum(mask.sum(), 1.0)
        score_mean = (scores * mask).sum() / denom
        kill_mean = (kill_rates * mask).sum() / denom
        return score_mean, kill_mean

    return fitness_fn


def make_env(variant: str = "classic"):
    from training.skill_backends import get_backend

    return get_backend(variant).make_env()
