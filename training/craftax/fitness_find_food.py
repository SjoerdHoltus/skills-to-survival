"""OpenES fitness for find_food on full Craftax (passive mobs / cows)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
from training.craftax.bootstrap import make_find_food_obs
from training.craftax.bootstrap_training import bootstrap_find_food_state
from training.params import flat_params_to_pytree
from training.skill_backends import get_backend

_DO = jnp.int32(5)
_BIG = jnp.int32(1_000_000)


def make_env(variant: str = "craftax"):
    return get_backend(variant).make_env()


def _lvl_passive(state):
    lvl = state.player_level.astype(jnp.int32)
    p = state.passive_mobs
    return p.position[lvl], p.mask[lvl], p.health[lvl]


def make_dual_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = FIND_FOOD_ROLLOUT_STEPS,
    hidden_size: int = FIND_FOOD_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "craftax",
    static_cow: bool = False,
):
    """Rollout returning ``(score_native, score_classic_formula, kill_rate)``."""

    @jax.jit
    def single_rollout(flat_params, rng, policy_temperature):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)
        rng, reset_rng, boot_rng = jax.random.split(rng, 3)
        _, state = env.reset(reset_rng, env_params)
        state = bootstrap_find_food_state(state, boot_rng, env_params, static_cow=static_cow)
        lvl = state.player_level.astype(jnp.int32)
        init_alive = state.passive_mobs.mask[lvl].sum().astype(jnp.float32)
        denom = jnp.maximum(init_alive, 1.0)
        obs = make_find_food_obs(env, state)
        done = jnp.array(False)

        def step(carry, _):
            obs, state, rng, done = carry
            rng, prng, estep = jax.random.split(rng, 3)
            logits = FindFoodPolicy(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )
            if stochastic_actions:
                temp = jnp.maximum(policy_temperature, jnp.float32(1e-6))
                la = jax.random.categorical(prng, logits / temp)
            else:
                la = jnp.argmax(logits)
            action = la + 1
            pos0, mask0, health0 = _lvl_passive(state)
            p = state.player_position.astype(jnp.int32)
            d0 = jnp.abs(pos0.astype(jnp.int32) - p).sum(axis=-1)
            d0m = jnp.where(mask0, d0, _BIG).min()
            prev_sum = health0.sum()
            nobs, nstate, _, term, _ = env.step(estep, state, action, env_params)
            pos1, mask1, health1 = _lvl_passive(nstate)
            damage = jnp.maximum(prev_sum - health1.sum(), 0).astype(jnp.float32)
            d1 = jnp.abs(pos1.astype(jnp.int32) - nstate.player_position.astype(jnp.int32)).sum(
                axis=-1
            )
            d1m = jnp.where(mask1, d1, _BIG).min()
            progress = (1.0 / (d1m + 1.0) - 1.0 / (d0m + 1.0)) * jnp.any(mask0).astype(jnp.float32)
            wasted = jnp.logical_and(action == _DO, damage <= 0).astype(jnp.float32)
            alive = mask1.sum() > 0
            active = jnp.logical_not(done).astype(jnp.float32)
            r_native = active * (
                FIND_FOOD_KILL_REWARD * damage
                + FIND_FOOD_DO_FACING_COW_BONUS * damage
                + FIND_FOOD_PROGRESS_ALPHA * progress
                - FIND_FOOD_WASTED_DO_PENALTY * wasted
                - FIND_FOOD_STEP_PENALTY * alive.astype(jnp.float32)
            )
            # Classic formula: pay for kills + binary DO-hit bonus (not raw damage).
            kills = jnp.logical_and(mask0, jnp.logical_not(mask1)).astype(jnp.float32).sum()
            do_hit = jnp.logical_and(action == _DO, damage > 0).astype(jnp.float32)
            r_classic = active * (
                FIND_FOOD_KILL_REWARD * kills
                + FIND_FOOD_DO_FACING_COW_BONUS * do_hit
                + FIND_FOOD_PROGRESS_ALPHA * progress
                - FIND_FOOD_WASTED_DO_PENALTY * wasted
                - FIND_FOOD_STEP_PENALTY * alive.astype(jnp.float32)
            )
            if static_cow:
                lock = mask0
                frozen_pos = jnp.where(lock[:, None], pos0, pos1)
                pm = nstate.passive_mobs
                pm = pm.replace(
                    position=pm.position.at[lvl].set(frozen_pos.astype(pm.position.dtype))
                )
                nstate = nstate.replace(passive_mobs=pm)
            done = jnp.logical_or(done, term)
            return (make_find_food_obs(env, nstate), nstate, rng, done), (
                r_native,
                r_classic,
            )

        (_, final_state, _, _), rewards = jax.lax.scan(
            step, (obs, state, rng, done), None, length=rollout_steps
        )
        final_alive = final_state.passive_mobs.mask[lvl].sum().astype(jnp.float32)
        kill_rate = jnp.clip((init_alive - final_alive) / denom, 0.0, 1.0)
        score_native = rewards[0].sum()
        score_classic = rewards[1].sum()
        return score_native, score_classic, kill_rate

    return single_rollout


def make_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = FIND_FOOD_ROLLOUT_STEPS,
    hidden_size: int = FIND_FOOD_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "craftax",
    static_cow: bool = False,
):
    dual = make_dual_single_rollout_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic_actions=stochastic_actions,
        variant=variant,
        static_cow=static_cow,
    )

    def single_rollout(flat_params, rng, policy_temperature):
        score_native, _score_classic, kill_rate = dual(flat_params, rng, policy_temperature)
        return score_native, kill_rate

    return single_rollout


def make_dual_fitness_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps=FIND_FOOD_ROLLOUT_STEPS,
    num_rollouts=256,
    hidden_size=FIND_FOOD_HIDDEN_SIZE,
    num_actions=FIND_WOOD_NUM_ACTIONS,
    stochastic_actions=True,
    variant: str = "craftax",
    static_cow: bool = False,
):
    """Return ``(score_native, score_classic_formula, kill_rate)`` means."""
    single = make_dual_single_rollout_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic_actions=stochastic_actions,
        variant=variant,
        static_cow=static_cow,
    )
    vm = jax.vmap(single, in_axes=(None, 0, None))

    @jax.jit
    def fitness_fn(
        flat_params,
        rng,
        active_num_rollouts=jnp.int32(num_rollouts),
        policy_temperature=jnp.float32(FIND_FOOD_POLICY_TEMPERATURE),
    ):
        keys = jax.random.split(rng, num_rollouts)
        score_n, score_c, kill_rates = vm(flat_params, keys, policy_temperature)
        n = jnp.clip(active_num_rollouts, 1, num_rollouts)
        m = (jnp.arange(num_rollouts) < n).astype(jnp.float32)
        d = jnp.maximum(m.sum(), 1.0)
        return (
            (score_n * m).sum() / d,
            (score_c * m).sum() / d,
            (kill_rates * m).sum() / d,
        )

    return fitness_fn


def make_fitness_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps=FIND_FOOD_ROLLOUT_STEPS,
    num_rollouts=256,
    hidden_size=FIND_FOOD_HIDDEN_SIZE,
    num_actions=FIND_WOOD_NUM_ACTIONS,
    stochastic_actions=True,
    variant: str = "craftax",
    static_cow: bool = False,
):
    single = make_single_rollout_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic_actions=stochastic_actions,
        variant=variant,
        static_cow=static_cow,
    )
    vm = jax.vmap(single, in_axes=(None, 0, None))

    @jax.jit
    def fitness_fn(
        flat_params,
        rng,
        active_num_rollouts=jnp.int32(num_rollouts),
        policy_temperature=jnp.float32(FIND_FOOD_POLICY_TEMPERATURE),
    ):
        keys = jax.random.split(rng, num_rollouts)
        scores, kill_rates = vm(flat_params, keys, policy_temperature)
        n = jnp.clip(active_num_rollouts, 1, num_rollouts)
        idx = jnp.arange(num_rollouts) < n
        m = idx.astype(jnp.float32)
        d = jnp.maximum(m.sum(), 1.0)
        return (scores * m).sum() / d, (kill_rates * m).sum() / d

    return fitness_fn
