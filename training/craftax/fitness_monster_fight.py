"""OpenES fitness for monster_fight on full Craftax (melee mobs)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from configs import (
    FIND_WOOD_NUM_ACTIONS,
    MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS,
    MONSTER_FIGHT_HIDDEN_SIZE,
    MONSTER_FIGHT_KILL_REWARD,
    MONSTER_FIGHT_POLICY_TEMPERATURE,
    MONSTER_FIGHT_ROLLOUT_STEPS,
    MONSTER_FIGHT_STEP_PENALTY,
    MONSTER_FIGHT_WASTED_DO_PENALTY,
    MONSTER_TOUCH_PROGRESS_ALPHA,
)
from policies.monster_fight import MonsterFightPolicy
from training.craftax.bootstrap import make_monster_fight_obs
from training.craftax.bootstrap_training import bootstrap_monster_fight_state
from training.params import flat_params_to_pytree
from training.skill_backends import get_backend

_DO = jnp.int32(5)
_BIG = jnp.int32(1_000_000)


def make_env(variant: str = "craftax"):
    return get_backend(variant).make_env()


def _lvl_melee(state):
    lvl = state.player_level.astype(jnp.int32)
    m = state.melee_mobs
    return m.position[lvl], m.mask[lvl], m.health[lvl]


def _min_hostile_dist(player, pos, mask):
    p = player.astype(jnp.int32)
    d = jnp.abs(pos.astype(jnp.int32) - p).sum(axis=-1)
    d = jnp.where(mask, d, _BIG)
    any_h = jnp.any(mask)
    return jnp.min(d), any_h


def make_dual_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = MONSTER_FIGHT_ROLLOUT_STEPS,
    hidden_size: int = MONSTER_FIGHT_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "craftax",
    static_melee: bool = False,
    melee_type_id: int = 0,
):
    """Rollout returning ``(score_native, score_classic_formula, kill_rate)``."""

    @jax.jit
    def single_rollout(flat_params, rng, policy_temperature):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)
        rng, reset_rng, boot_rng = jax.random.split(rng, 3)
        _, state = env.reset(reset_rng, env_params)
        state = bootstrap_monster_fight_state(
            state,
            boot_rng,
            env_params,
            static_melee=static_melee,
            melee_type_id=melee_type_id,
        )
        lvl = state.player_level.astype(jnp.int32)
        init_alive = state.melee_mobs.mask[lvl].sum().astype(jnp.float32)
        denom = jnp.maximum(init_alive, 1.0)
        obs = make_monster_fight_obs(env, state)
        done = jnp.array(False)

        def step(carry, _):
            obs, state, rng, done = carry
            rng, prng, estep = jax.random.split(rng, 3)
            logits = MonsterFightPolicy(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )
            if stochastic_actions:
                temp = jnp.maximum(policy_temperature, jnp.float32(1e-6))
                la = jax.random.categorical(prng, logits / temp)
            else:
                la = jnp.argmax(logits)
            action = la + 1
            pos0, mask0, health0 = _lvl_melee(state)
            prev_d, any_h = _min_hostile_dist(state.player_position, pos0, mask0)
            prev_sum = health0.sum()
            nobs, nstate, _, term, _ = env.step(estep, state, action, env_params)
            pos1, mask1, health1 = _lvl_melee(nstate)
            kills = jnp.logical_and(mask0, jnp.logical_not(mask1)).astype(jnp.float32).sum()
            damage = jnp.maximum(prev_sum - health1.sum(), 0).astype(jnp.float32)
            new_d, _ = _min_hostile_dist(nstate.player_position, pos1, mask1)
            progress = (1.0 / (new_d + 1.0) - 1.0 / (prev_d + 1.0)) * any_h.astype(jnp.float32)
            wasted = jnp.logical_and(action == _DO, damage <= 0).astype(jnp.float32)
            do_hit = jnp.logical_and(action == _DO, damage > 0).astype(jnp.float32)
            alive = mask1.sum() > 0
            active = jnp.logical_not(done).astype(jnp.float32)
            # Native Extended: DO bonus scales with damage amount.
            r_native = active * (
                MONSTER_FIGHT_KILL_REWARD * kills
                + MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS * damage
                + MONSTER_TOUCH_PROGRESS_ALPHA * progress
                - MONSTER_FIGHT_WASTED_DO_PENALTY * wasted
                - MONSTER_FIGHT_STEP_PENALTY * alive.astype(jnp.float32)
            )
            # Classic formula: binary DO-hit bonus.
            r_classic = active * (
                MONSTER_FIGHT_KILL_REWARD * kills
                + MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS * do_hit
                + MONSTER_TOUCH_PROGRESS_ALPHA * progress
                - MONSTER_FIGHT_WASTED_DO_PENALTY * wasted
                - MONSTER_FIGHT_STEP_PENALTY * alive.astype(jnp.float32)
            )
            if static_melee:
                lock = mask0
                frozen_pos = jnp.where(lock[:, None], pos0, pos1)
                mm = nstate.melee_mobs
                mm = mm.replace(
                    position=mm.position.at[lvl].set(frozen_pos.astype(mm.position.dtype))
                )
                nstate = nstate.replace(melee_mobs=mm)
            done = jnp.logical_or(done, term)
            return (make_monster_fight_obs(env, nstate), nstate, rng, done), (
                r_native,
                r_classic,
            )

        (_, final_state, _, _), rewards = jax.lax.scan(
            step, (obs, state, rng, done), None, length=rollout_steps
        )
        final_alive = final_state.melee_mobs.mask[lvl].sum().astype(jnp.float32)
        kill_rate = jnp.clip((init_alive - final_alive) / denom, 0.0, 1.0)
        return rewards[0].sum(), rewards[1].sum(), kill_rate

    return single_rollout


def make_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = MONSTER_FIGHT_ROLLOUT_STEPS,
    hidden_size: int = MONSTER_FIGHT_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "craftax",
    static_melee: bool = False,
    melee_type_id: int = 0,
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
        static_melee=static_melee,
        melee_type_id=melee_type_id,
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
    rollout_steps=MONSTER_FIGHT_ROLLOUT_STEPS,
    num_rollouts=256,
    hidden_size=MONSTER_FIGHT_HIDDEN_SIZE,
    num_actions=FIND_WOOD_NUM_ACTIONS,
    stochastic_actions=True,
    variant: str = "craftax",
    static_melee: bool = False,
    melee_type_id: int = 0,
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
        static_melee=static_melee,
        melee_type_id=melee_type_id,
    )
    vm = jax.vmap(single, in_axes=(None, 0, None))

    @jax.jit
    def fitness_fn(
        flat_params,
        rng,
        active_num_rollouts=jnp.int32(num_rollouts),
        policy_temperature=jnp.float32(MONSTER_FIGHT_POLICY_TEMPERATURE),
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
    rollout_steps=MONSTER_FIGHT_ROLLOUT_STEPS,
    num_rollouts=256,
    hidden_size=MONSTER_FIGHT_HIDDEN_SIZE,
    num_actions=FIND_WOOD_NUM_ACTIONS,
    stochastic_actions=True,
    variant: str = "craftax",
    static_melee: bool = False,
    melee_type_id: int = 0,
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
        static_melee=static_melee,
        melee_type_id=melee_type_id,
    )
    vm = jax.vmap(single, in_axes=(None, 0, None))

    @jax.jit
    def fitness_fn(
        flat_params,
        rng,
        active_num_rollouts=jnp.int32(num_rollouts),
        policy_temperature=jnp.float32(MONSTER_FIGHT_POLICY_TEMPERATURE),
    ):
        keys = jax.random.split(rng, num_rollouts)
        scores, kill_rates = vm(flat_params, keys, policy_temperature)
        n = jnp.clip(active_num_rollouts, 1, num_rollouts)
        idx = jnp.arange(num_rollouts) < n
        m = idx.astype(jnp.float32)
        d = jnp.maximum(m.sum(), 1.0)
        return (scores * m).sum() / d, (kill_rates * m).sum() / d

    return fitness_fn
