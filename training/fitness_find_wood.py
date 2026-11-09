"""OpenES fitness for find-wood (12-D obs, tree bootstrap)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BlockType

import configs as fw_cfg
from configs import (
    FIND_WOOD_HIDDEN_SIZE,
    FIND_WOOD_NUM_ACTIONS,
    FIND_WOOD_ROLLOUT_STEPS,
    MANHATTAN_THRESHOLD,
    MAX_RESOURCES,
    WOOD_BONUS,
)
from policies.find_wood import FindWoodPolicy
from training.params import flat_params_to_pytree
from training.skill_backends import get_backend

_BIG_DIST = jnp.asarray(1_000_000, dtype=jnp.int32)
_TREE_V = jnp.int32(BlockType.TREE.value)
_MAX_WOOD = jnp.int32(MAX_RESOURCES)
_REACH_THR_F = jnp.float32(MANHATTAN_THRESHOLD)
_DO_ACTION_VALUE = jnp.int32(5)
_DELTA_BY_DIR = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)


def _nearest_tree_manhattan(player_pos: jnp.ndarray, mmap: jnp.ndarray) -> jnp.ndarray:
    pp = player_pos.astype(jnp.int32)
    h, w = mmap.shape
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    is_tree = mmap == _TREE_V
    md = jnp.abs(rr - pp[0]) + jnp.abs(cc - pp[1])
    md_masked = jnp.where(is_tree, md, _BIG_DIST)
    return jnp.min(md_masked)


def make_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = FIND_WOOD_ROLLOUT_STEPS,
    hidden_size: int = FIND_WOOD_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "classic",
):
    backend = get_backend(variant)

    @jax.jit
    def single_rollout(
        flat_params: jnp.ndarray,
        rollout_key: jax.Array,
        policy_temperature: jnp.ndarray,
    ):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)

        carry_key, reset_key, boot_key = jax.random.split(rollout_key, 3)
        _, state = env.reset(reset_key, env_params)
        state = backend.bootstrap_wood_state(state, boot_key, env_params)

        obs = backend.make_find_wood_obs(env, state)
        episode_done = jnp.array(False)
        init_nm = _nearest_tree_manhattan(
            state.player_position.astype(jnp.int32), backend.level_map(state)
        ).astype(jnp.float32)

        def step_fn(carry, _):
            obs, state, rng, episode_done, min_d_ever_in = carry
            active_pre = jnp.logical_not(episode_done).astype(jnp.float32)
            rng, policy_rng, env_step_rng = jax.random.split(rng, 3)

            logits = FindWoodPolicy(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )

            if stochastic_actions:
                temp = jnp.maximum(jnp.float32(policy_temperature), jnp.float32(1e-6))
                local_action = jax.random.categorical(policy_rng, logits / temp)
            else:
                local_action = jnp.argmax(logits)
            action = local_action + 1

            prev_wood = state.inventory.wood.astype(jnp.int32)
            new_obs, new_state, env_reward, done, _info = env.step(
                env_step_rng, state, action, env_params
            )

            new_wood_i = new_state.inventory.wood.astype(jnp.int32)
            wood_gain = jnp.maximum(
                jnp.float32(new_wood_i) - jnp.float32(prev_wood),
                jnp.float32(0.0),
            )
            new_min_d = _nearest_tree_manhattan(
                new_state.player_position.astype(jnp.int32), backend.level_map(new_state)
            ).astype(jnp.float32)

            min_d_ever_next = jnp.minimum(min_d_ever_in, new_min_d)
            active = jnp.logical_not(episode_done).astype(jnp.float32)

            step_score = active * (WOOD_BONUS * wood_gain)

            new_obs_final = backend.make_find_wood_obs(env, new_state)
            full_done = new_wood_i >= _MAX_WOOD
            episode_done = jnp.logical_or(
                jnp.logical_or(episode_done, done),
                full_done,
            )
            return (
                new_obs_final,
                new_state,
                rng,
                episode_done,
                min_d_ever_next,
            ), (step_score, active_pre)

        (_, final_state, _, _, min_d_best), (step_scores, active_pres) = jax.lax.scan(
            step_fn,
            (obs, state, carry_key, episode_done, init_nm),
            None,
            length=rollout_steps,
        )

        scores_sum = step_scores.sum()
        steps_active_sum = active_pres.sum()
        final_wood = final_state.inventory.wood.astype(jnp.float32)
        success = (final_state.inventory.wood.astype(jnp.int32) >= _MAX_WOOD).astype(jnp.float32)
        has_tree = init_nm < jnp.float32(_BIG_DIST)
        reached_tree = jnp.logical_and(has_tree, min_d_best <= _REACH_THR_F).astype(jnp.float32)
        return (
            scores_sum,
            success,
            steps_active_sum,
            final_wood,
            reached_tree,
            min_d_best,
        )

    return single_rollout


def make_fitness_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = FIND_WOOD_ROLLOUT_STEPS,
    num_rollouts: int = 256,
    hidden_size: int = FIND_WOOD_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "classic",
):
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
        policy_temperature: jnp.ndarray = jnp.float32(fw_cfg.FIND_WOOD_POLICY_TEMPERATURE),
    ):
        rollout_keys = jax.random.split(rng, num_rollouts)
        (
            scores,
            success_flags,
            _steps_unused,
            final_woods,
            reached_flags,
            _md_unused,
        ) = vmapped_rollouts(flat_params, rollout_keys, policy_temperature)
        n = jnp.clip(active_num_rollouts.astype(jnp.int32), 1, jnp.int32(num_rollouts))
        idx = jnp.arange(num_rollouts, dtype=jnp.int32)
        mask = (idx < n).astype(jnp.float32)
        denom = jnp.maximum(mask.sum(), 1.0)
        score_mean = (scores * mask).sum() / denom
        success_mean = (success_flags * mask).sum() / denom
        reach_mean = (reached_flags * mask).sum() / denom
        wood_mean = (final_woods * mask).sum() / denom
        return score_mean, success_mean, reach_mean, wood_mean

    return fitness_fn


def make_env(variant: str = "classic"):
    return get_backend(variant).make_env()
