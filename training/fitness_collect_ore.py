"""OpenES fitness for collect-ore (15-D obs, walled ore behind stone).

Training uses a single ``COLLECT_ORE_TRAINING_TARGET``; deploy/eval pass another
``target_id`` via bootstrap obs only. Shaping: env + ore gain + DO-facing bonuses +
wall-break bonus - step penalty.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BlockType

import configs as co_cfg
from configs import (
    COLLECT_ORE_BONUS,
    COLLECT_ORE_HIDDEN_SIZE,
    COLLECT_ORE_NUM_ACTIONS,
    COLLECT_ORE_REACH_MANHATTAN_THRESHOLD,
    COLLECT_ORE_ROLLOUT_STEPS,
    COLLECT_ORE_TRAINING_TARGET_ID,
    COLLECT_ORE_WALL_BREAK_BONUS,
)
from policies.collect_ore import CollectOrePolicy
from training.bootstrap_ore import (
    TARGET_BLOCK_VALUES,
    TARGET_CAPS,
    collect_ore_has_target,
    wall_count_for_curriculum,
)
from training.params import flat_params_to_pytree
from training.skill_backends import get_backend

_BIG_DIST = jnp.asarray(1_000_000, dtype=jnp.int32)
_STONE_V = jnp.int32(BlockType.STONE.value)
_DO_ACTION_VALUE = jnp.int32(5)
_REACH_THR_F = jnp.float32(COLLECT_ORE_REACH_MANHATTAN_THRESHOLD)
_DELTA_BY_DIR = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)


def _inventory_amount(state, target_id: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.switch(
        target_id.astype(jnp.int32),
        [
            lambda s: s.inventory.coal.astype(jnp.int32),
            lambda s: s.inventory.iron.astype(jnp.int32),
            lambda s: s.inventory.diamond.astype(jnp.int32),
        ],
        state,
    )


def _nearest_target_manhattan(
    player_pos: jnp.ndarray, mmap: jnp.ndarray, block_value: jnp.ndarray
) -> jnp.ndarray:
    pp = player_pos.astype(jnp.int32)
    h, w = mmap.shape
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    is_target = mmap == block_value.astype(jnp.int32)
    md = jnp.abs(rr - pp[0]) + jnp.abs(cc - pp[1])
    md_masked = jnp.where(is_target, md, _BIG_DIST)
    return jnp.min(md_masked)


def make_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = COLLECT_ORE_ROLLOUT_STEPS,
    hidden_size: int = COLLECT_ORE_HIDDEN_SIZE,
    num_actions: int = COLLECT_ORE_NUM_ACTIONS,
    stochastic_actions: bool = True,
    fixed_target_id: jnp.ndarray | None = None,
    fixed_wall_count: jnp.ndarray | None = None,
    variant: str = "classic",
):
    backend = get_backend(variant)
    train_tid = jnp.int32(COLLECT_ORE_TRAINING_TARGET_ID)

    @jax.jit
    def single_rollout(
        flat_params: jnp.ndarray,
        rollout_key: jax.Array,
        policy_temperature: jnp.ndarray,
        curriculum_gen: jnp.ndarray,
        curriculum_num_gens: jnp.ndarray,
    ):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)

        carry_key, reset_key, boot_key, shell_key = jax.random.split(rollout_key, 4)
        _, state = env.reset(reset_key, env_params)

        target_id = fixed_target_id.astype(jnp.int32) if fixed_target_id is not None else train_tid

        shell_count = (
            wall_count_for_curriculum(shell_key, curriculum_gen, curriculum_num_gens)
            if fixed_wall_count is None
            else fixed_wall_count.astype(jnp.int32)
        )

        state = backend.bootstrap_ore_state(
            state,
            boot_key,
            env_params,
            target_id,
            wall_count=shell_count,
            curriculum_gen=curriculum_gen,
            curriculum_num_gens=curriculum_num_gens,
        )
        block_value = TARGET_BLOCK_VALUES[target_id]
        cap = TARGET_CAPS[target_id]

        obs = backend.make_collect_ore_obs(env, state, target_id)
        target_in_fov = collect_ore_has_target(obs)
        episode_done = jnp.array(False)
        init_nm = _nearest_target_manhattan(
            state.player_position.astype(jnp.int32), backend.level_map(state), block_value
        ).astype(jnp.float32)

        def step_fn(carry, _):
            obs, state, rng, episode_done, min_d_ever_in, tid, cap_i, bv = carry
            active_pre = jnp.logical_not(episode_done).astype(jnp.float32)
            rng, policy_rng, env_step_rng = jax.random.split(rng, 3)

            logits = CollectOrePolicy(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )

            if stochastic_actions:
                temp = jnp.maximum(jnp.float32(policy_temperature), jnp.float32(1e-6))
                local_action = jax.random.categorical(policy_rng, logits / temp)
            else:
                local_action = jnp.argmax(logits)
            action = local_action + 1

            prev_amt = _inventory_amount(state, tid)
            stone_tiles_before = (backend.level_map(state) == _STONE_V).sum()

            new_obs, new_state, env_reward, done, _info = env.step(
                env_step_rng, state, action, env_params
            )

            new_amt = _inventory_amount(new_state, tid)
            gain = jnp.maximum(
                jnp.float32(new_amt) - jnp.float32(prev_amt),
                jnp.float32(0.0),
            )
            stone_tiles_after = (backend.level_map(new_state) == _STONE_V).sum()
            broke_stone = stone_tiles_after < stone_tiles_before
            new_min_d = _nearest_target_manhattan(
                new_state.player_position.astype(jnp.int32), backend.level_map(new_state), bv
            ).astype(jnp.float32)
            min_d_ever_next = jnp.minimum(min_d_ever_in, new_min_d)

            active = jnp.logical_not(episode_done).astype(jnp.float32)

            step_score = active * (
                COLLECT_ORE_BONUS * gain
                + COLLECT_ORE_WALL_BREAK_BONUS * broke_stone.astype(jnp.float32)
            )

            new_obs_final = backend.make_collect_ore_obs(env, new_state, tid)
            full_done = new_amt >= cap_i
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
                tid,
                cap_i,
                bv,
            ), (step_score, active_pre, gain)

        (
            (_, final_state, _, _, min_d_best, tid_f, cap, _),
            (
                step_scores,
                active_pres,
                gains,
            ),
        ) = jax.lax.scan(
            step_fn,
            (obs, state, carry_key, episode_done, init_nm, target_id, cap, block_value),
            None,
            length=rollout_steps,
        )

        scores_sum = step_scores.sum()
        final_amt = _inventory_amount(final_state, tid_f).astype(jnp.float32)
        success = (final_amt >= cap.astype(jnp.float32)).astype(jnp.float32)
        has_target = init_nm < jnp.float32(_BIG_DIST)
        reached_target = jnp.logical_and(has_target, min_d_best <= _REACH_THR_F).astype(jnp.float32)
        mined_any = (gains.sum() > 0.0).astype(jnp.float32)
        return (
            scores_sum,
            success,
            active_pres.sum(),
            final_amt,
            reached_target,
            mined_any,
            shell_count.astype(jnp.float32),
            target_in_fov,
        )

    return single_rollout


def make_fitness_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = COLLECT_ORE_ROLLOUT_STEPS,
    num_rollouts: int = 256,
    hidden_size: int = COLLECT_ORE_HIDDEN_SIZE,
    num_actions: int = COLLECT_ORE_NUM_ACTIONS,
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
    vmapped_rollouts = jax.vmap(single_rollout, in_axes=(None, 0, None, None, None))

    @jax.jit
    def fitness_fn(
        flat_params: jnp.ndarray,
        rng: jax.Array,
        active_num_rollouts: jnp.ndarray = jnp.int32(num_rollouts),
        policy_temperature: jnp.ndarray = jnp.float32(co_cfg.COLLECT_ORE_POLICY_TEMPERATURE_START),
        curriculum_gen: jnp.ndarray = jnp.int32(0),
        curriculum_num_gens: jnp.ndarray = jnp.int32(1),
    ):
        rollout_keys = jax.random.split(rng, num_rollouts)
        (
            scores,
            success_flags,
            _steps,
            final_amts,
            reached_flags,
            mined_flags,
            shell_counts,
            _target_in_fov,
        ) = vmapped_rollouts(
            flat_params,
            rollout_keys,
            policy_temperature,
            curriculum_gen,
            curriculum_num_gens,
        )
        n = jnp.clip(active_num_rollouts.astype(jnp.int32), 1, jnp.int32(num_rollouts))
        idx = jnp.arange(num_rollouts, dtype=jnp.int32)
        mask = (idx < n).astype(jnp.float32)
        denom = jnp.maximum(mask.sum(), 1.0)
        return (
            (scores * mask).sum() / denom,
            (success_flags * mask).sum() / denom,
            (reached_flags * mask).sum() / denom,
            (final_amts * mask).sum() / denom,
            (mined_flags * mask).sum() / denom,
            (shell_counts * mask).sum() / denom,
        )

    return fitness_fn


def make_env(variant: str = "classic"):
    return get_backend(variant).make_env()
