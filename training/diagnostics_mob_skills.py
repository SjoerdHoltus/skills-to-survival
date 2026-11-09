"""Optional ES training diagnostics: action histograms over probe rollouts."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

import configs as rl_cfg
from training.params import flat_params_to_pytree

# Local policy indices 0..4 -> Craftax actions LEFT, RIGHT, UP, DOWN, DO.
ACTION_LABELS = ("left", "right", "up", "down", "do")


def action_fractions_and_entropy(counts: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """Normalized action fractions and Shannon entropy in bits."""
    total = counts.sum()
    fracs = counts / jnp.maximum(total, 1.0)
    p = jnp.clip(fracs, 1e-8, 1.0)
    entropy_nats = -jnp.sum(p * jnp.log(p))
    entropy_bits = float(entropy_nats / jnp.log(2.0))
    return fracs, entropy_bits


def make_action_count_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    *,
    policy_cls,
    bootstrap_state_fn: Callable,
    make_obs_fn: Callable,
    apply_static_freeze_fn: Callable | None,
    rollout_steps: int,
    hidden_size: int,
    num_actions: int,
    stochastic: bool,
    temperature: float,
    num_rollouts: int,
):
    """Return ``fn(flat_params, rng) -> float32 action counts`` of shape ``(num_actions,)``."""

    @jax.jit
    def count_actions(flat_params: jnp.ndarray, rng: jax.Array):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)
        rollout_rngs = jax.random.split(rng, num_rollouts)
        temp = jnp.float32(temperature)

        def one_rollout(rollout_key: jax.Array):
            carry_key, reset_key, boot_key = jax.random.split(rollout_key, 3)
            _, state = env.reset(reset_key, env_params)
            state = bootstrap_state_fn(state, boot_key, env_params)
            obs = make_obs_fn(env, state)
            counts = jnp.zeros((num_actions,), dtype=jnp.int32)

            def step_fn(carry, _):
                obs_in, state_in, rng_in, counts_in = carry
                rng_in, policy_rng, step_rng = jax.random.split(rng_in, 3)

                logits = policy_cls(hidden_size=hidden_size, num_actions=num_actions).apply(
                    {"params": params}, obs_in
                )
                if stochastic:
                    t = jnp.maximum(temp, jnp.float32(1e-6))
                    local_action = jax.random.categorical(policy_rng, logits / t)
                else:
                    local_action = jnp.argmax(logits)
                counts_out = counts_in.at[local_action].add(jnp.int32(1))
                action = local_action + jnp.int32(1)

                new_obs, new_state, _reward, done, _info = env.step(
                    step_rng, state_in, action, env_params
                )
                if apply_static_freeze_fn is not None:
                    new_state = apply_static_freeze_fn(state_in, new_state)
                new_obs = make_obs_fn(env, new_state)
                return (new_obs, new_state, rng_in, counts_out), None

            (_, _, _, counts_out), _ = jax.lax.scan(
                step_fn,
                (obs, state, carry_key, counts),
                None,
                length=rollout_steps,
            )
            return counts_out

        per_rollout = jax.vmap(one_rollout)(rollout_rngs)
        return per_rollout.sum(axis=0).astype(jnp.float32)

    return count_actions


def _freeze_find_food_cows(state_in, state_out):
    if not rl_cfg.FIND_FOOD_STATIC_COW:
        return state_out
    from training.monster_world_bootstrap import rebuild_mob_map_from_state

    prev_c = state_in.cows
    lock_c = prev_c.mask
    frozen_c = state_out.cows.replace(
        position=jnp.where(
            lock_c[:, None],
            prev_c.position.astype(state_out.cows.position.dtype),
            state_out.cows.position,
        ),
    )
    out = state_out.replace(cows=frozen_c)
    return out.replace(mob_map=rebuild_mob_map_from_state(out))


def _freeze_monster_hostiles(state_in, state_out):
    from training.monster_world_bootstrap import rebuild_mob_map_from_state

    out = state_out
    if rl_cfg.MONSTER_FIGHT_STATIC_ZOMBIE:
        prev_z = state_in.zombies
        lock_z = prev_z.mask
        frozen_z = out.zombies.replace(
            position=jnp.where(
                lock_z[:, None],
                prev_z.position.astype(out.zombies.position.dtype),
                out.zombies.position,
            ),
        )
        out = out.replace(zombies=frozen_z)
    if rl_cfg.MONSTER_FIGHT_STATIC_SKELETON:
        prev_s = state_in.skeletons
        lock_s = prev_s.mask
        frozen_s = out.skeletons.replace(
            position=jnp.where(
                lock_s[:, None],
                prev_s.position.astype(out.skeletons.position.dtype),
                out.skeletons.position,
            ),
        )
        out = out.replace(skeletons=frozen_s)
    return out.replace(mob_map=rebuild_mob_map_from_state(out))
