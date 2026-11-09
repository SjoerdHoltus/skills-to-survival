#!/usr/bin/env python3
"""Evaluate official Craftax baseline checkpoints (flat / meta, MLP / RNN)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


def _parse_device(argv: list[str]) -> str:
    if "--cpu" in argv:
        return "cpu"
    return "gpu"


def _configure_jax_before_import(device: str, mem_fraction: float | None) -> None:
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        return
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)


_DEVICE = _parse_device(sys.argv)
_MEM_FRACTION: float | None = None
for i, arg in enumerate(sys.argv):
    if arg == "--mem-fraction" and i + 1 < len(sys.argv):
        _MEM_FRACTION = float(sys.argv[i + 1])
    elif arg.startswith("--mem-fraction="):
        _MEM_FRACTION = float(arg.split("=", 1)[1])
_configure_jax_before_import(_DEVICE, _MEM_FRACTION)


import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from configs.survival_meta import SURVIVAL_META_ACTION_LABELS
from ppo.common import load_params
from ppo.configs import ENV_CLASSIC, META_NUM_ACTIONS, resolve_skill_paths
from ppo.meta_ppo import build_skill_bundle
from ppo.skill_step import meta_mode_to_env_action
from training.craftax_classic_metrics import (
    CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS,
)
from training.craftax_classic_metrics import (
    summarize_benchmark_episodes as summarize_classic_benchmark_episodes,
)
from training.craftax_extended_metrics import (
    CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS,
)
from training.craftax_extended_metrics import (
    summarize_benchmark_episodes as summarize_extended_benchmark_episodes,
)
from training.meta_action_mask import survival_meta_greedy_mode

DEFAULT_MAX_STEPS = 1024


def _env_action_labels(env_name: str) -> list[str]:
    if _is_classic_env(env_name):
        from craftax.craftax_classic.constants import Action
    else:
        from craftax.craftax.constants import Action
    return [a.name for a in Action]


def _distribution_payload(counts, labels: list[str], space: str) -> dict:
    counts_i = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts_i.size < len(labels):
        pad = np.zeros(len(labels) - counts_i.size, dtype=np.int64)
        counts_i = np.concatenate([counts_i, pad])
    elif counts_i.size > len(labels):
        counts_i = counts_i[: len(labels)]
    total = int(counts_i.sum())
    fractions = (counts_i / total).tolist() if total else [0.0] * len(labels)
    by_label = {
        labels[i]: {"count": int(counts_i[i]), "fraction": float(fractions[i])}
        for i in range(len(labels))
    }
    return {
        "space": space,
        "labels": list(labels),
        "counts": counts_i.tolist(),
        "fractions": fractions,
        "total_steps": total,
        "by_label": by_label,
    }


def _meta_mode_group_fractions(mode_counts) -> dict[str, float]:
    """Coarse buckets over survival meta modes."""
    c = np.asarray(mode_counts, dtype=np.float64).reshape(-1)
    if c.size < META_NUM_ACTIONS:
        c = np.pad(c, (0, META_NUM_ACTIONS - c.size))
    total = float(c.sum())
    if total <= 0:
        z = 0.0
        return {"noop": z, "skills": z, "craft_place": z, "move_sleep": z}
    return {
        "noop": float(c[0] / total),
        "skills": float(c[1:9].sum() / total),
        "craft_place": float(c[9:19].sum() / total),
        "move_sleep": float(c[19:24].sum() / total),
    }


def _is_classic_env(env_name: str) -> bool:
    return "Classic" in env_name


def _num_achievements(env_name: str) -> int:
    return (
        CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS
        if _is_classic_env(env_name)
        else CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS
    )


def _episode_achievements(state, num_achievements: int) -> np.ndarray:
    return np.asarray(state.achievements[:num_achievements], dtype=np.float32)


def _episode_achievements_jax(state, num_achievements: int):
    return jnp.asarray(state.achievements[:num_achievements], dtype=jnp.float32)


def _make_batched_flat_rollout(
    env, env_params, policy_step, max_steps: int, num_achievements: int, num_actions: int
):
    def one_rollout(key):
        obs, state = env.reset(key, env_params)
        init = (
            obs,
            state,
            key,
            jnp.float32(0.0),
            jnp.int32(0),
            _episode_achievements_jax(state, num_achievements),
            jnp.bool_(False),
            jnp.zeros((num_actions,), dtype=jnp.int32),
        )

        def step_fn(carry, _):
            obs, state, rng, total_reward, length, achievements, done_ever, action_counts = carry
            active = jnp.logical_not(done_ever)
            action = policy_step(obs)
            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, _ = env.step(step_rng, state, action, env_params)
            achievements = jnp.maximum(
                achievements, _episode_achievements_jax(state, num_achievements)
            )
            total_reward = total_reward + reward * active.astype(jnp.float32)
            length = length + active.astype(jnp.int32)
            action_counts = action_counts.at[action].add(active.astype(jnp.int32))
            done_ever = jnp.logical_or(done_ever, done)
            return (
                obs,
                state,
                rng,
                total_reward,
                length,
                achievements,
                done_ever,
                action_counts,
            ), None

        (
            (
                _,
                _,
                _,
                total_reward,
                length,
                achievements,
                done_ever,
                action_counts,
            ),
            _,
        ) = jax.lax.scan(step_fn, init, None, max_steps)
        return (
            total_reward,
            length.astype(jnp.float32),
            done_ever,
            achievements,
            action_counts,
        )

    return jax.jit(jax.vmap(one_rollout))


def _make_batched_meta_rollout(
    env,
    env_params,
    policy_step,
    skill_bundle,
    max_steps: int,
    num_achievements: int,
    num_env_actions: int,
    num_modes: int = META_NUM_ACTIONS,
    skill_horizon: int = 1,
):
    sb = skill_bundle
    horizon = max(1, int(skill_horizon))

    def one_rollout(key):
        obs, state = env.reset(key, env_params)
        init = (
            obs,
            state,
            key,
            jnp.float32(0.0),
            jnp.int32(0),
            _episode_achievements_jax(state, num_achievements),
            jnp.bool_(False),
            jnp.zeros((num_modes,), dtype=jnp.int32),
            jnp.zeros((num_env_actions,), dtype=jnp.int32),
            jnp.int32(0),  # sticky mode
            jnp.int32(0),  # steps left in option (0 => resample)
        )

        def step_fn(carry, _):
            (
                obs,
                state,
                rng,
                total_reward,
                length,
                achievements,
                done_ever,
                mode_counts,
                env_action_counts,
                mode,
                steps_left,
            ) = carry
            active = jnp.logical_not(done_ever)
            need_new = steps_left <= 0
            new_mode = policy_step(obs, state)
            mode = jnp.where(need_new, new_mode, mode)
            steps_left = jnp.where(need_new, jnp.int32(horizon), steps_left)
            mode_counts = mode_counts.at[mode].add((active & need_new).astype(jnp.int32))
            action = meta_mode_to_env_action(
                mode,
                env,
                state,
                wp=sb["wp"],
                dp=sb["dp"],
                fp=sb["fp"],
                mp=sb["mp"],
                sp=sb["sp"],
                op=sb["op"],
                h_wood=sb["h_wood"],
                h_drink=sb["h_drink"],
                h_ff=sb["h_ff"],
                h_mf=sb["h_mf"],
                h_stone=sb["h_stone"],
                h_ore=sb["h_ore"],
            )
            rng, step_rng = jax.random.split(rng)
            obs, state, reward, done, _ = env.step(step_rng, state, action, env_params)
            achievements = jnp.maximum(
                achievements, _episode_achievements_jax(state, num_achievements)
            )
            total_reward = total_reward + reward * active.astype(jnp.float32)
            length = length + active.astype(jnp.int32)
            env_action_counts = env_action_counts.at[action].add(active.astype(jnp.int32))
            done_ever = jnp.logical_or(done_ever, done)
            steps_left = jnp.where(done, jnp.int32(0), steps_left - 1)
            return (
                obs,
                state,
                rng,
                total_reward,
                length,
                achievements,
                done_ever,
                mode_counts,
                env_action_counts,
                mode,
                steps_left,
            ), None

        (
            (
                _,
                _,
                _,
                total_reward,
                length,
                achievements,
                done_ever,
                mode_counts,
                env_action_counts,
                _,
                _,
            ),
            _,
        ) = jax.lax.scan(step_fn, init, None, max_steps)
        return (
            total_reward,
            length.astype(jnp.float32),
            done_ever,
            achievements,
            mode_counts,
            env_action_counts,
        )

    return jax.jit(jax.vmap(one_rollout))


def _make_batched_flat_rnn_rollout(
    env,
    env_params,
    network,
    params,
    layer_size: int,
    max_steps: int,
    num_achievements: int,
    num_actions: int,
):
    from ppo.craftax_baselines.ppo_rnn import ScannedRNN

    def one_rollout(key):
        obs, state = env.reset(key, env_params)
        hstate = ScannedRNN.initialize_carry(1, layer_size)
        done_flag = jnp.zeros((1,), dtype=bool)
        init = (
            obs,
            state,
            key,
            hstate,
            done_flag,
            jnp.float32(0.0),
            jnp.int32(0),
            _episode_achievements_jax(state, num_achievements),
            jnp.bool_(False),
            jnp.zeros((num_actions,), dtype=jnp.int32),
        )

        def step_fn(carry, _):
            (
                obs,
                state,
                rng,
                hstate,
                done_flag,
                total_reward,
                length,
                achievements,
                done_ever,
                action_counts,
            ) = carry
            active = jnp.logical_not(done_ever)
            ac_in = (obs[None, None, :], done_flag[None, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            action = jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0, 0]
            # Match sequential RNN eval RNG: step with rng, then rng = split(rng)[0].
            obs, state, reward, done, _ = env.step(rng, state, action, env_params)
            rng = jax.random.split(rng)[0]
            achievements = jnp.maximum(
                achievements, _episode_achievements_jax(state, num_achievements)
            )
            total_reward = total_reward + reward * active.astype(jnp.float32)
            length = length + active.astype(jnp.int32)
            action_counts = action_counts.at[action].add(active.astype(jnp.int32))
            done_ever = jnp.logical_or(done_ever, done)
            # Match sequential eval: reset carry after terminal step.
            h_reset = ScannedRNN.initialize_carry(1, layer_size)
            hstate = jnp.where(done, h_reset, hstate)
            done_flag = jnp.zeros((1,), dtype=bool)
            return (
                obs,
                state,
                rng,
                hstate,
                done_flag,
                total_reward,
                length,
                achievements,
                done_ever,
                action_counts,
            ), None

        (
            (
                _,
                _,
                _,
                _,
                _,
                total_reward,
                length,
                achievements,
                done_ever,
                action_counts,
            ),
            _,
        ) = jax.lax.scan(step_fn, init, None, max_steps)
        return (
            total_reward,
            length.astype(jnp.float32),
            done_ever,
            achievements,
            action_counts,
        )

    return jax.jit(jax.vmap(one_rollout))


def _make_batched_meta_rnn_rollout(
    env,
    env_params,
    network,
    params,
    layer_size: int,
    skill_bundle,
    max_steps: int,
    num_achievements: int,
    num_env_actions: int,
    *,
    use_action_mask: bool = True,
    num_modes: int = META_NUM_ACTIONS,
):
    from ppo.craftax_baselines.ppo_rnn import ScannedRNN

    sb = skill_bundle

    def one_rollout(key):
        obs, state = env.reset(key, env_params)
        hstate = ScannedRNN.initialize_carry(1, layer_size)
        done_flag = jnp.zeros((1,), dtype=bool)
        init = (
            obs,
            state,
            key,
            hstate,
            done_flag,
            jnp.float32(0.0),
            jnp.int32(0),
            _episode_achievements_jax(state, num_achievements),
            jnp.bool_(False),
            jnp.zeros((num_modes,), dtype=jnp.int32),
            jnp.zeros((num_env_actions,), dtype=jnp.int32),
        )

        def step_fn(carry, _):
            (
                obs,
                state,
                rng,
                hstate,
                done_flag,
                total_reward,
                length,
                achievements,
                done_ever,
                mode_counts,
                env_action_counts,
            ) = carry
            active = jnp.logical_not(done_ever)
            ac_in = (obs[None, None, :], done_flag[None, :])
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            logits = pi.logits[0, 0]
            mode = (
                survival_meta_greedy_mode(logits, state)
                if use_action_mask
                else jnp.argmax(logits).astype(jnp.int32)
            )
            mode_counts = mode_counts.at[mode].add(active.astype(jnp.int32))
            action = meta_mode_to_env_action(
                mode,
                env,
                state,
                wp=sb["wp"],
                dp=sb["dp"],
                fp=sb["fp"],
                mp=sb["mp"],
                sp=sb["sp"],
                op=sb["op"],
                h_wood=sb["h_wood"],
                h_drink=sb["h_drink"],
                h_ff=sb["h_ff"],
                h_mf=sb["h_mf"],
                h_stone=sb["h_stone"],
                h_ore=sb["h_ore"],
            )
            # Match sequential RNN eval RNG: step with rng, then rng = split(rng)[0].
            obs, state, reward, done, _ = env.step(rng, state, action, env_params)
            rng = jax.random.split(rng)[0]
            achievements = jnp.maximum(
                achievements, _episode_achievements_jax(state, num_achievements)
            )
            total_reward = total_reward + reward * active.astype(jnp.float32)
            length = length + active.astype(jnp.int32)
            env_action_counts = env_action_counts.at[action].add(active.astype(jnp.int32))
            done_ever = jnp.logical_or(done_ever, done)
            h_reset = ScannedRNN.initialize_carry(1, layer_size)
            hstate = jnp.where(done, h_reset, hstate)
            done_flag = jnp.zeros((1,), dtype=bool)
            return (
                obs,
                state,
                rng,
                hstate,
                done_flag,
                total_reward,
                length,
                achievements,
                done_ever,
                mode_counts,
                env_action_counts,
            ), None

        (
            (
                _,
                _,
                _,
                _,
                _,
                total_reward,
                length,
                achievements,
                done_ever,
                mode_counts,
                env_action_counts,
            ),
            _,
        ) = jax.lax.scan(step_fn, init, None, max_steps)
        return (
            total_reward,
            length.astype(jnp.float32),
            done_ever,
            achievements,
            mode_counts,
            env_action_counts,
        )

    return jax.jit(jax.vmap(one_rollout))


def _run_batched_in_chunks(batched_rollout, keys, chunk_size: int | None):
    """Run vmapped rollout, optionally in episode chunks to limit peak memory."""
    n = int(keys.shape[0])
    if chunk_size is None or chunk_size <= 0 or n <= int(chunk_size):
        return batched_rollout(keys)
    chunk_size = int(chunk_size)
    parts = []
    for i in range(0, n, chunk_size):
        parts.append(batched_rollout(keys[i : i + chunk_size]))
        print(f"  episodes {min(i + chunk_size, n)}/{n}", flush=True)
    n_out = len(parts[0])
    return tuple(
        jnp.concatenate([parts[p][j] for p in range(len(parts))], axis=0) for j in range(n_out)
    )


def _summarize_benchmark(env_name: str, **kwargs):
    if _is_classic_env(env_name):
        return summarize_classic_benchmark_episodes(**kwargs)
    return summarize_extended_benchmark_episodes(**kwargs)


def _to_device(tree):
    dev = jax.devices()[0]
    return jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), dev), tree)


def _make_policy_step(network, params):
    @jax.jit
    def policy_step(obs):
        pi, _ = network.apply(params, obs[None, :])
        return jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0]

    return policy_step


def _make_meta_policy_step(network, params, *, use_action_mask: bool = True):
    @jax.jit
    def policy_step(obs, state):
        pi, _ = network.apply(params, obs[None, :])
        logits = pi.logits[0]
        if use_action_mask:
            return survival_meta_greedy_mode(logits, state)
        return jnp.argmax(logits).astype(jnp.int32)

    return policy_step


def _resolve_eval_action_mask(mode: str, hyp: dict, run_dir: str) -> bool:
    """Match train mask by default; CLI can force on/off."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    if "use_action_mask" in hyp:
        return bool(hyp["use_action_mask"])
    # Fallback for older runs / path naming.
    return "no_mask" not in os.path.abspath(run_dir)


def _flat_rollout(
    env, env_params, policy_step, rng, max_steps: int, num_achievements: int, num_actions: int
):
    obs, state = env.reset(rng, env_params)
    total_reward = 0.0
    length = 0
    achievements = np.zeros(num_achievements, dtype=np.float32)
    action_counts = np.zeros(num_actions, dtype=np.int64)
    died = False
    for _ in range(max_steps):
        action = policy_step(obs)
        action_i = int(action)
        action_counts[action_i] += 1
        obs, state, reward, done, _ = env.step(rng, state, action, env_params)
        rng = jax.random.split(rng)[0]
        total_reward += float(reward)
        length += 1
        achievements = np.maximum(achievements, _episode_achievements(state, num_achievements))
        if done:
            died = True
            break
    return total_reward, length, died, achievements, action_counts


def _meta_rollout(
    env,
    env_params,
    policy_step,
    skill_bundle,
    rng,
    max_steps: int,
    num_achievements: int,
    num_env_actions: int,
    num_modes: int = META_NUM_ACTIONS,
    skill_horizon: int = 1,
):
    obs, state = env.reset(rng, env_params)
    total_reward = 0.0
    length = 0
    achievements = np.zeros(num_achievements, dtype=np.float32)
    mode_counts = np.zeros(num_modes, dtype=np.int64)
    env_action_counts = np.zeros(num_env_actions, dtype=np.int64)
    died = False
    sb = skill_bundle
    horizon = max(1, int(skill_horizon))
    while length < max_steps:
        mode = policy_step(obs, state)
        mode_i = int(mode)
        mode_counts[mode_i] += 1
        for _ in range(horizon):
            if length >= max_steps:
                break
            action = meta_mode_to_env_action(
                mode,
                env,
                state,
                wp=sb["wp"],
                dp=sb["dp"],
                fp=sb["fp"],
                mp=sb["mp"],
                sp=sb["sp"],
                op=sb["op"],
                h_wood=sb["h_wood"],
                h_drink=sb["h_drink"],
                h_ff=sb["h_ff"],
                h_mf=sb["h_mf"],
                h_stone=sb["h_stone"],
                h_ore=sb["h_ore"],
            )
            env_action_counts[int(action)] += 1
            obs, state, reward, done, _ = env.step(rng, state, action, env_params)
            rng = jax.random.split(rng)[0]
            total_reward += float(reward)
            length += 1
            achievements = np.maximum(achievements, _episode_achievements(state, num_achievements))
            if done:
                died = True
                break
        if died:
            break
    return total_reward, length, died, achievements, mode_counts, env_action_counts


def _rnn_rollout(
    env,
    env_params,
    network,
    params,
    layer_size,
    rng,
    max_steps: int,
    num_achievements: int,
    num_actions: int,
):
    from ppo.craftax_baselines.ppo_rnn import ScannedRNN

    obs, state = env.reset(rng, env_params)
    hstate = ScannedRNN.initialize_carry(1, layer_size)
    done_flag = jnp.zeros((1,), dtype=bool)
    total_reward = 0.0
    length = 0
    achievements = np.zeros(num_achievements, dtype=np.float32)
    action_counts = np.zeros(num_actions, dtype=np.int64)
    died = False
    for _ in range(max_steps):
        ac_in = (obs[None, None, :], done_flag[None, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        action = jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0, 0]
        action_counts[int(action)] += 1
        obs, state, reward, done, _ = env.step(rng, state, action, env_params)
        rng = jax.random.split(rng)[0]
        done_flag = jnp.array([done], dtype=bool)
        if done:
            hstate = ScannedRNN.initialize_carry(1, layer_size)
            done_flag = jnp.zeros((1,), dtype=bool)
        total_reward += float(reward)
        length += 1
        achievements = np.maximum(achievements, _episode_achievements(state, num_achievements))
        if done:
            died = True
            break
    return total_reward, length, died, achievements, action_counts


def _meta_rnn_rollout(
    env,
    env_params,
    network,
    params,
    layer_size,
    skill_bundle,
    rng,
    max_steps: int,
    num_achievements: int,
    *,
    use_action_mask: bool = True,
    num_env_actions: int,
    num_modes: int = META_NUM_ACTIONS,
):
    from ppo.craftax_baselines.ppo_rnn import ScannedRNN

    obs, state = env.reset(rng, env_params)
    hstate = ScannedRNN.initialize_carry(1, layer_size)
    done_flag = jnp.zeros((1,), dtype=bool)
    total_reward = 0.0
    length = 0
    achievements = np.zeros(num_achievements, dtype=np.float32)
    mode_counts = np.zeros(num_modes, dtype=np.int64)
    env_action_counts = np.zeros(num_env_actions, dtype=np.int64)
    died = False
    sb = skill_bundle
    for _ in range(max_steps):
        ac_in = (obs[None, None, :], done_flag[None, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        logits = pi.logits[0, 0]
        mode = (
            survival_meta_greedy_mode(logits, state)
            if use_action_mask
            else jnp.argmax(logits).astype(jnp.int32)
        )
        mode_counts[int(mode)] += 1
        action = meta_mode_to_env_action(
            mode,
            env,
            state,
            wp=sb["wp"],
            dp=sb["dp"],
            fp=sb["fp"],
            mp=sb["mp"],
            sp=sb["sp"],
            op=sb["op"],
            h_wood=sb["h_wood"],
            h_drink=sb["h_drink"],
            h_ff=sb["h_ff"],
            h_mf=sb["h_mf"],
            h_stone=sb["h_stone"],
            h_ore=sb["h_ore"],
        )
        env_action_counts[int(action)] += 1
        obs, state, reward, done, _ = env.step(rng, state, action, env_params)
        rng = jax.random.split(rng)[0]
        done_flag = jnp.array([done], dtype=bool)
        if done:
            hstate = ScannedRNN.initialize_carry(1, layer_size)
            done_flag = jnp.zeros((1,), dtype=bool)
        total_reward += float(reward)
        length += 1
        achievements = np.maximum(achievements, _episode_achievements(state, num_achievements))
        if done:
            died = True
            break
    return total_reward, length, died, achievements, mode_counts, env_action_counts


def main() -> None:
    from ppo.craftax_baselines.models.actor_critic import ActorCritic

    parser = argparse.ArgumentParser(description="Evaluate official baseline checkpoints")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--policy", choices=("flat", "meta", "flat_rnn", "meta_rnn"), required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--layer-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=None,
        help="Optional cap on JAX GPU memory fraction (e.g. 0.85)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="JIT scan+vmap rollouts (flat/meta/rnn; much faster than sequential)",
    )
    parser.add_argument(
        "--fast-chunk",
        type=int,
        default=None,
        help="Episode chunk size for fast eval (default: 32 for rnn, all for mlp)",
    )
    parser.add_argument(
        "--eval-action-mask",
        choices=("auto", "on", "off"),
        default="auto",
        help="Meta only: auto=match training use_action_mask; on/off force masked/unmasked greedy",
    )
    args = parser.parse_args()

    print(f"Eval device: {jax.devices()[0]}", flush=True)
    print(
        "JAX memory: "
        f"preallocate={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default')}, "
        f"allocator={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'default')}",
        flush=True,
    )

    params = _to_device(load_params(args.run_dir))
    cfg_path = os.path.join(args.run_dir, "config.json")
    run_cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            run_cfg = json.load(f)
    hyp = run_cfg.get("hyperparameters") or {}
    layer_size = int(args.layer_size or hyp.get("layer_size", 512))
    env_name = hyp.get("env_name", ENV_CLASSIC)
    eval_use_action_mask = None
    skill_horizon = max(1, int(hyp.get("skill_horizon", 1)))
    if args.policy in ("meta", "meta_rnn"):
        eval_use_action_mask = _resolve_eval_action_mask(args.eval_action_mask, hyp, args.run_dir)
        print(
            f"Meta eval action mask: {eval_use_action_mask} "
            f"(mode={args.eval_action_mask}), skill_horizon={skill_horizon}",
            flush=True,
        )

    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    num_achievements = _num_achievements(env_name)
    env_action_labels = _env_action_labels(env_name)
    num_env_actions = len(env_action_labels)
    policy_action_distribution = None
    meta_mode_distribution = None
    env_action_distribution = None
    meta_mode_group_fractions = None

    if args.policy == "flat":
        network = ActorCritic(num_env_actions, layer_size)
        policy_step = _make_policy_step(network, params)
        rollout = lambda key: _flat_rollout(
            env,
            env_params,
            policy_step,
            key,
            args.max_steps,
            num_achievements,
            num_env_actions,
        )
    elif args.policy == "meta":
        network = ActorCritic(META_NUM_ACTIONS, layer_size)
        policy_step = _make_meta_policy_step(
            network, params, use_action_mask=bool(eval_use_action_mask)
        )
        skill_bundle = _to_device(
            build_skill_bundle(resolve_skill_paths(run_cfg.get("skill_weights")))
        )
        rollout = lambda key: _meta_rollout(
            env,
            env_params,
            policy_step,
            skill_bundle,
            key,
            args.max_steps,
            num_achievements,
            num_env_actions,
            skill_horizon=skill_horizon,
        )
    elif args.policy == "flat_rnn":
        from ppo.craftax_baselines.ppo_rnn import ActorCriticRNN

        network = ActorCriticRNN(num_env_actions, config={"LAYER_SIZE": layer_size})
        rollout = lambda key: _rnn_rollout(
            env,
            env_params,
            network,
            params,
            layer_size,
            key,
            args.max_steps,
            num_achievements,
            num_env_actions,
        )
    else:
        from ppo.craftax_baselines.ppo_rnn import ActorCriticRNN

        network = ActorCriticRNN(META_NUM_ACTIONS, config={"LAYER_SIZE": layer_size})
        skill_bundle = _to_device(
            build_skill_bundle(resolve_skill_paths(run_cfg.get("skill_weights")))
        )
        rollout = lambda key: _meta_rnn_rollout(
            env,
            env_params,
            network,
            params,
            layer_size,
            skill_bundle,
            key,
            args.max_steps,
            num_achievements,
            use_action_mask=bool(eval_use_action_mask),
            num_env_actions=num_env_actions,
        )

    keys = jax.random.split(jax.random.PRNGKey(args.master_seed), args.num_episodes)
    dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
    if args.policy == "flat":
        jax.block_until_ready(policy_step(dummy_obs))
    elif args.policy == "meta":
        dummy_rng, dummy_reset = jax.random.split(jax.random.PRNGKey(0))
        _, dummy_state = env.reset(dummy_reset, env_params)
        jax.block_until_ready(policy_step(dummy_obs, dummy_state))
    elif args.policy in ("flat_rnn", "meta_rnn"):
        from ppo.craftax_baselines.ppo_rnn import ScannedRNN

        dummy_h = ScannedRNN.initialize_carry(1, layer_size)
        dummy_done = jnp.zeros((1, 1), dtype=bool)
        jax.block_until_ready(
            network.apply(params, dummy_h, (dummy_obs[None, None, :], dummy_done))
        )

    use_fast = args.fast and args.policy in ("flat", "meta", "flat_rnn", "meta_rnn")
    if use_fast:
        fast_chunk = args.fast_chunk
        if fast_chunk is None and args.policy in ("flat_rnn", "meta_rnn"):
            fast_chunk = 32
        chunk_note = f", chunk={fast_chunk}" if fast_chunk else ""
        print(
            f"Fast eval: {args.num_episodes} episodes x {args.max_steps} steps "
            f"(scan+vmap{chunk_note})",
            flush=True,
        )
        t0 = time.time()
        if args.policy == "flat":
            batched_rollout = _make_batched_flat_rollout(
                env,
                env_params,
                policy_step,
                args.max_steps,
                num_achievements,
                num_env_actions,
            )
            rets, lens, died_arr, ach, act_counts = _run_batched_in_chunks(
                batched_rollout, keys, fast_chunk
            )
            jax.block_until_ready(rets)
            action_counts_sum = np.asarray(act_counts, dtype=np.int64).sum(axis=0)
            policy_action_distribution = _distribution_payload(
                action_counts_sum, env_action_labels, "env_action"
            )
        elif args.policy == "meta":
            batched_rollout = _make_batched_meta_rollout(
                env,
                env_params,
                policy_step,
                skill_bundle,
                args.max_steps,
                num_achievements,
                num_env_actions,
                skill_horizon=skill_horizon,
            )
            rets, lens, died_arr, ach, mode_counts, env_act_counts = _run_batched_in_chunks(
                batched_rollout, keys, fast_chunk
            )
            jax.block_until_ready(rets)
            mode_counts_sum = np.asarray(mode_counts, dtype=np.int64).sum(axis=0)
            env_action_counts_sum = np.asarray(env_act_counts, dtype=np.int64).sum(axis=0)
            meta_mode_distribution = _distribution_payload(
                mode_counts_sum, list(SURVIVAL_META_ACTION_LABELS), "meta_mode"
            )
            env_action_distribution = _distribution_payload(
                env_action_counts_sum, env_action_labels, "env_action"
            )
            meta_mode_group_fractions = _meta_mode_group_fractions(mode_counts_sum)
            policy_action_distribution = meta_mode_distribution
        elif args.policy == "flat_rnn":
            batched_rollout = _make_batched_flat_rnn_rollout(
                env,
                env_params,
                network,
                params,
                layer_size,
                args.max_steps,
                num_achievements,
                num_env_actions,
            )
            rets, lens, died_arr, ach, act_counts = _run_batched_in_chunks(
                batched_rollout, keys, fast_chunk
            )
            jax.block_until_ready(rets)
            action_counts_sum = np.asarray(act_counts, dtype=np.int64).sum(axis=0)
            policy_action_distribution = _distribution_payload(
                action_counts_sum, env_action_labels, "env_action"
            )
        else:
            batched_rollout = _make_batched_meta_rnn_rollout(
                env,
                env_params,
                network,
                params,
                layer_size,
                skill_bundle,
                args.max_steps,
                num_achievements,
                num_env_actions,
                use_action_mask=bool(eval_use_action_mask),
            )
            rets, lens, died_arr, ach, mode_counts, env_act_counts = _run_batched_in_chunks(
                batched_rollout, keys, fast_chunk
            )
            jax.block_until_ready(rets)
            mode_counts_sum = np.asarray(mode_counts, dtype=np.int64).sum(axis=0)
            env_action_counts_sum = np.asarray(env_act_counts, dtype=np.int64).sum(axis=0)
            meta_mode_distribution = _distribution_payload(
                mode_counts_sum, list(SURVIVAL_META_ACTION_LABELS), "meta_mode"
            )
            env_action_distribution = _distribution_payload(
                env_action_counts_sum, env_action_labels, "env_action"
            )
            meta_mode_group_fractions = _meta_mode_group_fractions(mode_counts_sum)
            policy_action_distribution = meta_mode_distribution
        elapsed = time.time() - t0
        returns = np.asarray(rets, dtype=np.float32)
        lengths = np.asarray(lens, dtype=np.float32)
        died = np.asarray(died_arr, dtype=bool)
        achievements = np.asarray(ach, dtype=np.float32)
    else:
        t0 = time.time()
        results = []
        for i, key in enumerate(keys):
            results.append(rollout(key))
            if (i + 1) % max(1, args.num_episodes // 8) == 0 or i + 1 == args.num_episodes:
                print(f"  episodes {i + 1}/{args.num_episodes}", flush=True)
        elapsed = time.time() - t0
        returns = np.array([r[0] for r in results], dtype=np.float32)
        lengths = np.array([r[1] for r in results], dtype=np.float32)
        died = np.array([r[2] for r in results], dtype=bool)
        achievements = np.stack([r[3] for r in results], axis=0)
        if args.policy in ("flat", "flat_rnn"):
            action_counts_sum = np.sum([r[4] for r in results], axis=0)
            policy_action_distribution = _distribution_payload(
                action_counts_sum, env_action_labels, "env_action"
            )
        else:
            mode_counts_sum = np.sum([r[4] for r in results], axis=0)
            env_action_counts_sum = np.sum([r[5] for r in results], axis=0)
            meta_mode_distribution = _distribution_payload(
                mode_counts_sum, list(SURVIVAL_META_ACTION_LABELS), "meta_mode"
            )
            env_action_distribution = _distribution_payload(
                env_action_counts_sum, env_action_labels, "env_action"
            )
            meta_mode_group_fractions = _meta_mode_group_fractions(mode_counts_sum)
            policy_action_distribution = meta_mode_distribution

    summary = _summarize_benchmark(
        env_name,
        episode_returns=returns,
        episode_lengths=lengths,
        achievement_matrix=achievements,
        died=died,
    )
    summary.update(
        {
            "policy": args.policy,
            "env_name": env_name,
            "run_dir": os.path.abspath(args.run_dir),
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "layer_size": layer_size,
            "eval_device": str(jax.devices()[0]),
            "eval_elapsed_sec": elapsed,
            "eval_mode": "fast_scan_vmap" if use_fast else "sequential",
            "eval_use_action_mask": eval_use_action_mask,
            "skill_horizon": skill_horizon if args.policy in ("meta", "meta_rnn") else 1,
            "master_seed": int(args.master_seed),
            "policy_action_distribution": policy_action_distribution,
            "meta_mode_distribution": meta_mode_distribution,
            "env_action_distribution": env_action_distribution,
            "meta_mode_group_fractions": meta_mode_group_fractions,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    out_path = args.output or os.path.join(
        args.run_dir,
        "evals",
        f"benchmark_official_{args.policy}_{datetime.now(timezone.utc).strftime('%d%m%Y_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
