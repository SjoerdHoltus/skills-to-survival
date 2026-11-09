"""
Render a GIF for an evolved monster-fighting policy (narrow 5-action head).

Matches training when the run's ``config.json`` is applied: Craftax-Classic-Symbolic
for logits, Craftax-Classic-Pixels for frames; mob spawns disabled in env params;
``bootstrap_monster_fight_state`` after reset (variable zombies/skeletons/trees
from configs); greedy argmax -> env action = local + 1.

By default, if weights live under ``.../run_xxx/weights/``, eval loads
``run_xxx/config.json`` and applies mob counts, FOV, static flags, rollout horizon,
kill reward, policy temperature, and hidden size so the GIF matches that training run.
Use ``--no-run-config`` to keep using your current ``config.py`` only.

CPU JAX only (see JAX_PLATFORMS below).

Usage:
  python -m evaluation.eval_monster_fight runs/monster_fight/run_xxx/weights/final_....npy --seed 0

The run directory (and eval output location) is inferred from the weight
path: weights living under ``.../run_xxx/weights/`` save GIFs to
``.../run_xxx/evals/``. Weights elsewhere fall back to ``eval_weights/<stem>/``.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import craftax  # noqa: F401
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name

import configs as mf_cfg
from configs import (
    FIND_WOOD_NUM_ACTIONS,
    MONSTER_FIGHT_HIDDEN_SIZE,
    MONSTER_FIGHT_OBS_SIZE,
    MONSTER_FIGHT_POLICY_TEMPERATURE,
    MONSTER_FIGHT_ROLLOUT_STEPS,
    OBS_SIZE,
)
from policies.monster_fight import MonsterFightPolicy
from training.monster_world_bootstrap import (
    bootstrap_monster_fight_state,
    make_monster_fight_obs,
    rebuild_mob_map_from_state,
)
from training.params import flat_params_to_pytree

# Fields overwritten when applying a training run's ``config.json`` (restore between jobs).
_MONSTER_FIGHT_CFG_KEYS = (
    "MONSTER_FIGHT_NUM_ZOMBIES_MIN",
    "MONSTER_FIGHT_NUM_ZOMBIES_MAX",
    "MONSTER_FIGHT_NUM_SKELETONS_MAX",
    "MONSTER_FIGHT_NUM_TREES_MIN",
    "MONSTER_FIGHT_NUM_TREES_MAX",
    "MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST",
    "MONSTER_FIGHT_FOV_CHEBYSHEV_R",
    "MONSTER_FIGHT_STATIC_ZOMBIE",
    "MONSTER_FIGHT_STATIC_SKELETON",
    "MONSTER_FIGHT_ROLLOUT_STEPS",
    "MONSTER_FIGHT_KILL_REWARD",
    "MONSTER_FIGHT_POLICY_TEMPERATURE",
    "MONSTER_FIGHT_HIDDEN_SIZE",
    "MONSTER_FIGHT_PLAYER_MAX_HEALTH",
)


def _snapshot_monster_fight_config() -> dict:
    return {k: getattr(mf_cfg, k) for k in _MONSTER_FIGHT_CFG_KEYS}


def _restore_monster_fight_config(snap: dict) -> None:
    for k, v in snap.items():
        setattr(mf_cfg, k, v)


def apply_monster_fight_run_config_dict(run_cfg: dict) -> dict[str, object]:
    """Apply ``hyperparameters`` / ``policy`` from a training ``config.json`` to ``mf_cfg``."""
    h = run_cfg.get("hyperparameters")
    policy = run_cfg.get("policy") or {}
    if not isinstance(h, dict):
        return {}
    applied: dict[str, object] = {}
    if "num_zombies" in h and "num_zombies_min" not in h:
        nz = int(h["num_zombies"])
        mf_cfg.MONSTER_FIGHT_NUM_ZOMBIES_MIN = nz
        mf_cfg.MONSTER_FIGHT_NUM_ZOMBIES_MAX = nz
        applied["num_zombies (legacy)"] = nz
    if "num_trees" in h and "num_trees_min" not in h and "num_trees_max" not in h:
        nt = int(h["num_trees"])
        mf_cfg.MONSTER_FIGHT_NUM_TREES_MIN = nt
        mf_cfg.MONSTER_FIGHT_NUM_TREES_MAX = nt
        applied["num_trees (legacy)"] = nt
    for json_key, attr in (
        ("num_zombies_min", "MONSTER_FIGHT_NUM_ZOMBIES_MIN"),
        ("num_zombies_max", "MONSTER_FIGHT_NUM_ZOMBIES_MAX"),
        ("num_skeletons_max", "MONSTER_FIGHT_NUM_SKELETONS_MAX"),
        ("num_trees_min", "MONSTER_FIGHT_NUM_TREES_MIN"),
        ("num_trees_max", "MONSTER_FIGHT_NUM_TREES_MAX"),
        ("spawn_fov_min_chebyshev_dist", "MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST"),
        ("spawn_fov_chebyshev_r", "MONSTER_FIGHT_FOV_CHEBYSHEV_R"),
        ("rollout_steps", "MONSTER_FIGHT_ROLLOUT_STEPS"),
    ):
        if json_key in h:
            v = int(h[json_key])
            setattr(mf_cfg, attr, v)
            applied[json_key] = v
    for json_key, attr in (
        ("kill_reward", "MONSTER_FIGHT_KILL_REWARD"),
        ("policy_temperature", "MONSTER_FIGHT_POLICY_TEMPERATURE"),
    ):
        if json_key in h:
            v = float(h[json_key])
            setattr(mf_cfg, attr, v)
            applied[json_key] = v
    for json_key, attr in (
        ("static_zombie", "MONSTER_FIGHT_STATIC_ZOMBIE"),
        ("static_skeleton", "MONSTER_FIGHT_STATIC_SKELETON"),
    ):
        if json_key in h:
            v = bool(h[json_key])
            setattr(mf_cfg, attr, v)
            applied[json_key] = v
    hid = None
    if "hidden_size" in h:
        hid = int(h["hidden_size"])
    elif isinstance(policy, dict) and "hidden_size" in policy:
        hid = int(policy["hidden_size"])
    if hid is not None:
        mf_cfg.MONSTER_FIGHT_HIDDEN_SIZE = hid
        applied["hidden_size"] = hid
    return applied


def _monster_fight_env_params(env):
    """Mirror training/fitness_monster_fight.make_env: zero spawn chances and
    push the mob despawn distance way out so a zombie can never silently
    flip its mask via despawn (which would falsely register as a kill)."""
    p = env.default_params
    return p.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
        mob_despawn_distance=10_000,
    )


def load_params(path, hidden_size=None, num_actions=None):
    flat_params = jnp.load(path)
    h = hidden_size if hidden_size is not None else mf_cfg.MONSTER_FIGHT_HIDDEN_SIZE
    n = num_actions if num_actions is not None else FIND_WOOD_NUM_ACTIONS
    policy = MonsterFightPolicy(hidden_size=h, num_actions=n)
    # Pick the obs size by total parameter count: a flat checkpoint trained on
    # the augmented obs has a different first-layer kernel size than one
    # trained on the raw obs, so we autodetect to stay backward-compatible.
    aug_init = policy.init(jax.random.PRNGKey(0), jnp.zeros(MONSTER_FIGHT_OBS_SIZE))["params"]
    aug_n = sum(int(np.prod(l.shape)) for l in jax.tree_util.tree_leaves(aug_init))
    raw_init = policy.init(jax.random.PRNGKey(0), jnp.zeros(OBS_SIZE))["params"]
    raw_n = sum(int(np.prod(l.shape)) for l in jax.tree_util.tree_leaves(raw_init))
    flat_n = int(flat_params.shape[0])
    if flat_n == aug_n:
        init_params = aug_init
    elif flat_n == raw_n:
        init_params = raw_init
    else:
        raise ValueError(
            f"Checkpoint has {flat_n} params; expected {aug_n} (augmented obs) "
            f"or {raw_n} (raw obs)."
        )

    leaves, tree_def = jax.tree_util.tree_flatten(init_params)
    shapes = [l.shape for l in leaves]

    return flat_params_to_pytree(flat_params, shapes, tree_def)


def pick_action(
    params,
    obs,
    hidden_size=None,
    num_actions=None,
    *,
    rng=None,
    temperature=1.0,
):
    """Greedy argmax by default. If ``rng`` is provided, sample a categorical
    action with ``logits / temperature`` (matches training-time sampling)."""
    h = hidden_size if hidden_size is not None else mf_cfg.MONSTER_FIGHT_HIDDEN_SIZE
    n = num_actions if num_actions is not None else FIND_WOOD_NUM_ACTIONS
    logits = MonsterFightPolicy(hidden_size=h, num_actions=n).apply({"params": params}, obs)
    if rng is None:
        local_action = jnp.argmax(logits)
    else:
        scaled = logits / jnp.float32(temperature)
        local_action = jax.random.categorical(rng, scaled)
    return int(local_action + 1)


def _sync_pixel_from_symbolic(sym_state, pix_state):
    """Keep pixel env aligned with symbolic after bootstrap/step.

    This includes terrain map syncing so curriculum stages that rewrite the map
    (e.g. flat-grass stage) render correctly in pixel observations.
    """
    return pix_state.replace(
        map=sym_state.map,
        mob_map=sym_state.mob_map,
        zombies=sym_state.zombies,
        skeletons=sym_state.skeletons,
        cows=sym_state.cows,
        arrows=sym_state.arrows,
        inventory=sym_state.inventory,
    )


def print_achievements(achievements):
    unlocked = [Achievement(i).name for i in range(len(Achievement)) if bool(achievements[i])]
    if unlocked:
        print(f"Achievements unlocked ({len(unlocked)}/{len(Achievement)}): {', '.join(unlocked)}")
    else:
        print("Achievements unlocked: none")


def rollout(
    params,
    rollout_steps: int | None = None,
    seed=0,
    hidden_size=None,
    num_actions=None,
    rng_key_uint32=None,
    stochastic=False,
    temperature=1.0,
):
    if rollout_steps is None:
        rollout_steps = int(mf_cfg.MONSTER_FIGHT_ROLLOUT_STEPS)
    sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    pix_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    sym_params = _monster_fight_env_params(sym_env)
    pix_params = _monster_fight_env_params(pix_env)

    if rng_key_uint32 is not None:
        a = int(rng_key_uint32[0]) % (1 << 32)
        b = int(rng_key_uint32[1]) % (1 << 32)
        rng = jnp.asarray([jnp.uint32(a), jnp.uint32(b)], dtype=jnp.uint32)
    else:
        rng = jax.random.PRNGKey(seed)

    rng, reset_rng, boot_rng = jax.random.split(rng, 3)
    sym_obs, sym_state = sym_env.reset(reset_rng, sym_params)
    pix_obs, pix_state = pix_env.reset(reset_rng, pix_params)

    sym_state = bootstrap_monster_fight_state(sym_state, boot_rng, sym_params)
    pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
    sym_obs = make_monster_fight_obs(sym_env, sym_state)
    pix_obs = pix_env.get_obs(pix_state)

    frames = []
    episode_done = False
    initial_alive_z = int(jax.device_get(sym_state.zombies.mask.sum()))
    initial_alive_s = int(jax.device_get(sym_state.skeletons.mask.sum()))
    initial_alive = initial_alive_z + initial_alive_s
    kills = 0
    touched_success = 0.0
    shaped_score = 0.0

    for _step in range(rollout_steps):
        frame = np.asarray(pix_obs)
        frames.append((frame * 255).astype(np.uint8))

        prev_z = sym_state.zombies
        prev_s = sym_state.skeletons

        if stochastic:
            rng, policy_rng = jax.random.split(rng)
            action = pick_action(
                params,
                sym_obs,
                hidden_size,
                num_actions,
                rng=policy_rng,
                temperature=temperature,
            )
        else:
            action = pick_action(params, sym_obs, hidden_size, num_actions)

        rng, step_rng = jax.random.split(rng)
        sym_obs, sym_state, _reward, done, _info = sym_env.step(
            step_rng, sym_state, action, sym_params
        )
        freeze_z = mf_cfg.MONSTER_FIGHT_STATIC_ZOMBIE
        freeze_s = mf_cfg.MONSTER_FIGHT_STATIC_SKELETON
        if freeze_z or freeze_s:
            post_z = sym_state.zombies
            post_s = sym_state.skeletons
            if freeze_z:
                lock_z = prev_z.mask
                frozen_z = post_z.replace(
                    position=jnp.where(
                        lock_z[:, None],
                        prev_z.position.astype(post_z.position.dtype),
                        post_z.position,
                    ),
                )
            else:
                frozen_z = post_z
            if freeze_s:
                lock_s = prev_s.mask
                frozen_s = post_s.replace(
                    position=jnp.where(
                        lock_s[:, None],
                        prev_s.position.astype(post_s.position.dtype),
                        post_s.position,
                    ),
                )
            else:
                frozen_s = post_s
            sym_state = sym_state.replace(zombies=frozen_z, skeletons=frozen_s)
            sym_state = sym_state.replace(mob_map=rebuild_mob_map_from_state(sym_state))
        sym_obs = make_monster_fight_obs(sym_env, sym_state)
        pix_obs, pix_state, _, _, _ = pix_env.step(step_rng, pix_state, action, pix_params)
        if freeze_z or freeze_s:
            pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
            pix_obs = pix_env.get_obs(pix_state)

        # Tally kills this step (zombies + skeletons). Require both
        # (a) mask flipped True->False and (b) health dropped to <= 0 —
        # same robust definition as the training fitness, so a despawn
        # never gets counted as a kill.
        prev_zmask = np.asarray(prev_z.mask)
        new_zmask = np.asarray(sym_state.zombies.mask)
        new_zhealth = np.asarray(sym_state.zombies.health)
        z_killed_mask = prev_zmask & (~new_zmask) & (new_zhealth <= 0)
        prev_smask = np.asarray(prev_s.mask)
        new_smask = np.asarray(sym_state.skeletons.mask)
        new_shealth = np.asarray(sym_state.skeletons.health)
        s_killed_mask = prev_smask & (~new_smask) & (new_shealth <= 0)
        kills_this_step = int(z_killed_mask.sum() + s_killed_mask.sum())
        kills += kills_this_step
        shaped_score += float(mf_cfg.MONSTER_FIGHT_KILL_REWARD) * kills_this_step

        any_z = bool(jax.device_get(sym_state.zombies.mask.any()))
        any_s = bool(jax.device_get(sym_state.skeletons.mask.any()))
        any_hostile = any_z or any_s
        if any_hostile:
            p = sym_state.player_position.astype(jnp.int32)
            zd = jnp.abs(sym_state.zombies.position.astype(jnp.int32) - p).sum(axis=-1)
            zd = jnp.where(sym_state.zombies.mask, zd, jnp.asarray(1_000_000, dtype=jnp.int32))
            sd = jnp.abs(sym_state.skeletons.position.astype(jnp.int32) - p).sum(axis=-1)
            sd = jnp.where(sym_state.skeletons.mask, sd, jnp.asarray(1_000_000, dtype=jnp.int32))
            min_d = int(jax.device_get(jnp.minimum(zd.min(), sd.min())))
            touched = 1.0 if min_d <= 1 else 0.0
        else:
            touched = 0.0
        touched_success = max(touched_success, touched)

        episode_done = episode_done or bool(jax.device_get(done))
        # End the rollout (and clip the GIF) once every spawned hostile is
        # dead — this is the new task-success criterion.
        if not any_hostile:
            episode_done = True
        if episode_done:
            break

    achievements = np.asarray(sym_state.achievements)
    kill_rate = kills / max(initial_alive, 1)
    print(
        f"Rollout finished: {len(frames)} steps, shaped_score={shaped_score:.2f}, "
        f"kills={kills}/{initial_alive} (kill_rate={kill_rate:.2f}), touch={touched_success:.1f}"
    )
    print_achievements(achievements)

    return frames, shaped_score, kill_rate, achievements


def upscale_frames(frames, scale=8):
    return [np.repeat(np.repeat(f, scale, axis=0), scale, axis=1) for f in frames]


def save_gif(frames, path, fps=10):
    iio.imwrite(path, frames, loop=0, duration=int(1000 / fps))
    print(f"Saved GIF ({len(frames)} frames, {fps} fps) to {path}")


def infer_run_dir_from_weight_path(weight_path):
    abs_path = os.path.abspath(weight_path)
    if not os.path.isfile(abs_path):
        return None
    weights_parent = os.path.dirname(abs_path)
    if os.path.basename(weights_parent) != "weights":
        return None
    run_dir = os.path.dirname(weights_parent)
    if not os.path.isdir(run_dir):
        return None
    return run_dir


def evals_dir_and_basename_for_weight(weight_path):
    abs_path = os.path.abspath(weight_path)
    label_stem = os.path.splitext(os.path.basename(abs_path))[0]
    run_dir = infer_run_dir_from_weight_path(abs_path)
    if run_dir:
        evals_dir = os.path.join(run_dir, "evals")
        run_basename = os.path.basename(run_dir.rstrip(os.sep))
        return evals_dir, run_basename, label_stem
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evals_dir = os.path.join(project_root, "eval_weights", label_stem)
    return evals_dir, label_stem, label_stem


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate monster_fight policy on Craftax-Classic with pixel GIF (CPU JAX)"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        help="Path(s) to .npy MonsterFightPolicy weights (run dir inferred from path).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Rollout horizon (default: run config's rollout_steps if loaded, else "
            f"config MONSTER_FIGHT_ROLLOUT_STEPS={MONSTER_FIGHT_ROLLOUT_STEPS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rng-key",
        nargs=2,
        type=int,
        metavar=("U32_0", "U32_1"),
        default=None,
        help="Two uint32s for JAX PRNG replay (overrides --seed).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help=f"Hidden width (default {MONSTER_FIGHT_HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=None,
        help=f"Policy head size (default {FIND_WOOD_NUM_ACTIONS})",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Sample actions from the categorical policy (matches training) "
            "instead of taking argmax. Useful to distinguish 'the trained "
            "policy is passive' from 'argmax at eval time collapses onto DO'."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Softmax temperature for --stochastic (default: run config policy_temperature "
            f"if loaded, else {MONSTER_FIGHT_POLICY_TEMPERATURE})"
        ),
    )
    parser.add_argument(
        "--run-config",
        type=str,
        default=None,
        metavar="PATH",
        help=("Explicit path to a training run config.json (overrides auto-discovery)."),
    )
    parser.add_argument(
        "--no-run-config",
        action="store_true",
        help="Do not load run_dir/config.json; use current config.py only.",
    )
    args = parser.parse_args()

    for wp in args.weights:
        if not os.path.isfile(wp):
            parser.error(f"Weight file not found: {wp}")

    jobs = []
    for wp in args.weights:
        abs_wp = os.path.abspath(wp)
        evals_dir, run_basename, label_stem = evals_dir_and_basename_for_weight(abs_wp)
        jobs.append((abs_wp, evals_dir, run_basename, label_stem))

    snapshot = _snapshot_monster_fight_config()

    for weight_path, evals_dir, run_basename, label_stem in jobs:
        _restore_monster_fight_config(snapshot)

        run_dir = infer_run_dir_from_weight_path(weight_path)
        cfg_path = args.run_config
        if cfg_path is None and run_dir and not args.no_run_config:
            candidate = os.path.join(run_dir, "config.json")
            if os.path.isfile(candidate):
                cfg_path = candidate

        if cfg_path is not None:
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError(f"Run config not found: {cfg_path}")
            with open(cfg_path, encoding="utf-8") as cf:
                run_data = json.load(cf)
            applied = apply_monster_fight_run_config_dict(run_data)
            print(f"Applied run config from {cfg_path}:")
            for ak, av in sorted(applied.items()):
                print(f"  {ak}={av}")
        else:
            print("Using config.py as-is (no run config.json loaded).")

        os.makedirs(evals_dir, exist_ok=True)
        print(f"Eval output dir: {evals_dir}")
        print(f"Loading weights from {weight_path}")

        h_eff = (
            args.hidden_size if args.hidden_size is not None else mf_cfg.MONSTER_FIGHT_HIDDEN_SIZE
        )
        n_eff = args.num_actions
        params = load_params(weight_path, h_eff, n_eff)

        steps_eff = (
            args.steps if args.steps is not None else int(mf_cfg.MONSTER_FIGHT_ROLLOUT_STEPS)
        )
        temp_eff = (
            args.temperature
            if args.temperature is not None
            else float(mf_cfg.MONSTER_FIGHT_POLICY_TEMPERATURE)
        )

        rng_key = (
            (int(args.rng_key[0]) % (1 << 32), int(args.rng_key[1]) % (1 << 32))
            if args.rng_key is not None
            else None
        )
        frames, shaped_score, kill_rate, _ = rollout(
            params,
            rollout_steps=steps_eff,
            seed=args.seed,
            hidden_size=h_eff,
            num_actions=n_eff,
            rng_key_uint32=rng_key,
            stochastic=args.stochastic,
            temperature=temp_eff,
        )

        mode_tag = f"stoch_t{temp_eff:.2f}".replace(".", "p") if args.stochastic else "greedy"
        base_name = f"{run_basename}_{label_stem}_monster_fight_{mode_tag}"
        if rng_key is not None:
            output_name = f"{base_name}_rngkey_{rng_key[0]}_{rng_key[1]}_score_{shaped_score:.2f}_kills_{kill_rate:.2f}.gif"
        else:
            output_name = (
                f"{base_name}_seed_{args.seed}_score_{shaped_score:.2f}_kills_{kill_rate:.2f}.gif"
            )
        output_path = os.path.join(evals_dir, output_name)

        print(f"Upscaling frames 63x63 -> {63 * args.scale}x{63 * args.scale}")
        frames = upscale_frames(frames, args.scale)

        save_gif(frames, output_path, args.fps)


if __name__ == "__main__":
    main()
