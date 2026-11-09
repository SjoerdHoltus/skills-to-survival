"""
GIF eval for find-food (cow hunt) policy (narrow 5-action head).

Loads ``bootstrap_find_food_state``, symbolic obs via ``make_find_food_obs``,
pixels for frames. Mirrors eval_monster_fight_symbolic.

When weights sit under ``.../run_xxx/weights/``, this script loads ``run_xxx/config.json``
by default and applies hyperparameters saved at training time—including ``static_cow``—
which **overrides** the corresponding entries in your current ``config.py``.

Use ``--no-run-config`` to evaluate with ``config.py`` only, or ``--cow-mode moving``
(or ``static``) to override cow mobility after any run config is applied.

Usage:
  python -m evaluation.eval_find_food runs/find_food/run_xxx/weights/final_....npy --seed 0
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

import configs as ff_cfg
from configs import (
    FIND_FOOD_HIDDEN_SIZE,
    FIND_FOOD_OBS_SIZE,
    FIND_FOOD_POLICY_TEMPERATURE,
    FIND_FOOD_ROLLOUT_STEPS,
    FIND_WOOD_NUM_ACTIONS,
    OBS_SIZE,
)
from policies.find_food import FindFoodPolicy
from training.monster_world_bootstrap import (
    bootstrap_find_food_state,
    make_find_food_obs,
    rebuild_mob_map_from_state,
)
from training.params import flat_params_to_pytree

_FIND_FOOD_SNAPSHOT_KEYS = (
    "FIND_FOOD_NUM_COWS_MIN",
    "FIND_FOOD_NUM_COWS_MAX",
    "FIND_FOOD_STATIC_COW",
    "FIND_FOOD_NUM_TREES_MIN",
    "FIND_FOOD_NUM_TREES_MAX",
    "MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST",
    "MONSTER_FIGHT_FOV_CHEBYSHEV_R",
    "FIND_FOOD_ROLLOUT_STEPS",
    "FIND_FOOD_KILL_REWARD",
    "FIND_FOOD_POLICY_TEMPERATURE",
    "FIND_FOOD_HIDDEN_SIZE",
    "FIND_FOOD_PLAYER_MAX_HEALTH",
)


def _snapshot_find_food_config() -> dict:
    return {k: getattr(ff_cfg, k) for k in _FIND_FOOD_SNAPSHOT_KEYS}


def _restore_find_food_config(snap: dict) -> None:
    for k, v in snap.items():
        setattr(ff_cfg, k, v)


def apply_find_food_run_config_dict(run_cfg: dict) -> dict[str, object]:
    h = run_cfg.get("hyperparameters")
    policy = run_cfg.get("policy") or {}
    if not isinstance(h, dict):
        return {}
    applied: dict[str, object] = {}
    if "num_cows" in h and "num_cows_min" not in h:
        nc = int(h["num_cows"])
        ff_cfg.FIND_FOOD_NUM_COWS_MIN = nc
        ff_cfg.FIND_FOOD_NUM_COWS_MAX = nc
        applied["num_cows (legacy)"] = nc
    if "num_trees" in h and "num_trees_min" not in h and "num_trees_max" not in h:
        nt = int(h["num_trees"])
        ff_cfg.FIND_FOOD_NUM_TREES_MIN = nt
        ff_cfg.FIND_FOOD_NUM_TREES_MAX = nt
        applied["num_trees (legacy)"] = nt
    for json_key, attr in (
        ("num_cows_min", "FIND_FOOD_NUM_COWS_MIN"),
        ("num_cows_max", "FIND_FOOD_NUM_COWS_MAX"),
        ("num_trees_min", "FIND_FOOD_NUM_TREES_MIN"),
        ("num_trees_max", "FIND_FOOD_NUM_TREES_MAX"),
        ("spawn_fov_min_chebyshev_dist", "MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST"),
        ("spawn_fov_chebyshev_r", "MONSTER_FIGHT_FOV_CHEBYSHEV_R"),
        ("rollout_steps", "FIND_FOOD_ROLLOUT_STEPS"),
    ):
        if json_key in h:
            v = int(h[json_key])
            setattr(ff_cfg, attr, v)
            applied[json_key] = v
    for json_key, attr in (
        ("kill_reward", "FIND_FOOD_KILL_REWARD"),
        ("policy_temperature", "FIND_FOOD_POLICY_TEMPERATURE"),
    ):
        if json_key in h:
            v = float(h[json_key])
            setattr(ff_cfg, attr, v)
            applied[json_key] = v
    if "static_cow" in h:
        ff_cfg.FIND_FOOD_STATIC_COW = bool(h["static_cow"])
        applied["static_cow"] = ff_cfg.FIND_FOOD_STATIC_COW
    hid = None
    if "hidden_size" in h:
        hid = int(h["hidden_size"])
    elif isinstance(policy, dict) and "hidden_size" in policy:
        hid = int(policy["hidden_size"])
    if hid is not None:
        ff_cfg.FIND_FOOD_HIDDEN_SIZE = hid
        applied["hidden_size"] = hid
    return applied


def _find_food_env_params(env):
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
    h = hidden_size if hidden_size is not None else ff_cfg.FIND_FOOD_HIDDEN_SIZE
    n = num_actions if num_actions is not None else FIND_WOOD_NUM_ACTIONS
    policy = FindFoodPolicy(hidden_size=h, num_actions=n)
    aug_init = policy.init(jax.random.PRNGKey(0), jnp.zeros(FIND_FOOD_OBS_SIZE))["params"]
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
    h = hidden_size if hidden_size is not None else ff_cfg.FIND_FOOD_HIDDEN_SIZE
    n = num_actions if num_actions is not None else FIND_WOOD_NUM_ACTIONS
    logits = FindFoodPolicy(hidden_size=h, num_actions=n).apply({"params": params}, obs)
    if rng is None:
        local_action = jnp.argmax(logits)
    else:
        scaled = logits / jnp.float32(temperature)
        local_action = jax.random.categorical(rng, scaled)
    return int(local_action + 1)


def _sync_pixel_from_symbolic(sym_state, pix_state):
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
        rollout_steps = int(ff_cfg.FIND_FOOD_ROLLOUT_STEPS)
    sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    pix_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    sym_params = _find_food_env_params(sym_env)
    pix_params = _find_food_env_params(pix_env)

    if rng_key_uint32 is not None:
        a = int(rng_key_uint32[0]) % (1 << 32)
        b = int(rng_key_uint32[1]) % (1 << 32)
        rng = jnp.asarray([jnp.uint32(a), jnp.uint32(b)], dtype=jnp.uint32)
    else:
        rng = jax.random.PRNGKey(seed)

    rng, reset_rng, boot_rng = jax.random.split(rng, 3)
    sym_obs, sym_state = sym_env.reset(reset_rng, sym_params)
    pix_obs, pix_state = pix_env.reset(reset_rng, pix_params)

    sym_state = bootstrap_find_food_state(sym_state, boot_rng, sym_params)
    pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
    sym_obs = make_find_food_obs(sym_env, sym_state)
    pix_obs = pix_env.get_obs(pix_state)

    frames = []
    episode_done = False
    initial_cows = int(jax.device_get(sym_state.cows.mask.sum()))
    kills = 0
    touched_success = 0.0
    shaped_score = 0.0

    for _step in range(rollout_steps):
        frame = np.asarray(pix_obs)
        frames.append((frame * 255).astype(np.uint8))

        prev_c = sym_state.cows

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
        freeze_c = ff_cfg.FIND_FOOD_STATIC_COW
        if freeze_c:
            post_c = sym_state.cows
            lock_c = prev_c.mask
            frozen_c = post_c.replace(
                position=jnp.where(
                    lock_c[:, None],
                    prev_c.position.astype(post_c.position.dtype),
                    post_c.position,
                ),
            )
            sym_state = sym_state.replace(cows=frozen_c)
            sym_state = sym_state.replace(mob_map=rebuild_mob_map_from_state(sym_state))

        sym_obs = make_find_food_obs(sym_env, sym_state)
        pix_obs, pix_state, _, _, _ = pix_env.step(step_rng, pix_state, action, pix_params)
        if freeze_c:
            pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
            pix_obs = pix_env.get_obs(pix_state)

        prev_cmask = np.asarray(prev_c.mask)
        new_cmask = np.asarray(sym_state.cows.mask)
        new_chealth = np.asarray(sym_state.cows.health)
        c_killed_mask = prev_cmask & (~new_cmask) & (new_chealth <= 0)
        kills_this_step = int(c_killed_mask.sum())
        kills += kills_this_step
        shaped_score += float(ff_cfg.FIND_FOOD_KILL_REWARD) * kills_this_step

        any_cow = bool(jax.device_get(sym_state.cows.mask.any()))
        if any_cow:
            p = sym_state.player_position.astype(jnp.int32)
            cd = jnp.abs(sym_state.cows.position.astype(jnp.int32) - p).sum(axis=-1)
            cd = jnp.where(sym_state.cows.mask, cd, jnp.asarray(1_000_000, dtype=jnp.int32))
            min_d = int(jax.device_get(cd.min()))
            touched = 1.0 if min_d <= 1 else 0.0
        else:
            touched = 0.0
        touched_success = max(touched_success, touched)

        episode_done = episode_done or bool(jax.device_get(done))
        if not any_cow:
            episode_done = True
        if episode_done:
            break

    achievements = np.asarray(sym_state.achievements)
    kill_rate = kills / max(initial_cows, 1)
    print(
        f"Rollout finished: {len(frames)} steps, shaped_score={shaped_score:.2f}, "
        f"cow_kills={kills}/{initial_cows} (kill_rate={kill_rate:.2f}), touch={touched_success:.1f}"
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
        description="Evaluate find_food policy on Craftax-Classic with pixel GIF (CPU JAX)"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="Path(s) to FindFoodPolicy .npy checkpoint(s).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Rollout horizon (default run config rollout_steps else "
            f"FIND_FOOD_ROLLOUT_STEPS={FIND_FOOD_ROLLOUT_STEPS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rng-key",
        nargs=2,
        type=int,
        metavar=("U32_0", "U32_1"),
        default=None,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help=f"Hidden width (default {FIND_FOOD_HIDDEN_SIZE})",
    )
    parser.add_argument("--num-actions", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Softmax temperature for --stochastic (default policy_temperature from run "
            f"config else {FIND_FOOD_POLICY_TEMPERATURE})"
        ),
    )
    parser.add_argument("--run-config", type=str, default=None, metavar="PATH")
    parser.add_argument("--no-run-config", action="store_true")
    parser.add_argument(
        "--cow-mode",
        choices=("static", "moving"),
        default=None,
        help=(
            "Override FIND_FOOD_STATIC_COW after loading run config: "
            "static=True, moving=False (default: use config.json / config.py)."
        ),
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

    snapshot = _snapshot_find_food_config()

    for weight_path, evals_dir, run_basename, label_stem in jobs:
        _restore_find_food_config(snapshot)

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
            applied = apply_find_food_run_config_dict(run_data)
            print(f"Applied run config from {cfg_path}:")
            for ak, av in sorted(applied.items()):
                print(f"  {ak}={av}")
        else:
            print("Using config.py as-is (no run config.json loaded).")

        if args.cow_mode is not None:
            ff_cfg.FIND_FOOD_STATIC_COW = args.cow_mode == "static"
            print(f"Override from --cow-mode: FIND_FOOD_STATIC_COW={ff_cfg.FIND_FOOD_STATIC_COW}")

        os.makedirs(evals_dir, exist_ok=True)
        print(f"Eval output dir: {evals_dir}")
        print(f"Loading weights from {weight_path}")

        h_eff = args.hidden_size if args.hidden_size is not None else ff_cfg.FIND_FOOD_HIDDEN_SIZE
        n_eff = args.num_actions
        params_obj = load_params(weight_path, h_eff, n_eff)

        steps_eff = args.steps if args.steps is not None else int(ff_cfg.FIND_FOOD_ROLLOUT_STEPS)
        temp_eff = (
            args.temperature
            if args.temperature is not None
            else float(ff_cfg.FIND_FOOD_POLICY_TEMPERATURE)
        )

        rng_key = (
            (int(args.rng_key[0]) % (1 << 32), int(args.rng_key[1]) % (1 << 32))
            if args.rng_key is not None
            else None
        )
        frames, shaped_score, kill_rate, _ = rollout(
            params_obj,
            rollout_steps=steps_eff,
            seed=args.seed,
            hidden_size=h_eff,
            num_actions=n_eff,
            rng_key_uint32=rng_key,
            stochastic=args.stochastic,
            temperature=temp_eff,
        )

        mode_tag = f"stoch_t{temp_eff:.2f}".replace(".", "p") if args.stochastic else "greedy"
        base_name = f"{run_basename}_{label_stem}_find_food_{mode_tag}"
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
