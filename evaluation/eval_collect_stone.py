"""
Evaluate collect-stone policy weights; save a pixel GIF.

Bootstrap (training layout):
  python -m evaluation.eval_collect_stone \\
    --weights runs/collect_stone/run_xxx/weights/final_....npy --seed 0

Natural Classic map (procedural terrain + ore/stone veins):
  python -m evaluation.eval_collect_stone \\
    --weights runs/collect_stone/run_xxx/weights/final_....npy --natural --seed 0

  # Scan seeds until stone is visible in FOV, then render:
  python -m evaluation.eval_collect_stone \\
    --weights ... --natural --seed 0 --require-target-in-fov
"""

from __future__ import annotations

import argparse
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import craftax  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name

from configs import (
    COLLECT_STONE_HIDDEN_SIZE,
    COLLECT_STONE_MAX_STONE,
    COLLECT_STONE_NUM_ACTIONS,
    COLLECT_STONE_OBS_SIZE,
    COLLECT_STONE_ROLLOUT_STEPS,
)
from evaluation.eval_common import (
    evals_dir_and_basename_for_weight,
    save_gif,
    upscale_frames,
)
from policies.collect_stone import CollectStonePolicy
from training.bootstrap_stone import bootstrap_stone_state, make_collect_stone_obs
from training.natural_mining_setup import (
    natural_stone_target_in_fov,
    resolve_natural_seed,
    setup_natural_collect_stone_state,
    target_visible_in_obs,
)
from training.params import flat_params_to_pytree, pytree_to_flat_params


def _env_params_no_spawn(env):
    p = env.default_params
    return p.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
        mob_despawn_distance=10_000,
    )


def load_params(path, hidden_size=None, num_actions=None):
    flat_on_disk = jnp.load(path)
    h = hidden_size if hidden_size is not None else COLLECT_STONE_HIDDEN_SIZE
    n = num_actions if num_actions is not None else COLLECT_STONE_NUM_ACTIONS
    policy = CollectStonePolicy(hidden_size=h, num_actions=n)
    dummy = jnp.zeros(COLLECT_STONE_OBS_SIZE)
    init_params = policy.init(jax.random.PRNGKey(0), dummy)["params"]
    flat_init, shapes, tree_def = pytree_to_flat_params(init_params)
    if flat_on_disk.shape[0] != flat_init.shape[0]:
        raise ValueError(
            f"Weights length {flat_on_disk.shape[0]} != policy {flat_init.shape[0]} "
            f"(obs={COLLECT_STONE_OBS_SIZE}, hidden={h}, actions={n}; "
            "13-D checkpoints need retraining for 12-D obs)"
        )
    return flat_params_to_pytree(flat_on_disk, shapes, tree_def)


def _sync_pixel_from_symbolic(sym_state, pix_state):
    return pix_state.replace(
        map=sym_state.map,
        mob_map=sym_state.mob_map,
        zombies=sym_state.zombies,
        skeletons=sym_state.skeletons,
        cows=sym_state.cows,
        arrows=sym_state.arrows,
        inventory=sym_state.inventory,
        player_position=sym_state.player_position,
        player_direction=sym_state.player_direction,
        player_health=sym_state.player_health,
        player_food=sym_state.player_food,
        player_drink=sym_state.player_drink,
        player_energy=sym_state.player_energy,
        player_recover=sym_state.player_recover,
        player_hunger=sym_state.player_hunger,
        player_thirst=sym_state.player_thirst,
        player_fatigue=sym_state.player_fatigue,
        is_sleeping=sym_state.is_sleeping,
        light_level=sym_state.light_level,
        achievements=sym_state.achievements,
        timestep=sym_state.timestep,
    )


def pick_action(params, obs, hidden_size=None, num_actions=None):
    h = hidden_size if hidden_size is not None else COLLECT_STONE_HIDDEN_SIZE
    n = num_actions if num_actions is not None else COLLECT_STONE_NUM_ACTIONS
    logits = CollectStonePolicy(hidden_size=h, num_actions=n).apply({"params": params}, obs)
    return int(jnp.argmax(logits) + 1)


def rollout(
    params,
    rollout_steps=COLLECT_STONE_ROLLOUT_STEPS,
    seed=0,
    hidden_size=None,
    num_actions=None,
    natural: bool = False,
):
    sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    pix_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    sym_params = _env_params_no_spawn(sym_env)
    pix_params = _env_params_no_spawn(pix_env)

    master = jax.random.PRNGKey(seed)
    carry_key, split2 = jax.random.split(master)
    reset_rng, boot_rng = jax.random.split(split2)

    sym_obs, sym_state = sym_env.reset(reset_rng, sym_params)
    pix_obs, pix_state = pix_env.reset(reset_rng, pix_params)
    if natural:
        sym_state = setup_natural_collect_stone_state(sym_state)
    else:
        sym_state = bootstrap_stone_state(sym_state, boot_rng, sym_params)
    pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
    sym_obs = make_collect_stone_obs(sym_env, sym_state)
    pix_obs = pix_env.get_obs(pix_state)
    target_in_fov = bool(float(target_visible_in_obs(sym_obs)) >= 0.5)

    frames = []
    full_stone = False
    steps_used = 0
    step_key = carry_key

    for _step in range(rollout_steps):
        steps_used += 1
        frame = np.asarray(pix_obs)
        frames.append((frame * 255).astype(np.uint8))

        action = pick_action(params, sym_obs, hidden_size, num_actions)
        step_key, step_rng = jax.random.split(step_key)
        sym_obs, sym_state, _r, done, _ = sym_env.step(step_rng, sym_state, action, sym_params)
        pix_obs, pix_state, _, _, _ = pix_env.step(step_rng, pix_state, action, pix_params)
        pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
        pix_obs = pix_env.get_obs(pix_state)
        sym_obs = make_collect_stone_obs(sym_env, sym_state)

        new_s = int(jax.device_get(sym_state.inventory.stone))
        if new_s >= COLLECT_STONE_MAX_STONE:
            full_stone = True
        if done or full_stone:
            break

    ach = np.asarray(jax.device_get(sym_state.achievements))
    collect_stone = bool(ach[Achievement.COLLECT_STONE.value])
    print(
        f"Rollout natural={natural} target_in_fov={target_in_fov}: steps={steps_used} "
        f"full_stone={full_stone} COLLECT_STONE_achievement={collect_stone} "
        f"final_stone={int(jax.device_get(sym_state.inventory.stone))}"
    )
    return frames, full_stone, target_in_fov


def main():
    p = argparse.ArgumentParser(description="GIF eval for collect-stone policy")
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=COLLECT_STONE_ROLLOUT_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-actions", type=int, default=None)
    p.add_argument(
        "--natural",
        action="store_true",
        help="Use procedurally generated map (pickaxe only); skip stone bootstrap layout",
    )
    p.add_argument(
        "--require-target-in-fov",
        action="store_true",
        help="With --natural: scan seeds until stone is visible in FOV before rendering",
    )
    p.add_argument(
        "--seed-scan-max",
        type=int,
        default=500,
        help="Max seeds to scan when --require-target-in-fov (starts at --seed)",
    )
    args = p.parse_args()

    if args.require_target_in_fov and not args.natural:
        p.error("--require-target-in-fov only applies with --natural")

    for wp in args.weights:
        if not os.path.isfile(wp):
            p.error(f"not found: {wp}")

    for weight_path in args.weights:
        abs_wp = os.path.abspath(weight_path)
        evals_dir, run_basename, label_stem = evals_dir_and_basename_for_weight(abs_wp)
        os.makedirs(evals_dir, exist_ok=True)
        params = load_params(abs_wp, args.hidden_size, args.num_actions)
        eval_seed = args.seed
        if args.natural and args.require_target_in_fov:
            sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
            sym_params = _env_params_no_spawn(sym_env)
            eval_seed = resolve_natural_seed(
                lambda s: natural_stone_target_in_fov(sym_env, sym_params, s),
                args.seed,
                args.seed_scan_max,
                True,
            )
        frames, full_stone, target_in_fov = rollout(
            params,
            rollout_steps=args.steps,
            seed=eval_seed,
            hidden_size=args.hidden_size,
            num_actions=args.num_actions,
            natural=args.natural,
        )
        mode = "natural" if args.natural else "bootstrap"
        base = (
            f"{run_basename}_{label_stem}_collect_stone_{mode}_"
            f"seed_{eval_seed}_fov_{int(target_in_fov)}_full_{int(full_stone)}"
        )
        out = os.path.join(evals_dir, f"{base}.gif")
        frames = upscale_frames(frames, args.scale)
        save_gif(frames, out, args.fps)


if __name__ == "__main__":
    main()
