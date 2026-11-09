"""
Evaluate collect-ore policy weights (coal|iron|diamond); save a pixel GIF.

Bootstrap (training layout):
  python -m evaluation.eval_collect_ore \\
    --weights runs/collect_ore/run_xxx/weights/final_....npy --target coal --seed 0

Replay a batch-eval rollout (``rng_key_uint32`` from ``worst_rollouts``):
  python -m evaluation.eval_collect_ore \\
    --weights ... --target coal --rng-key 1234567890 9876543210

Natural Classic map:
  python -m evaluation.eval_collect_ore \\
    --weights ... --target coal --natural --seed 0

  python -m evaluation.eval_collect_ore \\
    --weights ... --target iron --natural --require-target-in-fov --seed 0
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
    COLLECT_ORE_FEAT_HAS_TARGET,
    COLLECT_ORE_HIDDEN_SIZE,
    COLLECT_ORE_NUM_ACTIONS,
    COLLECT_ORE_OBS_SIZE,
    COLLECT_ORE_ROLLOUT_STEPS,
    COLLECT_ORE_TARGET_NAMES,
)
from evaluation.eval_common import (
    evals_dir_and_basename_for_weight,
    save_gif,
    upscale_frames,
)
from policies.collect_ore import CollectOrePolicy
from training.bootstrap_ore import (
    TARGET_CAPS,
    bootstrap_ore_state,
    make_collect_ore_obs,
    target_id_from_name,
)
from training.natural_mining_setup import (
    natural_ore_target_in_fov,
    resolve_natural_seed,
    setup_natural_collect_ore_state,
    target_visible_in_obs,
)
from training.params import flat_params_to_pytree


def _env_params_no_spawn(env):
    p = env.default_params
    return p.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
        mob_despawn_distance=10_000,
    )


def _achievement_for_target(name: str) -> Achievement:
    mapping = {
        "coal": Achievement.COLLECT_COAL,
        "iron": Achievement.COLLECT_IRON,
        "diamond": Achievement.COLLECT_DIAMOND,
    }
    return mapping[name.lower()]


def load_params(path, hidden_size=None, num_actions=None):
    flat_params = jnp.load(path)
    h = hidden_size if hidden_size is not None else COLLECT_ORE_HIDDEN_SIZE
    n = num_actions if num_actions is not None else COLLECT_ORE_NUM_ACTIONS
    policy = CollectOrePolicy(hidden_size=h, num_actions=n)
    dummy = jnp.zeros(COLLECT_ORE_OBS_SIZE)
    init_params = policy.init(jax.random.PRNGKey(0), dummy)["params"]
    leaves, tree_def = jax.tree_util.tree_flatten(init_params)
    shapes = [leaf.shape for leaf in leaves]
    return flat_params_to_pytree(flat_params, shapes, tree_def)


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
    h = hidden_size if hidden_size is not None else COLLECT_ORE_HIDDEN_SIZE
    n = num_actions if num_actions is not None else COLLECT_ORE_NUM_ACTIONS
    logits = CollectOrePolicy(hidden_size=h, num_actions=n).apply({"params": params}, obs)
    return int(jnp.argmax(logits) + 1)


def rollout(
    params,
    target_name: str,
    wall_tiles: int = 1,
    rollout_steps=COLLECT_ORE_ROLLOUT_STEPS,
    seed=0,
    hidden_size=None,
    num_actions=None,
    natural: bool = False,
    rng_key_uint32=None,
):
    target_id = jnp.int32(target_id_from_name(target_name))
    cap = int(TARGET_CAPS[target_id_from_name(target_name)])
    shell_depth = jnp.int32(wall_tiles)

    sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    pix_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    sym_params = _env_params_no_spawn(sym_env)
    pix_params = _env_params_no_spawn(pix_env)

    if rng_key_uint32 is not None:
        a = int(rng_key_uint32[0]) % (1 << 32)
        b = int(rng_key_uint32[1]) % (1 << 32)
        rollout_key = jnp.asarray([jnp.uint32(a), jnp.uint32(b)], dtype=jnp.uint32)
        # Same split as ``training.fitness_collect_ore`` single_rollout (batch eval keys).
        carry_key, reset_rng, boot_rng, _shell_rng = jax.random.split(rollout_key, 4)
    else:
        master = jax.random.PRNGKey(seed)
        carry_key, split2 = jax.random.split(master)
        reset_rng, boot_rng = jax.random.split(split2)

    sym_obs, sym_state = sym_env.reset(reset_rng, sym_params)
    pix_obs, pix_state = pix_env.reset(reset_rng, pix_params)
    if natural:
        sym_state = setup_natural_collect_ore_state(sym_state, target_id=target_id)
    else:
        sym_state = bootstrap_ore_state(
            sym_state, boot_rng, sym_params, target_id, wall_count=shell_depth
        )
    pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
    sym_obs = make_collect_ore_obs(sym_env, sym_state, target_id)
    pix_obs = pix_env.get_obs(pix_state)
    target_in_fov = bool(float(target_visible_in_obs(sym_obs, COLLECT_ORE_FEAT_HAS_TARGET)) >= 0.5)

    frames = []
    full_cap = False
    steps_used = 0
    step_key = carry_key
    inv_field = target_name.lower()

    for _step in range(rollout_steps):
        steps_used += 1
        frames.append((np.asarray(pix_obs) * 255).astype(np.uint8))

        action = pick_action(params, sym_obs, hidden_size, num_actions)
        step_key, step_rng = jax.random.split(step_key)
        sym_obs, sym_state, _r, done, _ = sym_env.step(step_rng, sym_state, action, sym_params)
        pix_obs, pix_state, _, _, _ = pix_env.step(step_rng, pix_state, action, pix_params)
        pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
        pix_obs = pix_env.get_obs(pix_state)
        sym_obs = make_collect_ore_obs(sym_env, sym_state, target_id)

        new_amt = int(getattr(jax.device_get(sym_state.inventory), inv_field))
        if new_amt >= cap:
            full_cap = True
        if done or full_cap:
            break

    ach = np.asarray(jax.device_get(sym_state.achievements))
    ach_enum = _achievement_for_target(target_name)
    got_ach = bool(ach[ach_enum.value])
    final_amt = int(getattr(jax.device_get(sym_state.inventory), inv_field))
    rng_note = (
        f"rng_key_uint32=({rng_key_uint32[0]}, {rng_key_uint32[1]})"
        if rng_key_uint32 is not None
        else f"seed={seed}"
    )
    print(
        f"Rollout target={target_name} natural={natural} {rng_note} "
        f"shell_depth={wall_tiles if not natural else 'n/a'} "
        f"target_in_fov={target_in_fov}: steps={steps_used} "
        f"full_cap={full_cap} achievement={ach_enum.name}={got_ach} "
        f"final_{inv_field}={final_amt}"
    )
    return frames, full_cap, final_amt >= 1, target_in_fov


def main():
    p = argparse.ArgumentParser(description="GIF eval for collect-ore policy (coal|iron|diamond)")
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--target", required=True, choices=list(COLLECT_ORE_TARGET_NAMES))
    p.add_argument(
        "--wall-tiles",
        type=int,
        default=1,
        help="Stone shell depth around target in bootstrap (1–3); kept as --wall-tiles for compatibility",
    )
    p.add_argument("--steps", type=int, default=COLLECT_ORE_ROLLOUT_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rng-key",
        nargs=2,
        type=int,
        metavar=("U32_0", "U32_1"),
        default=None,
        help="Explicit rollout PRNG key (overrides --seed); same as batch_eval worst_rollouts rng_key_uint32",
    )
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-actions", type=int, default=None)
    p.add_argument(
        "--natural",
        action="store_true",
        help="Use procedurally generated map (dual pickaxe only); skip walled bootstrap",
    )
    p.add_argument(
        "--require-target-in-fov",
        action="store_true",
        help="With --natural: scan seeds until target ore is visible in FOV before rendering",
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
    if args.rng_key is not None and args.require_target_in_fov:
        p.error("--rng-key cannot be used with --require-target-in-fov (fixed world)")

    rng_key = None
    if args.rng_key is not None:
        rng_key = (
            int(args.rng_key[0]) % (1 << 32),
            int(args.rng_key[1]) % (1 << 32),
        )

    for wp in args.weights:
        if not os.path.isfile(wp):
            p.error(f"not found: {wp}")

    for weight_path in args.weights:
        abs_wp = os.path.abspath(weight_path)
        evals_dir, run_basename, label_stem = evals_dir_and_basename_for_weight(abs_wp)
        os.makedirs(evals_dir, exist_ok=True)
        params = load_params(abs_wp, args.hidden_size, args.num_actions)
        target_id = target_id_from_name(args.target)
        eval_seed = args.seed
        if rng_key is not None:
            print(f"Using rollout RNG key ({rng_key[0]}, {rng_key[1]}) (batch_eval rng_key_uint32)")
        elif args.natural and args.require_target_in_fov:
            sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
            sym_params = _env_params_no_spawn(sym_env)
            probe = lambda s: natural_ore_target_in_fov(  # noqa: E731
                sym_env, sym_params, s, target_id
            )
            eval_seed = resolve_natural_seed(probe, args.seed, args.seed_scan_max, True)
        frames, full_cap, mined_any, target_in_fov = rollout(
            params,
            target_name=args.target,
            wall_tiles=args.wall_tiles,
            rollout_steps=args.steps,
            seed=eval_seed,
            hidden_size=args.hidden_size,
            num_actions=args.num_actions,
            natural=args.natural,
            rng_key_uint32=rng_key,
        )
        tag = f"rng_{rng_key[0]}_{rng_key[1]}" if rng_key is not None else f"seed_{eval_seed}"
        if args.natural:
            base = (
                f"{run_basename}_{label_stem}_collect_{args.target}_natural_"
                f"{tag}_fov_{int(target_in_fov)}_full_{int(full_cap)}_"
                f"mined_{int(mined_any)}"
            )
        else:
            base = (
                f"{run_basename}_{label_stem}_collect_{args.target}_shell{args.wall_tiles}_"
                f"{tag}_full_{int(full_cap)}_mined_{int(mined_any)}"
            )
        out = os.path.join(evals_dir, f"{base}.gif")
        save_gif(upscale_frames(frames, args.scale), out, args.fps)


if __name__ == "__main__":
    main()
