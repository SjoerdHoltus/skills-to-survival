"""
Evaluate drink-water policy weights; save a pixel GIF (Craftax-Classic).

Symbolic rollouts use ``bootstrap_water_state`` + 13-D compact features;
greedy argmax over 5 logits → env actions 1–5. Pixel env stays in lockstep for
rendering (state synced from symbolic after bootstrap and each step).

Usage:
  python -m evaluation.eval_drink_water --weights runs/drink_water/run_xxx/weights/final_....npy --seed 0
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
    DRINK_WATER_HIDDEN_SIZE,
    DRINK_WATER_MAX_DRINK,
    DRINK_WATER_OBS_SIZE,
    DRINK_WATER_ROLLOUT_STEPS,
    FIND_WOOD_NUM_ACTIONS,
)
from evaluation.eval_common import (
    evals_dir_and_basename_for_weight,
    save_gif,
    upscale_frames,
)
from policies.drink_water import DrinkWaterPolicy
from training.bootstrap_water import bootstrap_water_state, make_drink_water_obs
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


def load_params(path, hidden_size=None, num_actions=None):
    flat_params = jnp.load(path)
    h = hidden_size if hidden_size is not None else DRINK_WATER_HIDDEN_SIZE
    n = num_actions if num_actions is not None else FIND_WOOD_NUM_ACTIONS
    policy = DrinkWaterPolicy(hidden_size=h, num_actions=n)
    dummy = jnp.zeros(DRINK_WATER_OBS_SIZE)
    init_params = policy.init(jax.random.PRNGKey(0), dummy)["params"]
    leaves, tree_def = jax.tree_util.tree_flatten(init_params)
    shapes = [leaf.shape for leaf in leaves]
    return flat_params_to_pytree(flat_params, shapes, tree_def)


def _sync_pixel_from_symbolic(sym_state, pix_state):
    """Mirror symbolic rollouts into the pixel env for rendering.

    Bootstrap and steps update ``player_drink`` (and other vitals) only on the
    symbolic state. Craftax reset defaults ``player_drink=9``; without copying
    vitals here the GIF HUD would show a full bottle while training sees 0.
    """
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
    h = hidden_size if hidden_size is not None else DRINK_WATER_HIDDEN_SIZE
    n = num_actions if num_actions is not None else FIND_WOOD_NUM_ACTIONS
    logits = DrinkWaterPolicy(hidden_size=h, num_actions=n).apply({"params": params}, obs)
    return int(jnp.argmax(logits) + 1)


def rollout(
    params,
    rollout_steps=DRINK_WATER_ROLLOUT_STEPS,
    seed=0,
    hidden_size=None,
    num_actions=None,
    rng_key_uint32=None,
):
    sym_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    pix_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    sym_params = _env_params_no_spawn(sym_env)
    pix_params = _env_params_no_spawn(pix_env)

    if rng_key_uint32 is not None:
        a = int(rng_key_uint32[0]) % (1 << 32)
        b = int(rng_key_uint32[1]) % (1 << 32)
        master = jnp.asarray([jnp.uint32(a), jnp.uint32(b)], dtype=jnp.uint32)
    else:
        master = jax.random.PRNGKey(seed)

    carry_key, split2 = jax.random.split(master)
    reset_rng, boot_rng = jax.random.split(split2)

    sym_obs, sym_state = sym_env.reset(reset_rng, sym_params)
    pix_obs, pix_state = pix_env.reset(reset_rng, pix_params)
    sym_state = bootstrap_water_state(sym_state, boot_rng, sym_params)
    pix_state = _sync_pixel_from_symbolic(sym_state, pix_state)
    sym_obs = make_drink_water_obs(sym_env, sym_state)
    pix_obs = pix_env.get_obs(pix_state)

    frames = []
    full = False
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
        sym_obs = make_drink_water_obs(sym_env, sym_state)

        new_d = int(jax.device_get(sym_state.player_drink))
        if new_d >= DRINK_WATER_MAX_DRINK:
            full = True
        if done or full:
            break

    ach = np.asarray(jax.device_get(sym_state.achievements))
    collect_drink = bool(ach[Achievement.COLLECT_DRINK.value])
    print(
        f"Rollout: steps={steps_used} full={full} COLLECT_DRINK_achievement={collect_drink} "
        f"final_drink={int(jax.device_get(sym_state.player_drink))}"
    )
    return frames, full


def main():
    p = argparse.ArgumentParser(description="GIF eval for drink-water policy")
    p.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="Paths to .npy mean weights (DrinkWaterPolicy flat vector)",
    )
    p.add_argument("--steps", type=int, default=DRINK_WATER_ROLLOUT_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rng-key",
        nargs=2,
        type=int,
        metavar=("U32_0", "U32_1"),
        default=None,
        help="Explicit PRNG key (overrides --seed) for reset/bootstrap/step stream",
    )
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-actions", type=int, default=None)
    args = p.parse_args()

    for wp in args.weights:
        if not os.path.isfile(wp):
            p.error(f"not found: {wp}")

    rng_key = None
    if args.rng_key is not None:
        rng_key = (
            int(args.rng_key[0]) % (1 << 32),
            int(args.rng_key[1]) % (1 << 32),
        )

    for weight_path in args.weights:
        abs_wp = os.path.abspath(weight_path)
        evals_dir, run_basename, label_stem = evals_dir_and_basename_for_weight(abs_wp)
        os.makedirs(evals_dir, exist_ok=True)
        params = load_params(abs_wp, args.hidden_size, args.num_actions)
        frames, full = rollout(
            params,
            rollout_steps=args.steps,
            seed=args.seed,
            hidden_size=args.hidden_size,
            num_actions=args.num_actions,
            rng_key_uint32=rng_key,
        )
        tag = f"rng_{rng_key[0]}_{rng_key[1]}" if rng_key is not None else f"seed_{args.seed}"
        base = f"{run_basename}_{label_stem}_drink_water_{tag}_full_{int(full)}"
        out = os.path.join(evals_dir, f"{base}.gif")
        frames = upscale_frames(frames, args.scale)
        save_gif(frames, out, args.fps)


if __name__ == "__main__":
    main()
