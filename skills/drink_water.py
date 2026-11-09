"""
OpenES training for drinking from water on Craftax-Classic-Symbolic-v1.

Observation: 13-D features from ``training.bootstrap_water`` (handcrafted from
the symbolic tensor). After ``env.reset``, ``bootstrap_water_state`` strips
natural WATER and paints a single patch; mobs are cleared.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.open_es import Open_ES

import configs as rl_cfg
from configs import (
    DRINK_WATER_FULL_BONUS,
    DRINK_WATER_GREEDY_FINETUNE_FRACTION,
    DRINK_WATER_HIDDEN_SIZE,
    DRINK_WATER_LEARNING_RATE,
    DRINK_WATER_MAX_DRINK,
    DRINK_WATER_NUM_ROLLOUTS_ANNEAL_FRACTION,
    DRINK_WATER_NUM_ROLLOUTS_START,
    DRINK_WATER_OBS_SIZE,
    DRINK_WATER_PER_DRINK_REWARD,
    DRINK_WATER_POLICY_TEMPERATURE_ANNEAL_FRACTION,
    DRINK_WATER_POLICY_TEMPERATURE_END,
    DRINK_WATER_POLICY_TEMPERATURE_START,
    DRINK_WATER_REACH_MANHATTAN_THRESHOLD,
    DRINK_WATER_ROLLOUT_REFRESH_EVERY,
    DRINK_WATER_ROLLOUT_STEPS,
    DRINK_WATER_STD_END,
    DRINK_WATER_STD_START,
    DRINK_WATER_TARGET_FEATURES,
    FIND_WOOD_NUM_ACTIONS,
    OBS_SIZE,
)
from policies.drink_water import DrinkWaterPolicy
from training import (
    append_training_log_line,
    finalize_run,
    get_run_dir,
    save_best_es_weights,
    save_checkpoint,
    save_config,
    save_final,
    setup_run_dir,
)
from training.fitness_drink_water import make_env, make_fitness_fn
from training.params import flat_params_to_pytree, pytree_to_flat_params
from training.skill_variant import runs_policy_folder
from training.tb_logger import TBLogger


def _temperature_for_generation(gen: int, generations: int) -> float:
    if generations <= 1:
        return float(DRINK_WATER_POLICY_TEMPERATURE_END)
    p = gen / float(generations - 1)
    af = max(float(DRINK_WATER_POLICY_TEMPERATURE_ANNEAL_FRACTION), 1e-8)
    t = min(max(p / af, 0.0), 1.0)
    start = float(DRINK_WATER_POLICY_TEMPERATURE_START)
    end = float(DRINK_WATER_POLICY_TEMPERATURE_END)
    return start + (end - start) * t


def _rollouts_for_generation(gen: int, generations: int, target_rollouts: int) -> int:
    start = int(DRINK_WATER_NUM_ROLLOUTS_START)
    end = int(target_rollouts)
    start = max(1, min(start, end))
    if generations <= 1:
        return end
    p = gen / float(generations - 1)
    af = max(float(DRINK_WATER_NUM_ROLLOUTS_ANNEAL_FRACTION), 1e-8)
    frac = min(max(p / af, 0.0), 1.0)
    val = int(round(start + (end - start) * frac))
    return max(1, min(val, end))


def train_drink_water(
    variant: str = "classic",
    pop_size: int = 160,
    generations: int = 400,
    rollout_steps: int = DRINK_WATER_ROLLOUT_STEPS,
    num_rollouts: int = 96,
    seed: int = 42,
    resume_from: str | None = None,
    log_dir: str | None = None,
):
    run_dir, start_time = get_run_dir(runs_policy_folder("drink_water", variant))
    run_weights_dir, run_tb_dir = setup_run_dir(run_dir)

    metrics_json_path = os.path.join(run_dir, "training_metrics.json")
    logger = TBLogger(log_dir or run_tb_dir, json_path=metrics_json_path)

    rng = jax.random.PRNGKey(seed)
    env, env_params = make_env(variant=variant)

    policy = DrinkWaterPolicy(
        hidden_size=DRINK_WATER_HIDDEN_SIZE, num_actions=FIND_WOOD_NUM_ACTIONS
    )
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(DRINK_WATER_OBS_SIZE)
    init_params = policy.init(init_rng, dummy_obs)["params"]
    flat_init, param_shapes, param_tree_def = pytree_to_flat_params(init_params)
    num_params = int(flat_init.shape[0])

    cfg_out = {
        "runs_policy_folder": runs_policy_folder("drink_water", variant),
        "craftax_variant": variant,
        "task": "drink_water_symbolic",
        "hyperparameters": {
            "pop_size": pop_size,
            "generations": generations,
            "rollout_steps": rollout_steps,
            "num_rollouts": num_rollouts,
            "num_rollouts_start": DRINK_WATER_NUM_ROLLOUTS_START,
            "num_rollouts_anneal_fraction": DRINK_WATER_NUM_ROLLOUTS_ANNEAL_FRACTION,
            "seed": seed,
            "resume_from": resume_from,
            "hidden_size": DRINK_WATER_HIDDEN_SIZE,
            "shared_rollout_rng_across_population": True,
            "rollout_refresh_every_gens": DRINK_WATER_ROLLOUT_REFRESH_EVERY,
            "objective": "collect_drink_via_DO_on_water_until_full",
            "max_drink": int(DRINK_WATER_MAX_DRINK),
            "per_drink_reward": float(DRINK_WATER_PER_DRINK_REWARD),
            "full_bonus": float(DRINK_WATER_FULL_BONUS),
            "player_start_thirst": float(rl_cfg.DRINK_WATER_PLAYER_START_THIRST),
            "player_start_drink": int(rl_cfg.DRINK_WATER_PLAYER_START_DRINK),
            "reach_manhattan_threshold_blocks": int(DRINK_WATER_REACH_MANHATTAN_THRESHOLD),
            "water_bootstrap": {
                "spawn_min_chebyshev_r": int(rl_cfg.DRINK_WATER_SPAWN_MIN_CHEBYSHEV_R),
                "spawn_max_chebyshev_r": int(rl_cfg.DRINK_WATER_SPAWN_MAX_CHEBYSHEV_R),
                "water_patch_side": int(rl_cfg.DRINK_WATER_WATER_PATCH_SIDE),
            },
            "optimizer": "adam",
            "learning_rate": DRINK_WATER_LEARNING_RATE,
            "std_schedule": {
                "type": "linear",
                "start": float(DRINK_WATER_STD_START),
                "end": float(DRINK_WATER_STD_END),
                "transition_steps": max(int(generations) - 1, 1),
            },
            "action_selection": "categorical_gumbel_max",
            "policy_temperature_start": DRINK_WATER_POLICY_TEMPERATURE_START,
            "policy_temperature_end": DRINK_WATER_POLICY_TEMPERATURE_END,
            "policy_temperature_anneal_fraction": DRINK_WATER_POLICY_TEMPERATURE_ANNEAL_FRACTION,
            "greedy_finetune_fraction": DRINK_WATER_GREEDY_FINETUNE_FRACTION,
        },
        "policy": {
            "name": "DrinkWaterPolicy",
            "obs_size": DRINK_WATER_OBS_SIZE,
            "obs_size_raw_symbolic": int(OBS_SIZE),
            "feature_dim": DRINK_WATER_TARGET_FEATURES,
            "num_actions": FIND_WOOD_NUM_ACTIONS,
            "action_space": "LEFT, RIGHT, UP, DOWN, DO (Craftax 1–5)",
            "hidden_size": DRINK_WATER_HIDDEN_SIZE,
            "layers": [
                f"Dense({DRINK_WATER_HIDDEN_SIZE})",
                "ReLU",
                f"Dense({FIND_WOOD_NUM_ACTIONS})",
            ],
            "num_params": num_params,
        },
    }
    save_config(run_dir, cfg_out)

    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)
        append_training_log_line(run_dir, msg)

    log(
        f"Drink-water params={num_params} obs={DRINK_WATER_OBS_SIZE} "
        f"hidden={DRINK_WATER_HIDDEN_SIZE} actions={FIND_WOOD_NUM_ACTIONS}"
    )

    if resume_from and os.path.exists(resume_from):
        init_mean = jnp.load(resume_from)
        if int(init_mean.shape[0]) != num_params:
            raise ValueError(
                f"Checkpoint has {init_mean.shape[0]} params; DrinkWaterPolicy expects {num_params}."
            )
        log(f"Resuming from {resume_from}")
    else:
        init_mean = jnp.zeros(num_params)

    fitness_fn_stoch = make_fitness_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        num_rollouts=num_rollouts,
        hidden_size=DRINK_WATER_HIDDEN_SIZE,
        num_actions=FIND_WOOD_NUM_ACTIONS,
        stochastic_actions=True,
        variant=variant,
    )
    fitness_fn_greedy = make_fitness_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        num_rollouts=num_rollouts,
        hidden_size=DRINK_WATER_HIDDEN_SIZE,
        num_actions=FIND_WOOD_NUM_ACTIONS,
        stochastic_actions=False,
        variant=variant,
    )
    batch_stoch = jax.vmap(fitness_fn_stoch, in_axes=(0, None, None, None))
    batch_greed = jax.vmap(fitness_fn_greedy, in_axes=(0, None, None, None))

    refresh_every = max(1, int(DRINK_WATER_ROLLOUT_REFRESH_EVERY))
    eval_rng_base = jax.random.PRNGKey(seed + 414_414)

    std_schedule = optax.linear_schedule(
        init_value=float(DRINK_WATER_STD_START),
        end_value=float(DRINK_WATER_STD_END),
        transition_steps=max(int(generations) - 1, 1),
    )
    strategy = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        optimizer=optax.adam(learning_rate=DRINK_WATER_LEARNING_RATE),
        std_schedule=std_schedule,
    )
    es_params = strategy.default_params
    rng, init_rng = jax.random.split(rng)
    es_state = strategy.init(init_rng, init_mean, es_params)

    greedy_frac = min(max(float(DRINK_WATER_GREEDY_FINETUNE_FRACTION), 0.0), 1.0)
    greedy_start = int(round((1.0 - greedy_frac) * generations))
    greedy_start = min(max(greedy_start, 0), generations)
    best_so_far = -float("inf")
    mean_fitness = jnp.float32(0.0)

    for gen in range(generations):
        rng, ask_rng, tell_rng = jax.random.split(rng, 3)
        population, es_state = strategy.ask(ask_rng, es_state, es_params)

        use_greedy = gen >= greedy_start
        bank_id = gen // refresh_every
        eval_rng = jax.random.fold_in(eval_rng_base, bank_id)
        temp_used = 1.0 if use_greedy else _temperature_for_generation(gen, generations)
        has_transition = 0 < greedy_start < generations
        roll_gen = min(gen, greedy_start) if has_transition else gen
        n_roll = _rollouts_for_generation(roll_gen, generations, num_rollouts)

        if use_greedy:
            fitnesses, success_rate, reach_water_rate, mean_final_drink = batch_greed(
                population,
                eval_rng,
                jnp.int32(n_roll),
                jnp.float32(temp_used),
            )
        else:
            fitnesses, success_rate, reach_water_rate, mean_final_drink = batch_stoch(
                population,
                eval_rng,
                jnp.int32(n_roll),
                jnp.float32(temp_used),
            )

        es_state, _ = strategy.tell(tell_rng, population, -fitnesses, es_state, es_params)

        bf = fitnesses.max()
        mf = fitnesses.mean()
        sr = success_rate.mean()
        rw = reach_water_rate.mean()
        md = mean_final_drink.mean()

        log(
            f"Gen {gen:4d} | best: {bf:.2f} | mean: {mf:.2f} | "
            f"full_meter(drink={int(DRINK_WATER_MAX_DRINK)}): {sr:.3f} | "
            f"mean_drink(end): {md:.2f} | "
            f"reached_water(manhattan<={int(DRINK_WATER_REACH_MANHATTAN_THRESHOLD)}): {rw:.3f} | "
            f"greedy={use_greedy}"
        )

        logger.log_generation(
            gen,
            fitnesses,
            extra={
                "touch/success_rate": float(sr),
                "drink_water/mean_final_drink": float(md),
                "drink_water/reached_water_within_threshold": float(rw),
                "train/greedy_mode": 1.0 if use_greedy else 0.0,
                "train/num_rollouts_used": float(n_roll),
                "train/policy_temperature_used": float(temp_used),
            },
        )

        save_checkpoint(es_state.mean, start_time, gen, float(mf), run_weights_dir)
        mf_f = float(mf)
        if mf_f > best_so_far:
            best_so_far = mf_f
            save_best_es_weights(es_state.mean, start_time, run_weights_dir)
        mean_fitness = jnp.float32(mf)

    logger.close()
    flat = es_state.mean
    mean_params = flat_params_to_pytree(flat, param_shapes, param_tree_def)
    save_final(flat, start_time, float(mean_fitness), run_weights_dir)
    log(f"Saved ES mean ({num_params} params)")
    finalize_run(run_dir, run_tb_dir, start_time, log_lines, mean_fitness=float(mean_fitness))
    return mean_params


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train drink-water policy (OpenES)")
    p.add_argument("--resume", default=None, help=".npy mean vector checkpoint")
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=DRINK_WATER_ROLLOUT_STEPS,
    )
    p.add_argument("--num-rollouts", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--variant", choices=("classic", "craftax"), default="classic")
    args = p.parse_args()

    train_drink_water(
        variant=args.variant,
        pop_size=args.pop_size,
        generations=args.generations,
        rollout_steps=args.rollout_steps,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        resume_from=args.resume,
        log_dir=args.log_dir,
    )
