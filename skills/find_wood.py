"""
OpenES training for collecting wood on Craftax-Classic-Symbolic-v1.

Observation: 12-D features from ``training.bootstrap_wood``. After ``env.reset``,
``bootstrap_wood_state`` clears mobs, zeros wood, and spawns trees in the FOV.
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
    CHEBYSHEV_R,
    DO_FACING_TREE_BONUS,
    FIND_WOOD_GREEDY_FINETUNE_FRACTION,
    FIND_WOOD_HIDDEN_SIZE,
    FIND_WOOD_LEARNING_RATE,
    FIND_WOOD_NUM_ACTIONS,
    FIND_WOOD_NUM_ROLLOUTS_ANNEAL_FRACTION,
    FIND_WOOD_NUM_ROLLOUTS_START,
    FIND_WOOD_OBS_SIZE,
    FIND_WOOD_POLICY_TEMPERATURE_ANNEAL_FRACTION,
    FIND_WOOD_POLICY_TEMPERATURE_END,
    FIND_WOOD_POLICY_TEMPERATURE_START,
    FIND_WOOD_ROLLOUT_REFRESH_EVERY,
    FIND_WOOD_ROLLOUT_STEPS,
    FIND_WOOD_STD_END,
    FIND_WOOD_STD_START,
    FIND_WOOD_STEP_PENALTY,
    MANHATTAN_THRESHOLD,
    MAX_RESOURCES,
    OBS_SIZE,
    WOOD_BONUS,
)
from policies.find_wood import FindWoodPolicy
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
from training.fitness_find_wood import make_env, make_fitness_fn
from training.params import flat_params_to_pytree, pytree_to_flat_params
from training.skill_variant import runs_policy_folder
from training.tb_logger import TBLogger


def _temperature_for_generation(gen: int, generations: int) -> float:
    if generations <= 1:
        return float(FIND_WOOD_POLICY_TEMPERATURE_END)
    p = gen / float(generations - 1)
    af = max(float(FIND_WOOD_POLICY_TEMPERATURE_ANNEAL_FRACTION), 1e-8)
    t = min(max(p / af, 0.0), 1.0)
    start = float(FIND_WOOD_POLICY_TEMPERATURE_START)
    end = float(FIND_WOOD_POLICY_TEMPERATURE_END)
    return start + (end - start) * t


def _rollouts_for_generation(gen: int, generations: int, target_rollouts: int) -> int:
    start = int(FIND_WOOD_NUM_ROLLOUTS_START)
    end = int(target_rollouts)
    start = max(1, min(start, end))
    if generations <= 1:
        return end
    p = gen / float(generations - 1)
    af = max(float(FIND_WOOD_NUM_ROLLOUTS_ANNEAL_FRACTION), 1e-8)
    frac = min(max(p / af, 0.0), 1.0)
    val = int(round(start + (end - start) * frac))
    return max(1, min(val, end))


def train_find_wood(
    variant: str = "classic",
    pop_size: int = 100,
    generations: int = 200,
    rollout_steps: int = FIND_WOOD_ROLLOUT_STEPS,
    num_rollouts: int = 128,
    seed: int = 42,
    resume_from: str | None = None,
    log_dir: str | None = None,
):
    policy_folder = runs_policy_folder("find_wood", variant)
    run_dir, start_time = get_run_dir(policy_folder)
    run_weights_dir, run_tb_dir = setup_run_dir(run_dir)

    metrics_json_path = os.path.join(run_dir, "training_metrics.json")
    logger = TBLogger(log_dir or run_tb_dir, json_path=metrics_json_path)

    rng = jax.random.PRNGKey(seed)
    env, env_params = make_env(variant=variant)

    policy = FindWoodPolicy(
        hidden_size=FIND_WOOD_HIDDEN_SIZE,
        num_actions=FIND_WOOD_NUM_ACTIONS,
    )
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(FIND_WOOD_OBS_SIZE)
    init_params = policy.init(init_rng, dummy_obs)["params"]
    flat_init, param_shapes, param_tree_def = pytree_to_flat_params(init_params)
    num_params = int(flat_init.shape[0])

    cfg_out = {
        "runs_policy_folder": policy_folder,
        "craftax_variant": variant,
        "task": "find_wood_symbolic",
        "hyperparameters": {
            "pop_size": pop_size,
            "generations": generations,
            "rollout_steps": rollout_steps,
            "num_rollouts": num_rollouts,
            "num_rollouts_start": FIND_WOOD_NUM_ROLLOUTS_START,
            "num_rollouts_anneal_fraction": FIND_WOOD_NUM_ROLLOUTS_ANNEAL_FRACTION,
            "seed": seed,
            "resume_from": resume_from,
            "hidden_size": FIND_WOOD_HIDDEN_SIZE,
            "shared_rollout_rng_across_population": True,
            "rollout_refresh_every_gens": FIND_WOOD_ROLLOUT_REFRESH_EVERY,
            "objective": "maximize_wood_collected_until_inventory_cap",
            "max_wood": int(MAX_RESOURCES),
            "wood_bonus": float(WOOD_BONUS),
            "do_facing_tree_bonus": float(DO_FACING_TREE_BONUS),
            "find_wood_step_penalty": float(FIND_WOOD_STEP_PENALTY),
            "reach_manhattan_threshold_blocks": int(MANHATTAN_THRESHOLD),
            "wood_bootstrap": {
                "fov_chebyshev_r": int(CHEBYSHEV_R),
                "num_trees_min": int(rl_cfg.FIND_WOOD_NUM_TREES_MIN),
                "num_trees_max": int(rl_cfg.FIND_WOOD_NUM_TREES_MAX),
            },
            "optimizer": "adam",
            "learning_rate": FIND_WOOD_LEARNING_RATE,
            "std_schedule": {
                "type": "linear",
                "start": float(FIND_WOOD_STD_START),
                "end": float(FIND_WOOD_STD_END),
                "transition_steps": max(int(generations) - 1, 1),
            },
            "policy_temperature_start": FIND_WOOD_POLICY_TEMPERATURE_START,
            "policy_temperature_end": FIND_WOOD_POLICY_TEMPERATURE_END,
            "policy_temperature_anneal_fraction": FIND_WOOD_POLICY_TEMPERATURE_ANNEAL_FRACTION,
            "greedy_finetune_fraction": FIND_WOOD_GREEDY_FINETUNE_FRACTION,
        },
        "policy": {
            "name": "FindWoodPolicy",
            "obs_size": FIND_WOOD_OBS_SIZE,
            "obs_size_raw_symbolic": int(OBS_SIZE),
            "feature_dim": 5,
            "num_actions": FIND_WOOD_NUM_ACTIONS,
            "action_space": "LEFT, RIGHT, UP, DOWN, DO (Craftax 1–5)",
            "hidden_size": FIND_WOOD_HIDDEN_SIZE,
            "layers": [
                f"Dense({FIND_WOOD_HIDDEN_SIZE})",
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
        f"Find-wood params={num_params} obs={FIND_WOOD_OBS_SIZE} "
        f"hidden={FIND_WOOD_HIDDEN_SIZE} actions={FIND_WOOD_NUM_ACTIONS}"
    )

    if resume_from and os.path.exists(resume_from):
        init_mean = jnp.load(resume_from)
        if int(init_mean.shape[0]) != num_params:
            raise ValueError(
                f"Checkpoint has {init_mean.shape[0]} params; FindWoodPolicy expects {num_params}."
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
        stochastic_actions=False,
        variant=variant,
    )
    batch_stoch = jax.vmap(fitness_fn_stoch, in_axes=(0, None, None, None))
    batch_greed = jax.vmap(fitness_fn_greedy, in_axes=(0, None, None, None))

    refresh_every = max(1, int(FIND_WOOD_ROLLOUT_REFRESH_EVERY))
    eval_rng_base = jax.random.PRNGKey(seed + 515_151)

    std_schedule = optax.linear_schedule(
        init_value=float(FIND_WOOD_STD_START),
        end_value=float(FIND_WOOD_STD_END),
        transition_steps=max(int(generations) - 1, 1),
    )
    strategy = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        optimizer=optax.adam(learning_rate=FIND_WOOD_LEARNING_RATE),
        std_schedule=std_schedule,
    )
    es_params = strategy.default_params
    rng, init_rng = jax.random.split(rng)
    es_state = strategy.init(init_rng, init_mean, es_params)

    greedy_frac = min(max(float(FIND_WOOD_GREEDY_FINETUNE_FRACTION), 0.0), 1.0)
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
            fitnesses, success_rate, reach_tree_rate, mean_final_wood = batch_greed(
                population,
                eval_rng,
                jnp.int32(n_roll),
                jnp.float32(temp_used),
            )
        else:
            fitnesses, success_rate, reach_tree_rate, mean_final_wood = batch_stoch(
                population,
                eval_rng,
                jnp.int32(n_roll),
                jnp.float32(temp_used),
            )

        es_state, _ = strategy.tell(tell_rng, population, -fitnesses, es_state, es_params)

        bf = fitnesses.max()
        mf = fitnesses.mean()
        sr = success_rate.mean()
        rt = reach_tree_rate.mean()
        mw = mean_final_wood.mean()

        log(
            f"Gen {gen:4d} | best: {bf:.2f} | mean: {mf:.2f} | "
            f"full_wood(wood={int(MAX_RESOURCES)}): {sr:.3f} | "
            f"mean_wood(end): {mw:.2f} | "
            f"reached_tree(manhattan<={int(MANHATTAN_THRESHOLD)}): {rt:.3f} | "
            f"greedy={use_greedy}"
        )

        logger.log_generation(
            gen,
            fitnesses,
            extra={
                "find_wood/success_rate": float(sr),
                "find_wood/mean_final_wood": float(mw),
                "find_wood/reached_tree": float(rt),
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
    p = argparse.ArgumentParser(description="Train find-wood policy (OpenES)")
    p.add_argument("--resume", default=None, help=".npy mean vector checkpoint")
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=FIND_WOOD_ROLLOUT_STEPS,
    )
    p.add_argument("--num-rollouts", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--variant", choices=("classic", "craftax"), default="classic")
    args = p.parse_args()

    train_find_wood(
        variant=args.variant,
        pop_size=args.pop_size,
        generations=args.generations,
        rollout_steps=args.rollout_steps,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        resume_from=args.resume,
        log_dir=args.log_dir,
    )
