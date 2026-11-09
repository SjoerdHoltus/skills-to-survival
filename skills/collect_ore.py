"""
OpenES training for collect-ore (single training target; obs encodes deploy target).

Train on coal by default; reuse weights for iron at eval by changing bootstrap target_id.

Train:
  python -m skills.collect_ore --generations 200 --num-rollouts 128

Evaluate (other targets / wall depth):
  python -m evaluation.eval_collect_ore \\
    --weights runs/collect_ore/run_xxx/weights/final_....npy --target iron --wall-tiles 1
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
    COLLECT_ORE_BONUS,
    COLLECT_ORE_DO_FACING_ORE_BONUS,
    COLLECT_ORE_DO_FACING_STONE_BONUS,
    COLLECT_ORE_GREEDY_FINETUNE_FRACTION,
    COLLECT_ORE_HIDDEN_SIZE,
    COLLECT_ORE_LEARNING_RATE,
    COLLECT_ORE_MAX_COAL,
    COLLECT_ORE_MAX_IRON,
    COLLECT_ORE_NUM_ACTIONS,
    COLLECT_ORE_NUM_ROLLOUTS_ANNEAL_FRACTION,
    COLLECT_ORE_NUM_ROLLOUTS_START,
    COLLECT_ORE_OBS_SIZE,
    COLLECT_ORE_POLICY_TEMPERATURE_ANNEAL_FRACTION,
    COLLECT_ORE_POLICY_TEMPERATURE_END,
    COLLECT_ORE_POLICY_TEMPERATURE_START,
    COLLECT_ORE_REACH_MANHATTAN_THRESHOLD,
    COLLECT_ORE_ROLLOUT_REFRESH_EVERY,
    COLLECT_ORE_ROLLOUT_STEPS,
    COLLECT_ORE_STD_END,
    COLLECT_ORE_STD_START,
    COLLECT_ORE_STEP_PENALTY,
    COLLECT_ORE_TARGET_FEATURES,
    COLLECT_ORE_TARGET_NAMES,
    COLLECT_ORE_TRAINING_TARGET,
    COLLECT_ORE_WALL_BREAK_BONUS,
    COLLECT_ORE_WALL_CURRICULUM_PHASE1_FRACTION,
    COLLECT_ORE_WALL_CURRICULUM_PHASE2_FRACTION,
    OBS_SIZE,
)
from policies.collect_ore import CollectOrePolicy
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
from training.fitness_collect_ore import make_env, make_fitness_fn
from training.params import flat_params_to_pytree, pytree_to_flat_params
from training.skill_variant import runs_policy_folder
from training.tb_logger import TBLogger


def _temperature_for_generation(gen: int, generations: int) -> float:
    if generations <= 1:
        return float(COLLECT_ORE_POLICY_TEMPERATURE_END)
    p = gen / float(generations - 1)
    af = max(float(COLLECT_ORE_POLICY_TEMPERATURE_ANNEAL_FRACTION), 1e-8)
    t = min(max(p / af, 0.0), 1.0)
    start = float(COLLECT_ORE_POLICY_TEMPERATURE_START)
    end = float(COLLECT_ORE_POLICY_TEMPERATURE_END)
    return start + (end - start) * t


def _rollouts_for_generation(gen: int, generations: int, target_rollouts: int) -> int:
    start = int(COLLECT_ORE_NUM_ROLLOUTS_START)
    end = int(target_rollouts)
    start = max(1, min(start, end))
    if generations <= 1:
        return end
    p = gen / float(generations - 1)
    af = max(float(COLLECT_ORE_NUM_ROLLOUTS_ANNEAL_FRACTION), 1e-8)
    frac = min(max(p / af, 0.0), 1.0)
    val = int(round(start + (end - start) * frac))
    return max(1, min(val, end))


def train_collect_ore(
    variant: str = "classic",
    pop_size: int = 100,
    generations: int = 200,
    rollout_steps: int = COLLECT_ORE_ROLLOUT_STEPS,
    num_rollouts: int = 128,
    seed: int = 42,
    resume_from: str | None = None,
    log_dir: str | None = None,
):
    run_dir, start_time = get_run_dir(runs_policy_folder("collect_ore", variant))
    run_weights_dir, run_tb_dir = setup_run_dir(run_dir)

    metrics_json_path = os.path.join(run_dir, "training_metrics.json")
    logger = TBLogger(log_dir or run_tb_dir, json_path=metrics_json_path)

    rng = jax.random.PRNGKey(seed)
    env, env_params = make_env(variant=variant)

    policy = CollectOrePolicy(
        hidden_size=COLLECT_ORE_HIDDEN_SIZE,
        num_actions=COLLECT_ORE_NUM_ACTIONS,
    )
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(COLLECT_ORE_OBS_SIZE)
    init_params = policy.init(init_rng, dummy_obs)["params"]
    flat_init, param_shapes, param_tree_def = pytree_to_flat_params(init_params)
    num_params = int(flat_init.shape[0])

    cfg_out = {
        "runs_policy_folder": runs_policy_folder("collect_ore", variant),
        "craftax_variant": variant,
        "task": "collect_ore_symbolic",
        "hyperparameters": {
            "pop_size": pop_size,
            "generations": generations,
            "rollout_steps": rollout_steps,
            "num_rollouts": num_rollouts,
            "num_rollouts_start": COLLECT_ORE_NUM_ROLLOUTS_START,
            "num_rollouts_anneal_fraction": COLLECT_ORE_NUM_ROLLOUTS_ANNEAL_FRACTION,
            "seed": seed,
            "resume_from": resume_from,
            "hidden_size": COLLECT_ORE_HIDDEN_SIZE,
            "shared_rollout_rng_across_population": True,
            "rollout_refresh_every_gens": COLLECT_ORE_ROLLOUT_REFRESH_EVERY,
            "objective": "walled_single_target",
            "training_target": COLLECT_ORE_TRAINING_TARGET,
            "deploy_targets": list(COLLECT_ORE_TARGET_NAMES),
            "max_coal": int(COLLECT_ORE_MAX_COAL),
            "max_iron": int(COLLECT_ORE_MAX_IRON),
            "max_diamond": int(rl_cfg.COLLECT_ORE_MAX_DIAMOND),
            "collect_ore_bonus": float(COLLECT_ORE_BONUS),
            "step_penalty": float(COLLECT_ORE_STEP_PENALTY),
            "do_facing_stone_bonus": float(COLLECT_ORE_DO_FACING_STONE_BONUS),
            "do_facing_ore_bonus": float(COLLECT_ORE_DO_FACING_ORE_BONUS),
            "wall_break_bonus": float(COLLECT_ORE_WALL_BREAK_BONUS),
            "reach_manhattan_threshold_blocks": int(COLLECT_ORE_REACH_MANHATTAN_THRESHOLD),
            "wall_curriculum_phase1_fraction": float(COLLECT_ORE_WALL_CURRICULUM_PHASE1_FRACTION),
            "wall_curriculum_phase2_fraction": float(COLLECT_ORE_WALL_CURRICULUM_PHASE2_FRACTION),
            "ore_bootstrap": {
                "ore_distance_min": int(rl_cfg.COLLECT_ORE_ORE_DISTANCE_MIN),
                "ore_distance_max": int(rl_cfg.COLLECT_ORE_ORE_DISTANCE_MAX),
                "dual_pickaxe": True,
            },
            "optimizer": "adam",
            "learning_rate": COLLECT_ORE_LEARNING_RATE,
            "std_schedule": {
                "type": "linear",
                "start": float(COLLECT_ORE_STD_START),
                "end": float(COLLECT_ORE_STD_END),
                "transition_steps": max(int(generations) - 1, 1),
            },
            "policy_temperature_start": COLLECT_ORE_POLICY_TEMPERATURE_START,
            "policy_temperature_end": COLLECT_ORE_POLICY_TEMPERATURE_END,
            "policy_temperature_anneal_fraction": COLLECT_ORE_POLICY_TEMPERATURE_ANNEAL_FRACTION,
            "greedy_finetune_fraction": COLLECT_ORE_GREEDY_FINETUNE_FRACTION,
        },
        "policy": {
            "name": "CollectOrePolicy",
            "obs_size": COLLECT_ORE_OBS_SIZE,
            "obs_size_raw_symbolic": int(OBS_SIZE),
            "feature_dim": COLLECT_ORE_TARGET_FEATURES,
            "blocker_features": 4,
            "num_actions": COLLECT_ORE_NUM_ACTIONS,
            "hidden_size": COLLECT_ORE_HIDDEN_SIZE,
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
        f"Collect-ore params={num_params} obs={COLLECT_ORE_OBS_SIZE} "
        f"hidden={COLLECT_ORE_HIDDEN_SIZE} train_target={COLLECT_ORE_TRAINING_TARGET} "
        f"deploy_targets={COLLECT_ORE_TARGET_NAMES}"
    )

    if resume_from and os.path.exists(resume_from):
        init_mean = jnp.load(resume_from)
        if int(init_mean.shape[0]) != num_params:
            raise ValueError(
                f"Checkpoint has {init_mean.shape[0]} params; CollectOrePolicy expects {num_params}."
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
    batch_stoch = jax.vmap(fitness_fn_stoch, in_axes=(0, None, None, None, None, None))
    batch_greed = jax.vmap(fitness_fn_greedy, in_axes=(0, None, None, None, None, None))

    refresh_every = max(1, int(COLLECT_ORE_ROLLOUT_REFRESH_EVERY))
    eval_rng_base = jax.random.PRNGKey(seed + 737_373)

    std_schedule = optax.linear_schedule(
        init_value=float(COLLECT_ORE_STD_START),
        end_value=float(COLLECT_ORE_STD_END),
        transition_steps=max(int(generations) - 1, 1),
    )
    strategy = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        optimizer=optax.adam(learning_rate=COLLECT_ORE_LEARNING_RATE),
        std_schedule=std_schedule,
    )
    es_params = strategy.default_params
    rng, init_rng = jax.random.split(rng)
    es_state = strategy.init(init_rng, init_mean, es_params)

    greedy_frac = min(max(float(COLLECT_ORE_GREEDY_FINETUNE_FRACTION), 0.0), 1.0)
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
        cur_gen = jnp.int32(gen)
        cur_gens = jnp.int32(generations)

        if use_greedy:
            out = batch_greed(
                population,
                eval_rng,
                jnp.int32(n_roll),
                jnp.float32(temp_used),
                cur_gen,
                cur_gens,
            )
        else:
            out = batch_stoch(
                population,
                eval_rng,
                jnp.int32(n_roll),
                jnp.float32(temp_used),
                cur_gen,
                cur_gens,
            )

        fitnesses = out[0]
        reach_rate = out[2]
        mean_final_amt = out[3]
        mined_rate = out[4]
        shell_depth = out[5]

        es_state, _ = strategy.tell(tell_rng, population, -fitnesses, es_state, es_params)

        bf = fitnesses.max()
        mf = fitnesses.mean()
        rr = reach_rate.mean()
        ma = mean_final_amt.mean()
        mr = mined_rate.mean()

        log(
            f"Gen {gen:4d} | best: {bf:.2f} | mean: {mf:.2f} | "
            f"shell: {shell_depth.mean():.2f} | mined: {mr:.3f} | reach: {rr:.3f} | "
            f"amount: {ma:.2f} | "
            f"target={COLLECT_ORE_TRAINING_TARGET} | "
            f"rollouts={n_roll} temp={temp_used:.3f} greedy={use_greedy}"
        )

        logger.log_generation(
            gen,
            fitnesses,
            extra={
                "collect_ore/mean_final_amount": float(ma),
                "collect_ore/reached_target": float(rr),
                "collect_ore/mined_rate": float(mr),
                "collect_ore/mean_shell_depth": float(shell_depth.mean()),
                "collect_ore/training_target_id": float(rl_cfg.COLLECT_ORE_TRAINING_TARGET_ID),
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
    p = argparse.ArgumentParser(
        description="Train collect-ore policy (OpenES, single training target)"
    )
    p.add_argument("--resume", default=None, help=".npy mean vector checkpoint")
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--rollout-steps", type=int, default=COLLECT_ORE_ROLLOUT_STEPS)
    p.add_argument("--num-rollouts", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--variant", choices=("classic", "craftax"), default="classic")
    args = p.parse_args()

    train_collect_ore(
        variant=args.variant,
        pop_size=args.pop_size,
        generations=args.generations,
        rollout_steps=args.rollout_steps,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        resume_from=args.resume,
        log_dir=args.log_dir,
    )
