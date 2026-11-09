"""
OpenES training for a find-food (hunt cows) policy on Craftax-Classic-Symbolic-v1.

Same training loop shape as monster_fight_symbolic (OpenES + shared rollout RNG banks),
except bootstrap spawns cows and the handcrafted observation targets the cow mob channel.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.open_es import Open_ES

from configs import (
    FIND_FOOD_DO_FACING_COW_BONUS,
    FIND_FOOD_GREEDY_FINETUNE_FRACTION,
    FIND_FOOD_HIDDEN_SIZE,
    FIND_FOOD_KILL_REWARD,
    FIND_FOOD_LEARNING_RATE,
    FIND_FOOD_NUM_ROLLOUTS_ANNEAL_FRACTION,
    FIND_FOOD_NUM_ROLLOUTS_START,
    FIND_FOOD_OBS_SIZE,
    FIND_FOOD_POLICY_TEMPERATURE,
    FIND_FOOD_POLICY_TEMPERATURE_ANNEAL_FRACTION,
    FIND_FOOD_POLICY_TEMPERATURE_END,
    FIND_FOOD_POLICY_TEMPERATURE_START,
    FIND_FOOD_PROGRESS_ALPHA,
    FIND_FOOD_ROLLOUT_REFRESH_EVERY,
    FIND_FOOD_ROLLOUT_STEPS,
    FIND_FOOD_STD_END,
    FIND_FOOD_STD_START,
    FIND_FOOD_STEP_PENALTY,
    FIND_FOOD_TARGET_FEATURES,
    FIND_FOOD_WASTED_DO_PENALTY,
    FIND_WOOD_NUM_ACTIONS,
    OBS_SIZE,
)
from policies.find_food import FindFoodPolicy
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
from training.diagnostics_find_food import (
    ACTION_LABELS,
    action_fractions_and_entropy,
    make_action_count_fn,
)
from training.fitness_find_food import make_env, make_fitness_fn
from training.params import flat_params_to_pytree, pytree_to_flat_params
from training.skill_variant import runs_policy_folder
from training.tb_logger import TBLogger

_DIAG_NUM_ROLLOUTS = 32
_DIAG_RNG_OFFSET = 1_000_003


def _temperature_for_generation(gen: int, generations: int) -> float:
    if generations <= 1:
        return float(FIND_FOOD_POLICY_TEMPERATURE_END)
    p = gen / float(generations - 1)
    anneal_frac = max(float(FIND_FOOD_POLICY_TEMPERATURE_ANNEAL_FRACTION), 1e-8)
    t = min(max(p / anneal_frac, 0.0), 1.0)
    start = float(FIND_FOOD_POLICY_TEMPERATURE_START)
    end = float(FIND_FOOD_POLICY_TEMPERATURE_END)
    return start + (end - start) * t


def _rollouts_for_generation(gen: int, generations: int, target_rollouts: int) -> int:
    start = int(FIND_FOOD_NUM_ROLLOUTS_START)
    end = int(target_rollouts)
    start = max(1, min(start, end))
    if generations <= 1:
        return end
    p = gen / float(generations - 1)
    anneal_frac = max(float(FIND_FOOD_NUM_ROLLOUTS_ANNEAL_FRACTION), 1e-8)
    t = min(max(p / anneal_frac, 0.0), 1.0)
    val = int(round(start + (end - start) * t))
    return max(1, min(val, end))


def train_find_food(
    variant: str = "classic",
    pop_size: int = 100,
    generations: int = 100,
    rollout_steps: int = FIND_FOOD_ROLLOUT_STEPS,
    num_rollouts: int = 128,
    seed: int = 42,
    resume_from: str | None = None,
    log_dir: str | None = None,
    enable_diagnostics: bool = False,
    static_cow: bool | None = None,
    num_cows_min: int | None = None,
    num_cows_max: int | None = None,
    num_trees_min: int | None = None,
    num_trees_max: int | None = None,
):
    import configs as rl_cfg

    if static_cow is not None:
        rl_cfg.FIND_FOOD_STATIC_COW = bool(static_cow)
    if num_cows_min is not None:
        rl_cfg.FIND_FOOD_NUM_COWS_MIN = int(num_cows_min)
    if num_cows_max is not None:
        rl_cfg.FIND_FOOD_NUM_COWS_MAX = int(num_cows_max)
    if num_trees_min is not None:
        rl_cfg.FIND_FOOD_NUM_TREES_MIN = int(num_trees_min)
    if num_trees_max is not None:
        rl_cfg.FIND_FOOD_NUM_TREES_MAX = int(num_trees_max)

    run_dir, start_time = get_run_dir(runs_policy_folder("find_food", variant))
    run_weights_dir, run_tb_dir = setup_run_dir(run_dir)

    metrics_json_path = os.path.join(run_dir, "training_metrics.json")
    logger = TBLogger(log_dir or run_tb_dir, json_path=metrics_json_path)

    rng = jax.random.PRNGKey(seed)
    env, env_params = make_env(variant=variant)

    policy = FindFoodPolicy(
        hidden_size=FIND_FOOD_HIDDEN_SIZE,
        num_actions=FIND_WOOD_NUM_ACTIONS,
    )
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(FIND_FOOD_OBS_SIZE)
    init_params = policy.init(init_rng, dummy_obs)["params"]
    flat_init, param_shapes, param_tree_def = pytree_to_flat_params(init_params)
    num_params = int(flat_init.shape[0])

    config_dict = {
        "runs_policy_folder": runs_policy_folder("find_food", variant),
        "craftax_variant": variant,
        "task": "find_food_symbolic",
        "hyperparameters": {
            "pop_size": pop_size,
            "generations": generations,
            "rollout_steps": rollout_steps,
            "num_rollouts": num_rollouts,
            "num_rollouts_start": FIND_FOOD_NUM_ROLLOUTS_START,
            "num_rollouts_anneal_fraction": FIND_FOOD_NUM_ROLLOUTS_ANNEAL_FRACTION,
            "seed": seed,
            "resume_from": resume_from,
            "hidden_size": FIND_FOOD_HIDDEN_SIZE,
            "shared_rollout_rng_across_population": True,
            "shared_rollout_rng_across_rollouts": False,
            "shared_rollout_rng_across_generations": False,
            "rollout_refresh_every_gens": FIND_FOOD_ROLLOUT_REFRESH_EVERY,
            "objective_stage": "kill_cows_n_with_trees",
            "num_cows_min": int(rl_cfg.FIND_FOOD_NUM_COWS_MIN),
            "num_cows_max": int(rl_cfg.FIND_FOOD_NUM_COWS_MAX),
            "num_trees_min": int(rl_cfg.FIND_FOOD_NUM_TREES_MIN),
            "num_trees_max": int(rl_cfg.FIND_FOOD_NUM_TREES_MAX),
            "spawn_fov_chebyshev_r": int(rl_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R),
            "spawn_fov_min_chebyshev_dist": int(rl_cfg.MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST),
            "kill_reward": FIND_FOOD_KILL_REWARD,
            "do_facing_cow_bonus": FIND_FOOD_DO_FACING_COW_BONUS,
            "wasted_do_penalty": FIND_FOOD_WASTED_DO_PENALTY,
            "step_penalty": FIND_FOOD_STEP_PENALTY,
            "touch_progress_alpha": FIND_FOOD_PROGRESS_ALPHA,
            "static_cow": bool(rl_cfg.FIND_FOOD_STATIC_COW),
            "optimizer": "adam",
            "learning_rate": FIND_FOOD_LEARNING_RATE,
            "std_schedule": {
                "type": "linear",
                "start": float(FIND_FOOD_STD_START),
                "end": float(FIND_FOOD_STD_END),
                "transition_steps": max(int(generations) - 1, 1),
            },
            "action_selection": "categorical_gumbel_max",
            "policy_temperature": FIND_FOOD_POLICY_TEMPERATURE,
            "policy_temperature_start": FIND_FOOD_POLICY_TEMPERATURE_START,
            "policy_temperature_end": FIND_FOOD_POLICY_TEMPERATURE_END,
            "policy_temperature_anneal_fraction": FIND_FOOD_POLICY_TEMPERATURE_ANNEAL_FRACTION,
            "greedy_finetune_fraction": FIND_FOOD_GREEDY_FINETUNE_FRACTION,
            "use_antithetic_sampling": True,
            "fitness_shaping": "centered_rank",
            "diagnostics_enabled": bool(enable_diagnostics),
        },
        "policy": {
            "name": "FindFoodPolicy",
            "obs_size": FIND_FOOD_OBS_SIZE,
            "obs_size_raw": OBS_SIZE,
            "obs_target_features": FIND_FOOD_TARGET_FEATURES,
            "num_actions": FIND_WOOD_NUM_ACTIONS,
            "action_space": "LEFT, RIGHT, UP, DOWN, DO (Craftax 1-5)",
            "hidden_size": FIND_FOOD_HIDDEN_SIZE,
            "layers": [
                f"Dense({FIND_FOOD_HIDDEN_SIZE})",
                "ReLU",
                f"Dense({FIND_FOOD_HIDDEN_SIZE})",
                "ReLU",
                f"Dense({FIND_WOOD_NUM_ACTIONS})",
            ],
            "num_params": num_params,
        },
    }
    save_config(run_dir, config_dict)

    log_lines: list[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)
        append_training_log_line(run_dir, msg)

    log(
        f"Policy has {num_params} parameters "
        f"(hidden={FIND_FOOD_HIDDEN_SIZE}, {FIND_WOOD_NUM_ACTIONS} actions)"
    )

    if resume_from and os.path.exists(resume_from):
        init_mean = jnp.load(resume_from)
        if int(init_mean.shape[0]) != num_params:
            raise ValueError(
                f"Resume checkpoint has {init_mean.shape[0]} params but FindFoodPolicy "
                f"expects {num_params}. Match hidden_size or train fresh."
            )
        log(f"Resuming from {resume_from}")
    else:
        init_mean = jnp.zeros(num_params)

    fitness_fn_stochastic = make_fitness_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        num_rollouts=num_rollouts,
        hidden_size=FIND_FOOD_HIDDEN_SIZE,
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
        hidden_size=FIND_FOOD_HIDDEN_SIZE,
        num_actions=FIND_WOOD_NUM_ACTIONS,
        stochastic_actions=False,
        variant=variant,
    )
    batch_fitness_stochastic = jax.vmap(fitness_fn_stochastic, in_axes=(0, None, None, None))
    batch_fitness_greedy = jax.vmap(fitness_fn_greedy, in_axes=(0, None, None, None))
    refresh_every = max(1, int(FIND_FOOD_ROLLOUT_REFRESH_EVERY))
    eval_rng_base = jax.random.PRNGKey(seed + 7_777)

    std_schedule = optax.linear_schedule(
        init_value=float(FIND_FOOD_STD_START),
        end_value=float(FIND_FOOD_STD_END),
        transition_steps=max(int(generations) - 1, 1),
    )
    strategy = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        optimizer=optax.adam(learning_rate=FIND_FOOD_LEARNING_RATE),
        std_schedule=std_schedule,
    )
    es_params = strategy.default_params
    rng, init_rng = jax.random.split(rng)
    es_state = strategy.init(init_rng, init_mean, es_params)

    greedy_action_count_fn = None
    stoch_action_count_fn = None
    if enable_diagnostics:
        greedy_action_count_fn = make_action_count_fn(
            env,
            env_params,
            param_shapes,
            param_tree_def,
            rollout_steps=rollout_steps,
            hidden_size=FIND_FOOD_HIDDEN_SIZE,
            num_actions=FIND_WOOD_NUM_ACTIONS,
            stochastic=False,
            temperature=1.0,
            num_rollouts=_DIAG_NUM_ROLLOUTS,
        )
        stoch_action_count_fn = make_action_count_fn(
            env,
            env_params,
            param_shapes,
            param_tree_def,
            rollout_steps=rollout_steps,
            hidden_size=FIND_FOOD_HIDDEN_SIZE,
            num_actions=FIND_WOOD_NUM_ACTIONS,
            stochastic=True,
            temperature=FIND_FOOD_POLICY_TEMPERATURE,
            num_rollouts=_DIAG_NUM_ROLLOUTS,
        )
        diag_rng = jax.random.PRNGKey(seed + _DIAG_RNG_OFFSET)

    best_mean_fitness_so_far = -float("inf")
    greedy_fraction = float(FIND_FOOD_GREEDY_FINETUNE_FRACTION)
    greedy_fraction = min(max(greedy_fraction, 0.0), 1.0)
    greedy_start_gen = int(round((1.0 - greedy_fraction) * generations))
    greedy_start_gen = min(max(greedy_start_gen, 0), generations)

    for gen in range(generations):
        rng, ask_rng, tell_rng = jax.random.split(rng, 3)
        population, es_state = strategy.ask(ask_rng, es_state, es_params)

        use_greedy_train = gen >= greedy_start_gen
        rollout_bank_id = gen // refresh_every
        eval_rng = jax.random.fold_in(eval_rng_base, rollout_bank_id)
        policy_temperature_used = (
            1.0 if use_greedy_train else _temperature_for_generation(gen, generations)
        )
        has_greedy_transition = 0 < greedy_start_gen < generations
        rollout_curriculum_gen = min(gen, greedy_start_gen) if has_greedy_transition else gen
        num_rollouts_used = _rollouts_for_generation(
            rollout_curriculum_gen, generations, num_rollouts
        )
        if use_greedy_train:
            fitnesses, kill_rates = batch_fitness_greedy(
                population,
                eval_rng,
                jnp.int32(num_rollouts_used),
                jnp.float32(policy_temperature_used),
            )
        else:
            fitnesses, kill_rates = batch_fitness_stochastic(
                population,
                eval_rng,
                jnp.int32(num_rollouts_used),
                jnp.float32(policy_temperature_used),
            )

        es_state, _ = strategy.tell(tell_rng, population, -fitnesses, es_state, es_params)

        best_fitness = fitnesses.max()
        mean_fitness = fitnesses.mean()
        kill_rate = kill_rates.mean()

        log(
            f"Gen {gen:4d} | best: {best_fitness:.2f} | mean: {mean_fitness:.2f} "
            f"| cow_kill_rate: {kill_rate:.3f}"
        )

        extra: dict[str, float] = {
            "touch/success_rate": float(kill_rate),
            "kill/rate": float(kill_rate),
            "cow/kill_rate": float(kill_rate),
        }
        extra["train/greedy_mode"] = 1.0 if use_greedy_train else 0.0
        extra["train/policy_temperature_used"] = float(policy_temperature_used)
        extra["train/num_rollouts_used"] = float(num_rollouts_used)
        greedy_entropy = 0.0
        stoch_entropy = 0.0
        if enable_diagnostics:
            diag_rng, diag_g_rng, diag_s_rng = jax.random.split(diag_rng, 3)
            assert greedy_action_count_fn is not None and stoch_action_count_fn is not None
            greedy_counts = greedy_action_count_fn(es_state.mean, diag_g_rng)
            stoch_counts = stoch_action_count_fn(es_state.mean, diag_s_rng)
            greedy_fracs, greedy_entropy = action_fractions_and_entropy(greedy_counts)
            stoch_fracs, stoch_entropy = action_fractions_and_entropy(stoch_counts)

            extra["diag/actions_greedy/do_frac"] = float(greedy_fracs[len(ACTION_LABELS) - 1])
            extra["diag/actions_stochastic/do_frac"] = float(stoch_fracs[len(ACTION_LABELS) - 1])
            extra["diag/actions_greedy/entropy_bits"] = greedy_entropy
            extra["diag/actions_stochastic/entropy_bits"] = stoch_entropy
            for i, name in enumerate(ACTION_LABELS):
                extra[f"actions_greedy/{name}"] = float(greedy_fracs[i])
                extra[f"actions_stochastic/{name}"] = float(stoch_fracs[i])
            extra["actions_greedy/entropy_bits"] = greedy_entropy
            extra["actions_stochastic/entropy_bits"] = stoch_entropy

        logger.log_generation(gen, fitnesses, extra=extra)
        if enable_diagnostics:
            log(
                "diag | "
                f"mode={'greedy' if use_greedy_train else 'stochastic'} "
                f"g_do={extra['diag/actions_greedy/do_frac']:.3f} "
                f"s_do={extra['diag/actions_stochastic/do_frac']:.3f} "
                f"g_H={greedy_entropy:.3f} "
                f"s_H={stoch_entropy:.3f}"
            )

        save_checkpoint(es_state.mean, start_time, gen, float(mean_fitness), run_weights_dir)
        mf = float(mean_fitness)
        if mf > best_mean_fitness_so_far:
            best_mean_fitness_so_far = mf
            save_best_es_weights(es_state.mean, start_time, run_weights_dir)

    logger.close()

    mean_flat = es_state.mean
    mean_params = flat_params_to_pytree(mean_flat, param_shapes, param_tree_def)

    save_path = save_final(mean_flat, start_time, float(mean_fitness), run_weights_dir)
    log(f"Saved ES mean ({num_params:,} params) to {save_path}")

    finalize_run(run_dir, run_tb_dir, start_time, log_lines, mean_fitness=float(mean_fitness))
    return mean_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train find-food (cow hunt) policy with OpenES")
    parser.add_argument("--resume", default=None, help="Path to .npy weights to resume")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=FIND_FOOD_ROLLOUT_STEPS,
        help=f"Horizon per rollout (default {FIND_FOOD_ROLLOUT_STEPS})",
    )
    parser.add_argument("--num-rollouts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default=None, help="TensorBoard log directory")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Extra per-generation action diagnostics",
    )
    parser.add_argument(
        "--cow-mode",
        choices=("static", "moving"),
        default=None,
        help="static: frozen cow positions during fitness (see FIND_FOOD_STATIC_COW).",
    )
    parser.add_argument("--cow-min", type=int, default=None, help="FIND_FOOD_NUM_COWS_MIN")
    parser.add_argument("--cow-max", type=int, default=None, help="FIND_FOOD_NUM_COWS_MAX")
    parser.add_argument("--tree-min", type=int, default=None, help="FIND_FOOD_NUM_TREES_MIN")
    parser.add_argument("--tree-max", type=int, default=None, help="FIND_FOOD_NUM_TREES_MAX")
    parser.add_argument("--variant", choices=("classic", "craftax"), default="classic")
    args = parser.parse_args()

    train_find_food(
        variant=args.variant,
        pop_size=args.pop_size,
        generations=args.generations,
        rollout_steps=args.rollout_steps,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        resume_from=args.resume,
        log_dir=args.log_dir,
        enable_diagnostics=args.diagnostics,
        static_cow=None if args.cow_mode is None else (args.cow_mode == "static"),
        num_cows_min=args.cow_min,
        num_cows_max=args.cow_max,
        num_trees_min=args.tree_min,
        num_trees_max=args.tree_max,
    )
