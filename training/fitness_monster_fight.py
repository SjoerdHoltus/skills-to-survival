"""
Fitness evaluation for monster_fight curriculum stages 2 / 3a / 3b.

Environment setup:
- natural mob spawns disabled in `make_env`
- bootstrap injects a variable zombie count in
  [MONSTER_FIGHT_NUM_ZOMBIES_MIN, MONSTER_FIGHT_NUM_ZOMBIES_MAX] plus 0..K
  skeletons (K=`MONSTER_FIGHT_NUM_SKELETONS_MAX`) in a Chebyshev annulus
  around the player.
- when MONSTER_FIGHT_STATIC_ZOMBIE / MONSTER_FIGHT_STATIC_SKELETON each freeze
  ONLY that mob-family's POSITION — health and mask flow through naturally so the
  env's combat logic registers damage and kills as normal.

Reward per active step (NO "first-kill" gating; the agent is incentivised
all the way through the rollout):
  + MONSTER_FIGHT_KILL_REWARD * hostile_kills_this_step           one-shot per kill (zombie OR skeleton)
  + MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS * did_damage_this_step  small dense
  + MONSTER_TOUCH_PROGRESS_ALPHA * progress_to_nearest_hostile    dense shaping
  - MONSTER_FIGHT_WASTED_DO_PENALTY * (chose_DO and !did_damage)  small penalty
  - MONSTER_FIGHT_STEP_PENALTY * (any hostile still alive)        per-step cost

Returned per rollout: (score_total, kill_rate)
  score_total is the summed shaped reward over the rollout horizon.
  kill_rate is `(zombie_kills + skeleton_kills) / (initial_zombies +
  initial_skeletons)` clipped to [0, 1] — this is what the dashboards plot
  as the "task succeeded" metric.

Why this shape:
- Kill reward (10.0) dominates: a perfect rollout gets ~30 from kills and
  loses ~1.5 from per-step penalty over 30 steps, so the optimum is firmly
  "kill all zombies as fast as possible".
- DO_FACING_HOSTILE_BONUS gives ES a small but immediate signal whenever a
  random perturbation accidentally fires DO into a zombie tile. Without it,
  the kill bonus is too sparse early in training (an ES perturbation has to
  chain "approach + face + DO" before any reward fires, which is a 3-event
  conjunction at sigma>~0.1).
- WASTED_DO_PENALTY discourages random DO firing on non-targets but is kept
  small (<< the bonus from a real hit) so it doesn't crowd out exploration.
- Per-step penalty applies as long as ANY zombie is still alive — once all
  are dead the surface goes flat (no incentive to wander or to camp), same
  philosophy as stage 1's "flat after first touch" trick.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import configs as mf_cfg
from configs import (
    FIND_WOOD_NUM_ACTIONS,
    MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS,
    MONSTER_FIGHT_HIDDEN_SIZE,
    MONSTER_FIGHT_KILL_REWARD,
    MONSTER_FIGHT_POLICY_TEMPERATURE,
    MONSTER_FIGHT_ROLLOUT_STEPS,
    MONSTER_FIGHT_STEP_PENALTY,
    MONSTER_FIGHT_WASTED_DO_PENALTY,
    MONSTER_TOUCH_PROGRESS_ALPHA,
)
from policies.monster_fight import MonsterFightPolicy
from training.monster_world_bootstrap import (
    bootstrap_monster_fight_state,
    make_monster_fight_obs,
    rebuild_mob_map_from_state,
)
from training.params import flat_params_to_pytree
from training.skill_variant import VARIANT_CRAFTAX, normalize_variant

_BIG_DIST = jnp.asarray(1_000_000, dtype=jnp.int32)
# Action enum value for DO (Craftax-Classic: NOOP=0,LEFT=1,RIGHT=2,UP=3,DOWN=4,DO=5).
# We compare the actual env-action (1-indexed) rather than the local 0..4 logit
# index so this stays correct if action ordering ever changes.
_DO_ACTION_VALUE = jnp.int32(5)


def _min_manhattan_to_active_hostiles(
    player_pos: jnp.ndarray,
    z_pos: jnp.ndarray,
    z_mask: jnp.ndarray,
    s_pos: jnp.ndarray,
    s_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (min_dist, any_hostile) over active zombies+skeletons (Manhattan).

    Inactive slots are set to a large constant so they never dominate the min.
    The shaping reward in the rollout uses this so progress is computed
    relative to whichever hostile (zombie or skeleton) is currently nearest.
    """
    p = player_pos.astype(jnp.int32)
    zd = jnp.abs(z_pos.astype(jnp.int32) - p).sum(axis=-1)
    zd = jnp.where(z_mask, zd, _BIG_DIST)
    sd = jnp.abs(s_pos.astype(jnp.int32) - p).sum(axis=-1)
    sd = jnp.where(s_mask, sd, _BIG_DIST)
    all_d = jnp.concatenate([zd, sd], axis=0)
    any_hostile = jnp.any(z_mask) | jnp.any(s_mask)
    return jnp.min(all_d), any_hostile


def make_single_rollout_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = MONSTER_FIGHT_ROLLOUT_STEPS,
    hidden_size: int = MONSTER_FIGHT_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "classic",
    melee_type_id: int = 0,
):
    """JIT-compiled single rollout. Returns (score_total, kill_rate)."""
    if normalize_variant(variant) == VARIANT_CRAFTAX:
        from training.craftax import fitness_monster_fight as cf

        static = bool(mf_cfg.MONSTER_FIGHT_STATIC_ZOMBIE)
        return cf.make_single_rollout_fn(
            env,
            env_params,
            param_shapes,
            param_tree_def,
            rollout_steps=rollout_steps,
            hidden_size=hidden_size,
            num_actions=num_actions,
            stochastic_actions=stochastic_actions,
            variant=variant,
            static_melee=static,
            melee_type_id=int(melee_type_id),
        )

    @jax.jit
    def single_rollout(flat_params: jnp.ndarray, rng: jax.Array, policy_temperature: jnp.ndarray):
        params = flat_params_to_pytree(flat_params, param_shapes, param_tree_def)

        rng, reset_rng, boot_rng = jax.random.split(rng, 3)
        _, state = env.reset(reset_rng, env_params)
        state = bootstrap_monster_fight_state(state, boot_rng, env_params)

        # Total hostiles present at rollout start (zombies + skeletons),
        # used as denominator for `kill_rate`. Stays constant through the
        # rollout because we never spawn new mobs (env spawn-chances
        # zeroed in `make_env`).
        initial_alive_z = jnp.sum(state.zombies.mask).astype(jnp.float32)
        initial_alive_s = jnp.sum(state.skeletons.mask).astype(jnp.float32)
        initial_alive = initial_alive_z + initial_alive_s
        kill_rate_denom = jnp.maximum(initial_alive, jnp.float32(1.0))

        obs = make_monster_fight_obs(env, state)
        episode_done = jnp.array(False)

        def step_fn(carry, _):
            obs, state, rng, episode_done = carry
            rng, policy_rng, env_step_rng = jax.random.split(rng, 3)

            logits = MonsterFightPolicy(hidden_size=hidden_size, num_actions=num_actions).apply(
                {"params": params}, obs
            )
            if stochastic_actions:
                temp = jnp.maximum(jnp.float32(policy_temperature), jnp.float32(1e-6))
                scaled_logits = logits / temp
                local_action = jax.random.categorical(policy_rng, scaled_logits)
            else:
                local_action = jnp.argmax(logits)
            action = local_action + 1  # 0..4 -> LEFT..DO (env uses 1..5)

            # Pre-step snapshots used for shaping / kill detection.
            prev_z = state.zombies
            prev_s = state.skeletons
            prev_player_pos = state.player_position.astype(jnp.int32)
            prev_min_d, prev_any_hostile = _min_manhattan_to_active_hostiles(
                prev_player_pos,
                prev_z.position,
                prev_z.mask,
                prev_s.position,
                prev_s.mask,
            )
            prev_health_sum_z = prev_z.health.astype(jnp.int32).sum()
            prev_health_sum_s = prev_s.health.astype(jnp.int32).sum()

            new_obs, new_state, _reward, done, _info = env.step(
                env_step_rng, state, action, env_params
            )

            # Capture kill events / damage dealt BEFORE static-hostile freezes
            # so position-only freezes below don't hide health changes.
            #
            # A "kill" requires THREE conditions, not just a mask flip:
            #   1. the slot was alive before the step
            #   2. the slot is dead after the step
            #   3. the slot's health dropped to <= 0 in this step
            # Condition (3) defends against natural despawns (Craftax flips
            # `mask` to False when a mob wanders past `mob_despawn_distance`
            # without changing its health). We additionally raise that
            # distance to ~infinity in `make_env` so this should never fire,
            # but keeping the check here means metric integrity doesn't
            # depend on env_params staying configured.
            post_z = new_state.zombies
            post_s = new_state.skeletons
            z_mask_flip = jnp.logical_and(prev_z.mask, jnp.logical_not(post_z.mask))
            z_dead = post_z.health.astype(jnp.int32) <= 0
            z_kills = jnp.logical_and(z_mask_flip, z_dead).astype(jnp.int32).sum()
            s_mask_flip = jnp.logical_and(prev_s.mask, jnp.logical_not(post_s.mask))
            s_dead = post_s.health.astype(jnp.int32) <= 0
            s_kills = jnp.logical_and(s_mask_flip, s_dead).astype(jnp.int32).sum()
            num_kills_this_step = (z_kills + s_kills).astype(jnp.float32)

            new_health_sum_z = post_z.health.astype(jnp.int32).sum()
            new_health_sum_s = post_s.health.astype(jnp.int32).sum()
            damage_dealt = jnp.maximum(
                (prev_health_sum_z - new_health_sum_z) + (prev_health_sum_s - new_health_sum_s),
                0,
            ).astype(jnp.float32)
            did_damage = damage_dealt > 0

            if mf_cfg.MONSTER_FIGHT_STATIC_ZOMBIE or mf_cfg.MONSTER_FIGHT_STATIC_SKELETON:
                post_z = new_state.zombies
                post_s = new_state.skeletons
                if mf_cfg.MONSTER_FIGHT_STATIC_ZOMBIE:
                    lock_z = prev_z.mask  # alive going into this step
                    frozen_z = post_z.replace(
                        position=jnp.where(
                            lock_z[:, None],
                            prev_z.position.astype(post_z.position.dtype),
                            post_z.position,
                        ),
                    )
                else:
                    frozen_z = post_z
                if mf_cfg.MONSTER_FIGHT_STATIC_SKELETON:
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
                new_state = new_state.replace(zombies=frozen_z, skeletons=frozen_s)
                new_state = new_state.replace(mob_map=rebuild_mob_map_from_state(new_state))

            new_obs = make_monster_fight_obs(env, new_state)

            active = jnp.logical_not(episode_done)
            active_f = active.astype(jnp.float32)

            # Distance shaping (still useful while approaching the next target).
            #
            # Critical: this telescoping shape only works when the SAME
            # hostile is the "nearest" in both the pre- and post-step
            # snapshots. When a kill event happens, the helper re-targets
            # to the next-nearest live hostile (zombie OR skeleton), so phi
            # can jump discontinuously and produce a LARGE NEGATIVE shaping
            # reward that fights against the (positive) kill_reward. We zero
            # out progress on any step that contains a kill, so the kill
            # bonus alone shapes the kill event and the dense reward
            # continues to do its job between kills.
            new_player_pos = new_state.player_position.astype(jnp.int32)
            new_min_d, any_hostile_now = _min_manhattan_to_active_hostiles(
                new_player_pos,
                new_state.zombies.position,
                new_state.zombies.mask,
                new_state.skeletons.position,
                new_state.skeletons.mask,
            )
            prev_phi = 1.0 / (prev_min_d.astype(jnp.float32) + 1.0)
            new_phi = 1.0 / (new_min_d.astype(jnp.float32) + 1.0)
            no_kill_this_step = jnp.logical_not(num_kills_this_step > 0)
            both_have_hostile = (prev_any_hostile & any_hostile_now & no_kill_this_step).astype(
                jnp.float32
            )
            progress = (new_phi - prev_phi) * both_have_hostile

            chose_do = action == _DO_ACTION_VALUE
            wasted_do = jnp.logical_and(chose_do, jnp.logical_not(did_damage))
            do_bonus = jnp.logical_and(chose_do, did_damage).astype(jnp.float32)
            wasted_do_f = wasted_do.astype(jnp.float32)

            # Per-step penalty applies only while there is still a hostile
            # to kill — once all are dead, the reward surface is flat and
            # the policy has no incentive to keep moving.
            any_alive_f = any_hostile_now.astype(jnp.float32)

            step_score = active_f * (
                MONSTER_FIGHT_KILL_REWARD * num_kills_this_step
                + MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS * do_bonus
                + MONSTER_TOUCH_PROGRESS_ALPHA * progress
                - MONSTER_FIGHT_WASTED_DO_PENALTY * wasted_do_f
                - MONSTER_FIGHT_STEP_PENALTY * any_alive_f
            )

            # Only count kills the agent banked while still alive. Without
            # this gate, with `auto_reset=False`, the env keeps stepping
            # after `done` (the policy keeps argmax-acting, the env keeps
            # responding) and the dead player can still register hostile
            # damage via DO — those kills are spurious for the metric.
            kills_for_metric = num_kills_this_step * active_f
            episode_done = jnp.logical_or(episode_done, done)
            return (new_obs, new_state, rng, episode_done), (
                step_score,
                kills_for_metric,
            )

        (_, final_state, _, _), step_outs = jax.lax.scan(
            step_fn,
            (obs, state, rng, episode_done),
            None,
            length=rollout_steps,
        )

        step_scores, step_kills = step_outs
        total_kills = step_kills.sum()
        kill_rate = jnp.clip(total_kills / kill_rate_denom, 0.0, 1.0)
        return step_scores.sum(), kill_rate

    return single_rollout


def make_fitness_fn(
    env,
    env_params,
    param_shapes,
    param_tree_def,
    rollout_steps: int = MONSTER_FIGHT_ROLLOUT_STEPS,
    num_rollouts: int = 256,
    hidden_size: int = MONSTER_FIGHT_HIDDEN_SIZE,
    num_actions: int = FIND_WOOD_NUM_ACTIONS,
    stochastic_actions: bool = True,
    variant: str = "classic",
    melee_type_id: int = 0,
):
    """Fitness for one individual: mean over ``num_rollouts``."""
    if normalize_variant(variant) == VARIANT_CRAFTAX:
        from training.craftax import fitness_monster_fight as cf

        static = bool(mf_cfg.MONSTER_FIGHT_STATIC_ZOMBIE)
        return cf.make_fitness_fn(
            env,
            env_params,
            param_shapes,
            param_tree_def,
            rollout_steps=rollout_steps,
            num_rollouts=num_rollouts,
            hidden_size=hidden_size,
            num_actions=num_actions,
            stochastic_actions=stochastic_actions,
            variant=variant,
            static_melee=static,
            melee_type_id=int(melee_type_id),
        )

    single_rollout = make_single_rollout_fn(
        env,
        env_params,
        param_shapes,
        param_tree_def,
        rollout_steps=rollout_steps,
        hidden_size=hidden_size,
        num_actions=num_actions,
        stochastic_actions=stochastic_actions,
        variant=variant,
    )
    vmapped_rollouts = jax.vmap(single_rollout, in_axes=(None, 0, None))

    @jax.jit
    def fitness_fn(
        flat_params: jnp.ndarray,
        rng: jax.Array,
        active_num_rollouts: jnp.ndarray = jnp.int32(num_rollouts),
        policy_temperature: jnp.ndarray = jnp.float32(MONSTER_FIGHT_POLICY_TEMPERATURE),
    ):
        rollout_rngs = jax.random.split(rng, num_rollouts)
        scores, kill_rates = vmapped_rollouts(flat_params, rollout_rngs, policy_temperature)
        n = jnp.clip(active_num_rollouts.astype(jnp.int32), 1, jnp.int32(num_rollouts))
        idx = jnp.arange(num_rollouts, dtype=jnp.int32)
        mask = (idx < n).astype(jnp.float32)
        denom = jnp.maximum(mask.sum(), 1.0)
        score_mean = (scores * mask).sum() / denom
        kill_mean = (kill_rates * mask).sum() / denom
        return score_mean, kill_mean

    return fitness_fn


def make_env(variant: str = "classic"):
    from training.skill_backends import get_backend

    return get_backend(variant).make_env()
