#!/usr/bin/env python3
"""Death-cause distribution for Craftax Classic greedy rollouts (meta vs flat).

Craftax only exposes boolean death. We classify the fatal step from state:

  lava > combat (zombie / skeleton-arrow) > needs (hunger/thirst/energy) > timeout

Needs subtypes are a bitmask of missing necessities on the pre-step state.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cause id <-> label (stable for JSON / plots)
CAUSE_LABELS = (
    "lava",
    "zombie",
    "arrow",
    "needs",
    "timeout",
    "unknown",
)
CAUSE_TO_ID = {n: i for i, n in enumerate(CAUSE_LABELS)}

DEFAULT_META_TASK = "baselines_classic_meta_ppo_no_mask_20m"
DEFAULT_FLAT_TASK = "baselines_classic_flat_ppo_20m"
DEFAULT_META = os.path.join(_ROOT, "runs", DEFAULT_META_TASK, "run_071726_003850")
DEFAULT_FLAT = os.path.join(_ROOT, "runs", DEFAULT_FLAT_TASK, "run_071626_231351")

# Collapsed buckets (same as thesis figure).
BUCKET_ORDER = ("needs", "combat", "other")
CAUSE_TO_BUCKET = {
    "needs": "needs",
    "zombie": "combat",
    "arrow": "combat",
    "lava": "other",
    "timeout": "other",
    "unknown": "other",
}


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_run_for_seed(task: str, seed: int) -> str:
    """Latest run under runs/<task> with matching train seed and params.msgpack."""
    import glob

    root = os.path.join(_ROOT, "runs", task)
    cands = []
    for d in sorted(glob.glob(os.path.join(root, "run_*"))):
        cfg_path = os.path.join(d, "config.json")
        weights = os.path.join(d, "weights", "params.msgpack")
        if not (os.path.isfile(cfg_path) and os.path.isfile(weights)):
            continue
        hyp = _load_json(cfg_path).get("hyperparameters") or {}
        if int(hyp.get("seed", -1)) == int(seed):
            cands.append(d)
    if not cands:
        raise FileNotFoundError(f"No {task} run with seed={seed} and params.msgpack")
    return cands[-1]


def _bucket_fractions(by_cause: dict) -> dict[str, float]:
    out = {b: 0.0 for b in BUCKET_ORDER}
    for cause, row in by_cause.items():
        bucket = CAUSE_TO_BUCKET.get(str(cause))
        if bucket is None:
            continue
        out[bucket] += float(row.get("fraction", 0.0))
    return out


def aggregate_across_seeds(per_seed: list[dict]) -> dict:
    """Mean ± sample std (ddof=1) of cause/bucket fractions across training seeds."""
    import statistics

    n = len(per_seed)
    cause_means: dict[str, dict] = {}
    for label in CAUSE_LABELS:
        vals = [float(s["summary"]["by_cause"][label]["fraction"]) for s in per_seed]
        cause_means[label] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.stdev(vals)) if n > 1 else 0.0,
            "per_seed": vals,
        }

    bucket_means: dict[str, dict] = {}
    for bucket in BUCKET_ORDER:
        vals = []
        for s in per_seed:
            vals.append(_bucket_fractions(s["summary"]["by_cause"])[bucket])
        bucket_means[bucket] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.stdev(vals)) if n > 1 else 0.0,
            "per_seed": vals,
        }

    death_rates = [float(s["summary"]["death_rate"]) for s in per_seed]
    lengths = [float(s["summary"]["mean_length"]) for s in per_seed]
    return {
        "num_seeds": n,
        "train_seeds": [s.get("train_seed") for s in per_seed],
        "by_cause": cause_means,
        "by_bucket": bucket_means,
        "death_rate": {
            "mean": float(statistics.fmean(death_rates)),
            "std": float(statistics.stdev(death_rates)) if n > 1 else 0.0,
        },
        "mean_length": {
            "mean": float(statistics.fmean(lengths)),
            "std": float(statistics.stdev(lengths)) if n > 1 else 0.0,
        },
    }


def classify_fatal_step(prev, nxt):
    """Return (cause_id, needs_mask) from pre/post EnvState on the terminal step."""
    import jax.numpy as jnp
    from craftax.craftax_classic.constants import BlockType

    pos = nxt.player_position
    on_lava = nxt.map[pos[0], pos[1]] == BlockType.LAVA.value
    # Timeout: episode ended without lethal health/lava (eval max_steps or env cap).
    still_alive = jnp.logical_and(nxt.player_health > 0, jnp.logical_not(on_lava))

    # Zombie melee on prev (game_logic: manh==1, mask, cooldown<=0 → dmg 2 or 7).
    z_pos = prev.zombies.position
    z_manh = jnp.abs(z_pos - prev.player_position).sum(axis=-1)
    z_hit = jnp.logical_and(
        prev.zombies.mask,
        jnp.logical_and(z_manh == 1, prev.zombies.attack_cooldown <= 0),
    )
    zombie_dmg = (
        jnp.where(prev.is_sleeping, jnp.int32(7), jnp.int32(2)) * z_hit.astype(jnp.int32).sum()
    )

    # Arrow about to land on player (proposed = pos + dir equals player).
    a_prop = prev.arrows.position + prev.arrow_directions
    a_hit = jnp.logical_and(
        prev.arrows.mask,
        (a_prop == prev.player_position).all(axis=-1),
    )
    arrow_dmg = jnp.int32(2) * a_hit.astype(jnp.int32).sum()

    combat_dmg = zombie_dmg + arrow_dmg
    health_after_combat = prev.player_health - combat_dmg

    # Needs drain: missing any necessity → recover ticks toward -15 → -1 HP.
    food0 = prev.player_food <= 0
    drink0 = prev.player_drink <= 0
    energy0 = jnp.logical_and(prev.player_energy <= 0, jnp.logical_not(prev.is_sleeping))
    missing_need = jnp.logical_or(food0, jnp.logical_or(drink0, energy0))
    # bit0=food, bit1=drink, bit2=energy
    needs_mask = (
        food0.astype(jnp.int32) + (drink0.astype(jnp.int32) << 1) + (energy0.astype(jnp.int32) << 2)
    )

    combat_lethal = health_after_combat <= 0
    # Needs can finish you if combat left you at 1 and recover triggers; or alone.
    # Approximate: died, not lava, combat not solely lethal, and a need was missing.
    needs_lethal = jnp.logical_and(missing_need, jnp.logical_not(combat_lethal))

    cause = jnp.int32(CAUSE_TO_ID["unknown"])
    cause = jnp.where(needs_lethal, jnp.int32(CAUSE_TO_ID["needs"]), cause)
    cause = jnp.where(
        jnp.logical_and(combat_lethal, arrow_dmg > zombie_dmg),
        jnp.int32(CAUSE_TO_ID["arrow"]),
        cause,
    )
    cause = jnp.where(
        jnp.logical_and(combat_lethal, zombie_dmg >= arrow_dmg),
        jnp.int32(CAUSE_TO_ID["zombie"]),
        cause,
    )
    # If both combat types and zombie_dmg == arrow_dmg > 0, zombie wins tie above.
    cause = jnp.where(on_lava, jnp.int32(CAUSE_TO_ID["lava"]), cause)
    cause = jnp.where(still_alive, jnp.int32(CAUSE_TO_ID["timeout"]), cause)
    return cause, needs_mask


def _summarize(causes: list[int], needs_masks: list[int], lengths: list[int]) -> dict:
    from collections import Counter

    n = len(causes)
    counts = Counter(causes)
    by_cause = {}
    for i, label in enumerate(CAUSE_LABELS):
        c = int(counts.get(i, 0))
        by_cause[label] = {
            "count": c,
            "fraction": (c / n) if n else 0.0,
        }
    # Needs subtypes among needs deaths
    need_rows = [m for c, m in zip(causes, needs_masks) if c == CAUSE_TO_ID["needs"]]
    subtype = Counter()
    for m in need_rows:
        parts = []
        if m & 1:
            parts.append("hunger")
        if m & 2:
            parts.append("thirst")
        if m & 4:
            parts.append("energy")
        subtype["+".join(parts) if parts else "unspecified"] += 1
    needs_n = len(need_rows)
    needs_subtypes = {
        k: {"count": v, "fraction_of_needs": (v / needs_n) if needs_n else 0.0}
        for k, v in sorted(subtype.items(), key=lambda kv: -kv[1])
    }
    return {
        "num_episodes": n,
        "by_cause": by_cause,
        "needs_subtypes": needs_subtypes,
        "mean_length": float(sum(lengths) / n) if n else 0.0,
        "death_rate": float(sum(1 for c in causes if c != CAUSE_TO_ID["timeout"]) / n)
        if n
        else 0.0,
    }


def eval_policy(
    *,
    policy: str,
    run_dir: str,
    num_episodes: int,
    master_seed: int,
    max_steps: int,
) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np  # noqa: F401 — used for rng_key_uint32
    from craftax.craftax_env import make_craftax_env_from_name

    from evaluation.benchmark_eval_official_ppo import _to_device
    from ppo.common import load_params
    from ppo.configs import META_NUM_ACTIONS, resolve_skill_paths
    from ppo.craftax_baselines.models.actor_critic import ActorCritic
    from ppo.meta_ppo import build_skill_bundle
    from ppo.skill_step import meta_mode_to_env_action
    from training.meta_action_mask import survival_meta_greedy_mode

    cfg = _load_json(os.path.join(run_dir, "config.json"))
    hyp = cfg.get("hyperparameters") or {}
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    params = _to_device(load_params(run_dir))

    if policy == "meta":
        layer_size = int(hyp.get("layer_size", 128))
        use_mask = bool(hyp.get("use_action_mask", False))
        network = ActorCritic(META_NUM_ACTIONS, layer_size)
        sb = _to_device(build_skill_bundle(resolve_skill_paths(cfg.get("skill_weights"))))

        @jax.jit
        def pick_action(obs, state):
            pi, _ = network.apply(params, obs[None, :])
            logits = pi.logits[0]
            mode = (
                survival_meta_greedy_mode(logits, state)
                if use_mask
                else jnp.argmax(logits).astype(jnp.int32)
            )
            return meta_mode_to_env_action(
                mode,
                env,
                state,
                wp=sb["wp"],
                dp=sb["dp"],
                fp=sb["fp"],
                mp=sb["mp"],
                sp=sb["sp"],
                op=sb["op"],
                h_wood=sb["h_wood"],
                h_drink=sb["h_drink"],
                h_ff=sb["h_ff"],
                h_mf=sb["h_mf"],
                h_stone=sb["h_stone"],
                h_ore=sb["h_ore"],
            )

    else:
        from craftax.craftax_classic.constants import Action

        layer_size = int(hyp.get("layer_size", 512))
        network = ActorCritic(len(Action), layer_size)

        @jax.jit
        def pick_action(obs, state):
            del state
            pi, _ = network.apply(params, obs[None, :])
            return jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0]

    @jax.jit
    def one_episode(rng):
        obs0, state0 = env.reset(rng, env_params)

        def step_fn(carry, _):
            obs, state, rng_in, length, done_ever, cause, needs_m = carry
            active = jnp.logical_not(done_ever)
            action = pick_action(obs, state)
            rng_in, key = jax.random.split(rng_in)
            obs2, state2, _r, done, _info = env.step(key, state, action, env_params)
            c_new, n_new = classify_fatal_step(state, state2)
            # Latch cause on first terminal step only.
            first_done = jnp.logical_and(done, active)
            cause_out = jnp.where(first_done, c_new, cause)
            needs_out = jnp.where(first_done, n_new, needs_m)
            length_out = length + active.astype(jnp.int32)
            done_ever2 = jnp.logical_or(done_ever, done)
            obs_out = jnp.where(done_ever, obs, obs2)
            state_out = jax.tree.map(lambda a, b: jnp.where(done_ever, a, b), state, state2)
            return (
                obs_out,
                state_out,
                rng_in,
                length_out,
                done_ever2,
                cause_out,
                needs_out,
            ), None

        init = (
            obs0,
            state0,
            rng,
            jnp.int32(0),
            jnp.bool_(False),
            jnp.int32(CAUSE_TO_ID["unknown"]),
            jnp.int32(0),
        )
        (*_, length, done_ever, cause, needs_m), _ = jax.lax.scan(
            step_fn, init, None, length=max_steps
        )
        # Hit max_steps without done → timeout
        cause = jnp.where(done_ever, cause, jnp.int32(CAUSE_TO_ID["timeout"]))
        return length, cause, needs_m, done_ever

    print(f"Warming {policy} JIT ...", flush=True)
    one_episode(jax.random.PRNGKey(0))

    keys = jax.random.split(jax.random.PRNGKey(master_seed), num_episodes)
    lengths, causes, needs, died = [], [], [], []
    print(f"Rolling {num_episodes} {policy} episodes ...", flush=True)
    for i, key in enumerate(keys):
        length, cause, needs_m, done_ever = one_episode(key)
        lengths.append(int(length))
        causes.append(int(cause))
        needs.append(int(needs_m))
        died.append(bool(done_ever) and int(cause) != CAUSE_TO_ID["timeout"])
        if (i + 1) % 32 == 0 or i == 0:
            print(f"  [{i + 1}/{num_episodes}] last_cause={CAUSE_LABELS[int(cause)]}", flush=True)

    summary = _summarize(causes, needs, lengths)
    summary["died_count"] = int(sum(died))
    return {
        "policy": policy,
        "run_dir": os.path.abspath(run_dir),
        "master_seed": master_seed,
        "max_steps": max_steps,
        "cause_labels": list(CAUSE_LABELS),
        "episodes": [
            {
                "index": i,
                "length": lengths[i],
                "cause": CAUSE_LABELS[causes[i]],
                "cause_id": causes[i],
                "needs_mask": needs[i],
                "rng_key_uint32": [int(x) for x in np.asarray(keys[i]).tolist()],
            }
            for i in range(num_episodes)
        ],
        "summary": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-run", default=DEFAULT_META)
    p.add_argument("--flat-run", default=DEFAULT_FLAT)
    p.add_argument("--meta-task", default=DEFAULT_META_TASK)
    p.add_argument("--flat-task", default=DEFAULT_FLAT_TASK)
    p.add_argument(
        "--policy",
        choices=("both", "meta", "flat"),
        default="both",
    )
    p.add_argument("--num-episodes", type=int, default=256)
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=1024)
    p.add_argument(
        "--train-seeds",
        default=None,
        help="Comma-separated training seeds for multi-seed sweep "
        "(e.g. 0,1,2,3,4). If set, discovers one run per seed.",
    )
    p.add_argument(
        "--keep-episodes",
        action="store_true",
        help="Keep per-episode records in JSON (large; off by default for sweeps).",
    )
    p.add_argument(
        "--out",
        default=os.path.join(_ROOT, "thesis/figures/presentation_rollouts/death_causes.json"),
    )
    args = p.parse_args()

    train_seeds: list[int] | None = None
    if args.train_seeds:
        train_seeds = [int(s.strip()) for s in args.train_seeds.split(",") if s.strip()]

    results: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Craftax Classic has no death-cause label; causes inferred on the fatal "
            "step with priority lava > combat(zombie/arrow) > needs > timeout. "
            "Needs subtypes from missing food/drink/energy on the pre-step state."
        ),
        "num_episodes_per_seed": args.num_episodes,
        "master_seed": args.master_seed,
        "max_steps": args.max_steps,
        "policies": {},
    }

    def _run_one(policy: str, run_dir: str, train_seed: int | None) -> dict:
        payload = eval_policy(
            policy=policy,
            run_dir=run_dir,
            num_episodes=args.num_episodes,
            master_seed=args.master_seed,
            max_steps=args.max_steps,
        )
        payload["train_seed"] = train_seed
        if not args.keep_episodes:
            payload.pop("episodes", None)
        return payload

    if train_seeds is None:
        if args.policy in ("both", "meta"):
            results["policies"]["meta"] = _run_one("meta", args.meta_run, None)
        if args.policy in ("both", "flat"):
            results["policies"]["flat"] = _run_one("flat", args.flat_run, None)
    else:
        results["train_seeds"] = train_seeds
        results["aggregation"] = "mean ± sample std (ddof=1) across training seeds"
        for policy in ("meta", "flat") if args.policy == "both" else (args.policy,):
            task = args.meta_task if policy == "meta" else args.flat_task
            per_seed = []
            for seed in train_seeds:
                run_dir = discover_run_for_seed(task, seed)
                print(f"\n>>> {policy} train_seed={seed} run={run_dir}", flush=True)
                per_seed.append(_run_one(policy, run_dir, seed))
            results["policies"][policy] = {
                "task": task,
                "per_seed": [
                    {
                        "train_seed": s["train_seed"],
                        "run_dir": s["run_dir"],
                        "summary": s["summary"],
                    }
                    for s in per_seed
                ],
                "aggregate": aggregate_across_seeds(per_seed),
            }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)

    for name, payload in results["policies"].items():
        if "aggregate" in payload:
            agg = payload["aggregate"]
            print(f"\n=== {name} (seeds {agg['train_seeds']}) ===")
            print(
                f"death_rate={100 * agg['death_rate']['mean']:.1f}%±{100 * agg['death_rate']['std']:.1f} "
                f"mean_len={agg['mean_length']['mean']:.1f}±{agg['mean_length']['std']:.1f}"
            )
            for bucket in BUCKET_ORDER:
                row = agg["by_bucket"][bucket]
                print(f"  {bucket:8s} {100 * row['mean']:.1f}% ± {100 * row['std']:.1f}%")
            continue
        s = payload["summary"]
        print(f"\n=== {name} ===")
        print(f"death_rate={s['death_rate']:.3f} mean_len={s['mean_length']:.1f}")
        for label, row in s["by_cause"].items():
            print(f"  {label:8s} {row['count']:4d}  ({100 * row['fraction']:.1f}%)")
        if s["needs_subtypes"]:
            print("  needs subtypes:")
            for k, row in s["needs_subtypes"].items():
                print(f"    {k}: {row['count']} ({100 * row['fraction_of_needs']:.1f}% of needs)")


if __name__ == "__main__":
    main()
