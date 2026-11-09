#!/usr/bin/env python3
"""Thesis figure: average mode/action selection vs timestep over many rollouts.

For each timestep t, among episodes still alive, compute the fraction choosing
each coarse action group (and, for meta, each skill). Stacked area charts.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(_ROOT, "figures", "output")
DEFAULT_META = os.path.join(_ROOT, "runs/baselines_classic_meta_ppo_no_mask_20m/run_071726_003850")
DEFAULT_FLAT = os.path.join(_ROOT, "runs/baselines_classic_flat_ppo_20m/run_071626_231351")

# Coarse buckets (match action-distribution legend language).
META_GROUPS = (
    ("skills", "#93C5FD"),
    ("craft_place", "#2563EB"),
    ("move", "#64748B"),
    ("sleep", "#0F172A"),
    ("noop", "#CBD5E1"),
)
FLAT_GROUPS = (
    ("do", "#2563EB"),
    ("craft_place", "#7C3AED"),
    ("move", "#64748B"),
    ("sleep", "#0F172A"),
    ("noop", "#CBD5E1"),
)

SKILL_COLORS = {
    "wood": "#166534",
    "drink": "#0369A1",
    "food": "#B45309",
    "fight": "#B91C1C",
    "stone": "#57534E",
    "coal": "#1E293B",
    "iron": "#A8A29E",
    "diamond": "#06B6D4",
}


def _style_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_rollouts(
    *,
    policy: str,
    run_dir: str,
    num_episodes: int,
    master_seed: int,
    max_steps: int,
) -> dict:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_classic.constants import Action
    from craftax.craftax_env import make_craftax_env_from_name

    from configs.survival_meta import SURVIVAL_META_ACTION_LABELS
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
        labels = [lb.lower() for lb in SURVIVAL_META_ACTION_LABELS]
        network = ActorCritic(META_NUM_ACTIONS, int(hyp.get("layer_size", 128)))
        use_mask = bool(hyp.get("use_action_mask", False))
        sb = _to_device(build_skill_bundle(resolve_skill_paths(cfg.get("skill_weights"))))

        @jax.jit
        def run_episode(rng):
            obs0, state0 = env.reset(rng, env_params)

            def step_fn(carry, _):
                obs, state, rng_in, done_ever = carry
                pi, _ = network.apply(params, obs[None, :])
                logits = pi.logits[0]
                mode = (
                    survival_meta_greedy_mode(logits, state)
                    if use_mask
                    else jnp.argmax(logits).astype(jnp.int32)
                )
                action = meta_mode_to_env_action(
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
                rng_in, key = jax.random.split(rng_in)
                obs2, state2, _r, done, _info = env.step(key, state, action, env_params)
                done_ever2 = jnp.logical_or(done_ever, done)
                obs_out = jnp.where(done_ever, obs, obs2)
                state_out = jax.tree.map(lambda a, b: jnp.where(done_ever, a, b), state, state2)
                mode_out = jnp.where(done_ever, jnp.int32(-1), mode)
                return (obs_out, state_out, rng_in, done_ever2), mode_out

            (_, _, _, _), modes = jax.lax.scan(
                step_fn,
                (obs0, state0, rng, jnp.array(False)),
                None,
                length=max_steps,
            )
            return modes

    else:
        labels = [a.name.lower() for a in Action]
        network = ActorCritic(len(labels), int(hyp.get("layer_size", 512)))

        @jax.jit
        def run_episode(rng):
            obs0, state0 = env.reset(rng, env_params)

            def step_fn(carry, _):
                obs, state, rng_in, done_ever = carry
                pi, _ = network.apply(params, obs[None, :])
                action = jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0]
                rng_in, key = jax.random.split(rng_in)
                obs2, state2, _r, done, _info = env.step(key, state, action, env_params)
                done_ever2 = jnp.logical_or(done_ever, done)
                obs_out = jnp.where(done_ever, obs, obs2)
                state_out = jax.tree.map(lambda a, b: jnp.where(done_ever, a, b), state, state2)
                act_out = jnp.where(done_ever, jnp.int32(-1), action)
                return (obs_out, state_out, rng_in, done_ever2), act_out

            (_, _, _, _), actions = jax.lax.scan(
                step_fn,
                (obs0, state0, rng, jnp.array(False)),
                None,
                length=max_steps,
            )
            return actions

    print(f"Warming {policy} JIT ...", flush=True)
    run_episode(jax.random.PRNGKey(0))

    keys = jax.random.split(jax.random.PRNGKey(master_seed), num_episodes)
    # (E, T) action ids; -1 after death
    mat = np.full((num_episodes, max_steps), -1, dtype=np.int32)
    print(f"Rolling {num_episodes} {policy} episodes ...", flush=True)
    for i, key in enumerate(keys):
        mat[i] = np.asarray(run_episode(key), dtype=np.int32)
        if (i + 1) % 32 == 0 or i == 0:
            print(f"  [{i + 1}/{num_episodes}]", flush=True)

    n_actions = len(labels)
    # counts[t, a] among alive; alive[t]
    counts = np.zeros((max_steps, n_actions), dtype=np.float64)
    alive = np.zeros((max_steps,), dtype=np.float64)
    for t in range(max_steps):
        col = mat[:, t]
        alive_mask = col >= 0
        alive[t] = float(alive_mask.sum())
        if alive[t] == 0:
            continue
        for a in range(n_actions):
            counts[t, a] = float(np.sum(col == a))
    frac = np.divide(counts, alive[:, None], out=np.zeros_like(counts), where=alive[:, None] > 0)
    return {
        "policy": policy,
        "run_dir": os.path.abspath(run_dir),
        "labels": labels,
        "num_episodes": num_episodes,
        "master_seed": master_seed,
        "max_steps": max_steps,
        "alive_count": alive.tolist(),
        "frac": frac.tolist(),  # T x A
    }


def _meta_group_id(label: str) -> str:
    if label in SKILL_COLORS:
        return "skills"
    if label.startswith("place_") or label.startswith("craft_"):
        return "craft_place"
    if label == "sleep":
        return "sleep"
    if label == "noop":
        return "noop"
    if label in ("up", "down", "left", "right"):
        return "move"
    return "noop"


def _flat_group_id(label: str) -> str:
    u = label.upper()
    if u == "DO":
        return "do"
    if u.startswith("PLACE_") or u.startswith("MAKE_"):
        return "craft_place"
    if u == "SLEEP":
        return "sleep"
    if u == "NOOP":
        return "noop"
    if u in ("LEFT", "RIGHT", "UP", "DOWN"):
        return "move"
    return "noop"


def _group_fracs(payload: dict, group_fn, group_order: tuple[str, ...]) -> np.ndarray:
    frac = np.asarray(payload["frac"], dtype=np.float64)
    labels = payload["labels"]
    T = frac.shape[0]
    out = np.zeros((T, len(group_order)), dtype=np.float64)
    idx = {g: i for i, g in enumerate(group_order)}
    for a, lb in enumerate(labels):
        g = group_fn(lb)
        if g in idx:
            out[:, idx[g]] += frac[:, a]
    return out


def _skill_fracs(payload: dict) -> tuple[list[str], np.ndarray]:
    frac = np.asarray(payload["frac"], dtype=np.float64)
    labels = payload["labels"]
    skills = [lb for lb in labels if lb in SKILL_COLORS]
    mats = []
    for lb in skills:
        a = labels.index(lb)
        mats.append(frac[:, a])
    return skills, np.stack(mats, axis=1) if mats else np.zeros((frac.shape[0], 0))


def plot_stacked(
    ax,
    group_frac: np.ndarray,
    group_specs: tuple[tuple[str, str], ...],
    *,
    max_t: int,
    alive: np.ndarray,
    min_alive_frac: float,
) -> None:
    T = min(max_t, group_frac.shape[0])
    # Trim trailing timesteps with almost nobody alive
    n0 = alive[0] if alive[0] > 0 else 1.0
    keep = alive[:T] >= (min_alive_frac * n0)
    if not np.any(keep):
        keep = np.ones(T, dtype=bool)
    t_end = int(np.where(keep)[0][-1]) + 1
    xs = np.arange(t_end)
    ys = [group_frac[:t_end, i] * 100.0 for i in range(len(group_specs))]
    colors = [c for _, c in group_specs]
    names = [
        {
            "skills": "Skills",
            "craft_place": "Craft / place",
            "move": "Move",
            "sleep": "Sleep",
            "noop": "No-op",
            "do": "DO",
        }[g]
        for g, _ in group_specs
    ]
    ax.stackplot(xs, *ys, labels=names, colors=colors, alpha=0.92, linewidth=0)
    ax.set_xlim(0, t_end - 1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of alive episodes (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def plot_figures(
    meta: dict,
    flat: dict,
    *,
    out_base: str,
    max_plot_steps: int,
    min_alive_frac: float,
) -> None:
    _style_rc()
    meta_alive = np.asarray(meta["alive_count"], dtype=np.float64)
    flat_alive = np.asarray(flat["alive_count"], dtype=np.float64)
    meta_g = _group_fracs(meta, _meta_group_id, tuple(g for g, _ in META_GROUPS))
    flat_g = _group_fracs(flat, _flat_group_id, tuple(g for g, _ in FLAT_GROUPS))

    # --- Coarse groups: meta | flat ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150, sharey=True)
    fig.patch.set_facecolor("white")
    plot_stacked(
        axes[0],
        meta_g,
        META_GROUPS,
        max_t=max_plot_steps,
        alive=meta_alive,
        min_alive_frac=min_alive_frac,
    )
    plot_stacked(
        axes[1],
        flat_g,
        FLAT_GROUPS,
        max_t=max_plot_steps,
        alive=flat_alive,
        min_alive_frac=min_alive_frac,
    )
    axes[0].set_xlabel("Episode timestep")
    axes[1].set_xlabel("Episode timestep")
    axes[0].legend(loc="upper right", framealpha=0.95, edgecolor="#E2E8F0")
    axes[1].legend(loc="upper right", framealpha=0.95, edgecolor="#E2E8F0")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            out_base + f".{ext}",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)
    print(f"Wrote {out_base}.png / .pdf")

    # --- Meta skills only (finer) ---
    skills, skill_mat = _skill_fracs(meta)
    if skill_mat.size == 0:
        return
    fig2, ax = plt.subplots(figsize=(10.0, 4.8), dpi=150, facecolor="white")
    fig2.patch.set_facecolor("white")
    T = min(max_plot_steps, skill_mat.shape[0])
    n0 = meta_alive[0] if meta_alive[0] > 0 else 1.0
    keep = meta_alive[:T] >= (min_alive_frac * n0)
    t_end = int(np.where(keep)[0][-1]) + 1 if np.any(keep) else T
    xs = np.arange(t_end)
    ys = [skill_mat[:t_end, i] * 100.0 for i in range(len(skills))]
    colors = [SKILL_COLORS[s] for s in skills]
    display = {
        "wood": "Collect wood",
        "drink": "Find water",
        "food": "Hunt food",
        "fight": "Combat",
        "stone": "Collect stone",
        "coal": "Collect coal",
        "iron": "Collect iron",
        "diamond": "Collect diamond",
    }
    ax.stackplot(
        xs,
        *ys,
        labels=[display.get(s, s) for s in skills],
        colors=colors,
        alpha=0.92,
        linewidth=0,
    )
    ax.set_xlim(0, t_end - 1)
    ax.set_ylim(0, max(40.0, float(np.sum(skill_mat[:t_end], axis=1).max()) * 100 * 1.15))
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Share of alive episodes (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, framealpha=0.95, edgecolor="#E2E8F0")
    fig2.tight_layout()
    skill_base = out_base + "_meta_skills"
    for ext in ("png", "pdf"):
        fig2.savefig(
            skill_base + f".{ext}",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig2)
    print(f"Wrote {skill_base}.png / .pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-run", default=DEFAULT_META)
    p.add_argument("--flat-run", default=DEFAULT_FLAT)
    p.add_argument("--num-episodes", type=int, default=256)
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument(
        "--max-plot-steps",
        type=int,
        default=350,
        help="X-axis cap (still trimmed by alive fraction).",
    )
    p.add_argument(
        "--min-alive-frac",
        type=float,
        default=0.15,
        help="Stop plotting once fewer than this fraction of episodes remain alive.",
    )
    p.add_argument(
        "--cache-json",
        default=os.path.join(_ROOT, "thesis/figures/presentation_rollouts/action_over_time.json"),
    )
    p.add_argument(
        "--figure-base",
        default=os.path.join(FIG_DIR, "action_over_time_20m"),
    )
    p.add_argument(
        "--load-cache",
        action="store_true",
        help="Skip rollouts; plot from --cache-json only.",
    )
    args = p.parse_args()

    if args.load_cache and os.path.isfile(args.cache_json):
        payload = _load_json(args.cache_json)
        meta, flat = payload["meta"], payload["flat"]
    else:
        meta = collect_rollouts(
            policy="meta",
            run_dir=args.meta_run,
            num_episodes=args.num_episodes,
            master_seed=args.master_seed,
            max_steps=args.max_steps,
        )
        flat = collect_rollouts(
            policy="flat",
            run_dir=args.flat_run,
            num_episodes=args.num_episodes,
            master_seed=args.master_seed,
            max_steps=args.max_steps,
        )
        payload = {
            "note": (
                "At each timestep t, fraction is among episodes still alive "
                "(action id != -1). Same master_seed for both policies."
            ),
            "meta": meta,
            "flat": flat,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.cache_json)), exist_ok=True)
        # frac arrays are large; keep them for replot
        with open(args.cache_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"Wrote {args.cache_json}", flush=True)

    plot_figures(
        meta,
        flat,
        out_base=args.figure_base,
        max_plot_steps=args.max_plot_steps,
        min_alive_frac=args.min_alive_frac,
    )


if __name__ == "__main__":
    main()
