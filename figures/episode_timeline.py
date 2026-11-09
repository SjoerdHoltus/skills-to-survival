#!/usr/bin/env python3
"""Thesis figure: skill/action swimlanes for one greedy episode.

Rolls one flat and one meta episode (or reuses --from-trace), then draws
Gantt-style swimlanes: one row per skill/mode group, bars = active intervals.
"""

from __future__ import annotations

import argparse
import json
import os


def _configure_jax(cpu: bool) -> None:
    if cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")

DEFAULT_FLAT_RUN = os.path.join(
    _PROJECT_ROOT,
    "runs/baselines_classic_flat_ppo_20m/run_071626_231351",
)
DEFAULT_META_RUN = os.path.join(
    _PROJECT_ROOT,
    "runs/baselines_classic_meta_ppo_no_mask_20m/run_071726_003850",
)

FLAT_COLOR = "#2563EB"
META_SKILL_COLOR = "#93C5FD"
META_NOOP_COLOR = "#94A3B8"
META_SKILLS = frozenset({"wood", "drink", "food", "fight", "stone", "coal", "iron", "diamond"})


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _style_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def _trace_from_scan(
    *,
    policy: str,
    run_dir: str,
    seed: int,
    labels_lookup: list[str],
    actions_or_modes,
    env_actions,
    unlocks,
    done_flags,
    extra: dict | None = None,
) -> dict:
    import numpy as np

    arr = np.asarray(actions_or_modes)
    unlocks_np = np.asarray(unlocks)
    done_np = np.asarray(done_flags)
    env_np = None if env_actions is None else np.asarray(env_actions)

    if done_np.any():
        length = int(np.argmax(done_np)) + 1
    else:
        length = int(arr.shape[0])

    steps = []
    unlock_events = []
    for t in range(length):
        idx = int(arr[t])
        if idx < 0:
            break
        row = {"step": t, "label": labels_lookup[idx]}
        if policy == "meta":
            row["mode"] = idx
            row["env_action"] = int(env_np[t])
        else:
            row["action"] = idx
        steps.append(row)
        for ach_i in np.flatnonzero(unlocks_np[t]):
            unlock_events.append({"step": t, "achievement_idx": int(ach_i)})

    out = {
        "policy": policy,
        "run_dir": os.path.abspath(run_dir),
        "seed": seed,
        "length": len(steps),
        "labels": [s["label"] for s in steps],
        "steps": steps,
        "achievement_unlocks": unlock_events,
    }
    if extra:
        out.update(extra)
    return out


def rollout_flat_trace(run_dir: str, *, seed: int, max_steps: int) -> dict:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_classic.constants import Action
    from craftax.craftax_env import make_craftax_env_from_name

    from evaluation.benchmark_eval_official_ppo import (
        _episode_achievements_jax,
        _to_device,
    )
    from ppo.common import load_params
    from ppo.craftax_baselines.models.actor_critic import ActorCritic

    print(f"Loading flat params from {run_dir} ...", flush=True)
    cfg = _load_json(os.path.join(run_dir, "config.json"))
    hyp = cfg.get("hyperparameters") or {}
    env_name = hyp.get("env_name", "Craftax-Classic-Symbolic-v1")
    layer_size = int(hyp.get("layer_size", 512))
    labels = [a.name.lower() for a in Action]
    num_ach = 22

    params = _to_device(load_params(run_dir))
    network = ActorCritic(len(labels), layer_size)
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params

    @jax.jit
    def run_episode(rng):
        obs0, state0 = env.reset(rng, env_params)
        ach0 = jnp.zeros((num_ach,), dtype=jnp.float32)

        def step_fn(carry, _):
            obs, state, rng_in, ach, done_ever = carry
            pi, _ = network.apply(params, obs[None, :])
            action = jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0]
            rng_in, key = jax.random.split(rng_in)
            obs2, state2, _r, done, _info = env.step(key, state, action, env_params)
            ach2 = jnp.maximum(ach, _episode_achievements_jax(state2, num_ach))
            new_unlock = jnp.logical_and(ach2 > 0, ach <= 0)
            done_ever2 = jnp.logical_or(done_ever, done)
            obs_out = jnp.where(done_ever, obs, obs2)
            state_out = jax.tree.map(lambda a, b: jnp.where(done_ever, a, b), state, state2)
            ach_out = jnp.where(done_ever, ach, ach2)
            action_out = jnp.where(done_ever, jnp.int32(-1), action)
            unlock_out = jnp.where(done_ever, jnp.zeros_like(new_unlock), new_unlock)
            return (obs_out, state_out, rng_in, ach_out, done_ever2), (
                action_out,
                unlock_out,
                done_ever2,
            )

        (_, _, _, _, _), (actions, unlocks, done_flags) = jax.lax.scan(
            step_fn,
            (obs0, state0, rng, ach0, jnp.array(False)),
            None,
            length=max_steps,
        )
        return actions, unlocks, done_flags

    print("Rolling flat episode (JIT) ...", flush=True)
    actions, unlocks, done_flags = run_episode(jax.random.PRNGKey(seed))
    trace = _trace_from_scan(
        policy="flat",
        run_dir=run_dir,
        seed=seed,
        labels_lookup=labels,
        actions_or_modes=actions,
        env_actions=None,
        unlocks=unlocks,
        done_flags=done_flags,
    )
    print(f"Flat episode length={trace['length']}", flush=True)
    return trace


def rollout_meta_trace(run_dir: str, *, seed: int, max_steps: int) -> dict:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name

    from configs.survival_meta import SURVIVAL_META_ACTION_LABELS
    from evaluation.benchmark_eval_official_ppo import (
        _episode_achievements_jax,
        _to_device,
    )
    from ppo.common import load_params
    from ppo.configs import META_NUM_ACTIONS, resolve_skill_paths
    from ppo.craftax_baselines.models.actor_critic import ActorCritic
    from ppo.meta_ppo import build_skill_bundle
    from ppo.skill_step import meta_mode_to_env_action
    from training.meta_action_mask import survival_meta_greedy_mode

    print(f"Loading meta params from {run_dir} ...", flush=True)
    cfg = _load_json(os.path.join(run_dir, "config.json"))
    hyp = cfg.get("hyperparameters") or {}
    env_name = hyp.get("env_name", "Craftax-Classic-Symbolic-v1")
    layer_size = int(hyp.get("layer_size", 128))
    use_mask = bool(hyp.get("use_action_mask", False))
    mode_labels = [lb.lower() for lb in SURVIVAL_META_ACTION_LABELS]
    num_ach = 22

    params = _to_device(load_params(run_dir))
    network = ActorCritic(META_NUM_ACTIONS, layer_size)
    skill_bundle = _to_device(build_skill_bundle(resolve_skill_paths(cfg.get("skill_weights"))))
    sb = skill_bundle

    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params

    @jax.jit
    def run_episode(rng):
        obs0, state0 = env.reset(rng, env_params)
        ach0 = jnp.zeros((num_ach,), dtype=jnp.float32)

        def step_fn(carry, _):
            obs, state, rng_in, ach, done_ever = carry
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
            ach2 = jnp.maximum(ach, _episode_achievements_jax(state2, num_ach))
            new_unlock = jnp.logical_and(ach2 > 0, ach <= 0)
            done_ever2 = jnp.logical_or(done_ever, done)
            obs_out = jnp.where(done_ever, obs, obs2)
            state_out = jax.tree.map(lambda a, b: jnp.where(done_ever, a, b), state, state2)
            ach_out = jnp.where(done_ever, ach, ach2)
            mode_out = jnp.where(done_ever, jnp.int32(-1), mode)
            action_out = jnp.where(done_ever, jnp.int32(-1), action)
            unlock_out = jnp.where(done_ever, jnp.zeros_like(new_unlock), new_unlock)
            return (obs_out, state_out, rng_in, ach_out, done_ever2), (
                mode_out,
                action_out,
                unlock_out,
                done_ever2,
            )

        (_, _, _, _, _), (modes, actions, unlocks, done_flags) = jax.lax.scan(
            step_fn,
            (obs0, state0, rng, ach0, jnp.array(False)),
            None,
            length=max_steps,
        )
        return modes, actions, unlocks, done_flags

    print("Rolling meta episode (JIT) ...", flush=True)
    modes, actions, unlocks, done_flags = run_episode(jax.random.PRNGKey(seed))
    trace = _trace_from_scan(
        policy="meta",
        run_dir=run_dir,
        seed=seed,
        labels_lookup=mode_labels,
        actions_or_modes=modes,
        env_actions=actions,
        unlocks=unlocks,
        done_flags=done_flags,
        extra={"use_action_mask": use_mask},
    )
    print(f"Meta episode length={trace['length']}", flush=True)
    return trace


META_SKILL_LANES = (
    "wood",
    "drink",
    "food",
    "fight",
    "stone",
    "coal",
    "iron",
    "diamond",
)
META_CRAFTAX_LANES = (
    (
        "place / craft",
        frozenset(
            {
                "place_table",
                "place_furnace",
                "place_stone",
                "place_plant",
                "craft_wood_pickaxe",
                "craft_stone_pickaxe",
                "craft_iron_pickaxe",
                "craft_wood_sword",
                "craft_stone_sword",
                "craft_iron_sword",
            }
        ),
    ),
    ("move / sleep", frozenset({"up", "left", "down", "right", "sleep"})),
    ("noop", frozenset({"noop"})),
)

FLAT_LANES = (
    ("do", frozenset({"do"})),
    ("place", frozenset({"place_stone", "place_table", "place_furnace", "place_plant"})),
    (
        "craft",
        frozenset(
            {
                "make_wood_pickaxe",
                "make_stone_pickaxe",
                "make_iron_pickaxe",
                "make_wood_sword",
                "make_stone_sword",
                "make_iron_sword",
            }
        ),
    ),
    ("move", frozenset({"up", "left", "down", "right"})),
    ("sleep", frozenset({"sleep"})),
    ("other", frozenset({"noop"})),
)

META_SKILL_LANE_COLORS = {
    "wood": "#1D4ED8",
    "drink": "#2563EB",
    "food": "#3B82F6",
    "fight": "#60A5FA",
    "stone": "#0EA5E9",
    "coal": "#0284C7",
    "iron": "#0369A1",
    "diamond": "#7DD3FC",
}
CRAFTAX_LANE_COLOR = FLAT_COLOR


def _collapse_segments(labels: list[str]) -> list[tuple[str, int, int]]:
    if not labels:
        return []
    segs: list[tuple[str, int, int]] = []
    start = 0
    cur = labels[0]
    for i, lb in enumerate(labels[1:], start=1):
        if lb != cur:
            segs.append((cur, start, i - start))
            start = i
            cur = lb
    segs.append((cur, start, len(labels) - start))
    return segs


def _lane_segments(
    labels: list[str], lanes: list[tuple[str, frozenset[str]]]
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {name: [] for name, _ in lanes}
    label_to_lane = {}
    for name, members in lanes:
        for m in members:
            label_to_lane[m] = name
    for lb, start, length in _collapse_segments(labels):
        lane = label_to_lane.get(lb)
        if lane is not None:
            out[lane].append((start, length))
    return out


def _meta_lanes_for_trace(labels: list[str]) -> list[tuple[str, frozenset[str]]]:
    used = set(labels)
    lanes: list[tuple[str, frozenset[str]]] = []
    for skill in META_SKILL_LANES:
        if skill in used:
            lanes.append((skill, frozenset({skill})))
    for name, members in META_CRAFTAX_LANES:
        if used & members:
            lanes.append((name, members))
    return lanes


def _flat_lanes_for_trace(labels: list[str]) -> list[tuple[str, frozenset[str]]]:
    used = set(labels)
    return [(name, members) for name, members in FLAT_LANES if used & members]


def plot_timelines(
    traces: list[dict],
    output_base: str,
    *,
    show_unlocks: bool = True,
) -> str:
    import matplotlib.pyplot as plt

    _style_rc()
    max_len = max((len(t["labels"]) for t in traces), default=1)

    specs: list[tuple[dict, list[tuple[str, frozenset[str]]], dict[str, str]]] = []
    for trace in traces:
        if trace["policy"] == "meta":
            lanes = _meta_lanes_for_trace(trace["labels"])
            colors = {
                **{s: META_SKILL_LANE_COLORS[s] for s in META_SKILL_LANES},
                "place / craft": CRAFTAX_LANE_COLOR,
                "move / sleep": "#1E40AF",
                "noop": META_NOOP_COLOR,
            }
        else:
            lanes = _flat_lanes_for_trace(trace["labels"])
            colors = {
                "do": "#1D4ED8",
                "place": "#2563EB",
                "craft": "#3B82F6",
                "move": "#60A5FA",
                "sleep": "#93C5FD",
                "other": META_NOOP_COLOR,
            }
        specs.append((trace, lanes, colors))

    fig_h = max(5.0, 0.42 * sum(len(lanes) for _, lanes, _ in specs) + 1.8)
    fig, axes = plt.subplots(
        len(specs),
        1,
        figsize=(12.0, fig_h),
        dpi=150,
        facecolor="white",
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [max(2.5, len(lanes)) for _, lanes, _ in specs]},
    )
    fig.patch.set_facecolor("white")
    light_text = {"diamond", "sleep", "fight", "food"}

    for ax, (trace, lanes, colors) in zip(axes[:, 0], specs):
        ax.set_facecolor("white")
        labels = trace["labels"]
        lane_segs = _lane_segments(labels, lanes)
        n = len(lanes)
        ytick_pos = []
        ytick_lbl = []
        for i, (name, _) in enumerate(lanes):
            y = n - 1 - i
            ytick_pos.append(y)
            ytick_lbl.append(name)
            color = colors.get(name, FLAT_COLOR)
            for start, length in lane_segs[name]:
                ax.broken_barh(
                    [(start, length)],
                    (y - 0.38, 0.76),
                    facecolors=color,
                    edgecolors="white",
                    linewidth=0.25,
                )
                if length >= 10:
                    ax.text(
                        start + length / 2,
                        y,
                        name,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#0F172A" if name in light_text else "white",
                        clip_on=True,
                    )

        if show_unlocks:
            for u in trace.get("achievement_unlocks", []):
                ax.axvline(
                    u["step"] + 0.5,
                    color="#0F172A",
                    linewidth=0.5,
                    alpha=0.22,
                    zorder=0,
                )

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lbl, fontsize=9)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(0, max_len)
        ax.grid(True, axis="x", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel("Episode timestep")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_base)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            output_base + f".{ext}",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )
    plt.close(fig)
    print(f"Wrote {output_base}.png")
    print(f"Wrote {output_base}.pdf")
    return output_base + ".png"


def write_latex_figure(output_path: str, *, figure_stem: str, seed: int) -> str:
    lines = [
        "% Auto-generated by scripts/generate_thesis_episode_timeline.py — do not hand-edit.",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{figure_stem}.pdf}}",
        "\\caption{Skill / action swimlanes for one greedy episode (eval seed~"
        f"{seed}). "
        "Each row is a mode group; filled bars mark contiguous intervals where that "
        "mode was selected. Meta skills appear on separate lanes; Craftax place/craft "
        "and move/sleep are grouped. Flat actions are grouped into do, place, craft, "
        "move, and sleep. Thin vertical marks indicate new achievement unlocks.}",
        "\\label{fig:episode-action-timeline}",
        "\\end{figure}",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate episode action/mode timeline figure")
    parser.add_argument("--flat-run", default=DEFAULT_FLAT_RUN)
    parser.add_argument("--meta-run", default=DEFAULT_META_RUN)
    parser.add_argument("--seed", type=int, default=42, help="Episode reset seed")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--from-trace",
        default=None,
        help="Reuse an existing timeline trace JSON (skip rollouts)",
    )
    parser.add_argument(
        "--figure-base",
        default=os.path.join(FIG_DIR, "episode_action_timeline_20m"),
    )
    parser.add_argument(
        "--latex",
        default=os.path.join(FIG_DIR, "episode_action_timeline_figure.tex"),
    )
    parser.add_argument(
        "--trace-json",
        default=os.path.join(FIG_DIR, "episode_action_timeline_20m_trace.json"),
    )
    args = parser.parse_args()

    if args.from_trace:
        payload = _load_json(args.from_trace)
        traces = payload["traces"]
        seed = int(payload.get("seed", args.seed))
    else:
        _configure_jax(args.cpu)
        import jax

        print(f"JAX devices: {jax.devices()}", flush=True)
        traces = [
            rollout_meta_trace(args.meta_run, seed=args.seed, max_steps=args.max_steps),
            rollout_flat_trace(args.flat_run, seed=args.seed, max_steps=args.max_steps),
        ]
        seed = args.seed
        payload = {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "traces": [
                {
                    "policy": t["policy"],
                    "run_dir": t["run_dir"],
                    "length": t["length"],
                    "labels": t["labels"],
                    "achievement_unlocks": t["achievement_unlocks"],
                }
                for t in traces
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.trace_json)), exist_ok=True)
        with open(args.trace_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.trace_json}")

    plot_timelines(traces, args.figure_base)
    write_latex_figure(
        args.latex,
        figure_stem=os.path.basename(args.figure_base),
        seed=seed,
    )
    for t in traces:
        print(f"  {t['policy']}: {t['length']} steps")


if __name__ == "__main__":
    main()
