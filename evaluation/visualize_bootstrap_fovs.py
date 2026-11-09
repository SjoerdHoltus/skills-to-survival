#!/usr/bin/env python3
"""
Generate N visualizations of the 7×9 symbolic FOV crops produced by bootstrap scripts.

Supports find_wood, drink_water, collect_stone, collect_ore (coal/iron/diamond),
monster_fight, and find_food. By default saves one PNG grid per bootstrap type.

Examples:
  python -m evaluation.visualize_bootstrap_fovs --num 12 --seed 0
  python -m evaluation.visualize_bootstrap_fovs --bootstrap find_wood collect_stone --num 8
  python -m evaluation.visualize_bootstrap_fovs --bootstrap collect_ore --ore-target diamond --num 6
  python -m evaluation.visualize_bootstrap_fovs --all --num 10 --out runs/bootstrap_fov_viz/demo
  python -m evaluation.visualize_bootstrap_fovs --bootstrap monster_fight --num 6 --show
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")


import craftax  # noqa: F401
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_env import make_craftax_env_from_name

import configs as cfg
from evaluation.eval_common import sync_pixel_from_symbolic, upscale_frames
from training.bootstrap_ore import (
    TARGET_BLOCK_VALUES,
    bootstrap_ore_state,
    target_id_from_name,
)
from training.bootstrap_stone import bootstrap_stone_state
from training.bootstrap_water import bootstrap_water_state
from training.bootstrap_wood import bootstrap_wood_state
from training.fitness_collect_ore import make_env as make_env_collect_ore
from training.fitness_collect_stone import make_env as make_env_collect_stone
from training.fitness_drink_water import make_env as make_env_drink_water
from training.fitness_find_food import make_env as make_env_find_food
from training.fitness_find_wood import make_env as make_env_find_wood
from training.fitness_monster_fight import make_env as make_env_monster_fight
from training.monster_world_bootstrap import (
    bootstrap_find_food_state,
    bootstrap_monster_fight_state,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYMBOLIC_FOV_ROWS = 7
SYMBOLIC_FOV_COLS = 9
SYMBOLIC_FOV_PR = 3
SYMBOLIC_FOV_PC = 4
_OOB = np.array([0.07, 0.07, 0.09], dtype=np.float32)

_BLOCK_RGB: dict[int, np.ndarray] = {
    int(BlockType.GRASS.value): np.array([0.20, 0.55, 0.20], dtype=np.float32),
    int(BlockType.PATH.value): np.array([0.55, 0.45, 0.30], dtype=np.float32),
    int(BlockType.WATER.value): np.array([0.15, 0.45, 0.95], dtype=np.float32),
    int(BlockType.SAND.value): np.array([0.85, 0.75, 0.45], dtype=np.float32),
    int(BlockType.TREE.value): np.array([0.10, 0.38, 0.12], dtype=np.float32),
    int(BlockType.STONE.value): np.array([0.55, 0.55, 0.58], dtype=np.float32),
    int(BlockType.COAL.value): np.array([0.18, 0.18, 0.20], dtype=np.float32),
    int(BlockType.IRON.value): np.array([0.72, 0.42, 0.28], dtype=np.float32),
    int(BlockType.DIAMOND.value): np.array([0.35, 0.85, 0.95], dtype=np.float32),
}
_DEFAULT_BLOCK_RGB = np.array([0.25, 0.25, 0.28], dtype=np.float32)
_PLAYER_RGB = np.array([0.15, 0.45, 1.0], dtype=np.float32)
_MOB_RGB = {
    "zombie": np.array([1.0, 0.15, 0.15], dtype=np.float32),
    "skeleton": np.array([0.95, 0.95, 1.0], dtype=np.float32),
    "cow": np.array([0.65, 0.40, 0.12], dtype=np.float32),
}

_BASE_BOOTSTRAP_KEYS = (
    "find_wood",
    "drink_water",
    "collect_stone",
    "monster_fight",
    "find_food",
)


@dataclass(frozen=True)
class BootstrapSpec:
    key: str
    label: str
    make_env: Callable
    apply_bootstrap: Callable
    config_summary: str
    panel_title: Callable


def _player_pos(state) -> tuple[int, int]:
    pr, pc = np.asarray(state.player_position).astype(int)
    return int(pr), int(pc)


def _crop_world_to_fov(pr: int, pc: int, wr: int, wc: int) -> tuple[int, int] | None:
    cri = wr - pr + SYMBOLIC_FOV_PR
    crj = wc - pc + SYMBOLIC_FOV_PC
    if 0 <= cri < SYMBOLIC_FOV_ROWS and 0 <= crj < SYMBOLIC_FOV_COLS:
        return cri, crj
    return None


def _count_block_in_fov(state, block_value: int) -> int:
    sm = np.asarray(state.map)
    pr, pc = _player_pos(state)
    total = 0
    for ci in range(SYMBOLIC_FOV_ROWS):
        for cj in range(SYMBOLIC_FOV_COLS):
            wr = pr - SYMBOLIC_FOV_PR + ci
            wc = pc - SYMBOLIC_FOV_PC + cj
            if 0 <= wr < sm.shape[0] and 0 <= wc < sm.shape[1]:
                if int(sm[wr, wc]) == block_value:
                    total += 1
    return total


def _count_mobs(state, mob_name: str) -> int:
    mob = getattr(state, mob_name)
    return int(np.asarray(mob.mask).sum())


def _state_to_rgb_fov(state) -> np.ndarray:
    sm = np.asarray(state.map)
    h, w = sm.shape
    pr, pc = _player_pos(state)
    rgb = np.tile(_OOB, (SYMBOLIC_FOV_ROWS, SYMBOLIC_FOV_COLS, 1)).astype(np.float32)

    for ci in range(SYMBOLIC_FOV_ROWS):
        for cj in range(SYMBOLIC_FOV_COLS):
            wr = pr - SYMBOLIC_FOV_PR + ci
            wc = pc - SYMBOLIC_FOV_PC + cj
            if 0 <= wr < h and 0 <= wc < w:
                cur = int(sm[wr, wc])
                rgb[ci, cj] = _BLOCK_RGB.get(cur, _DEFAULT_BLOCK_RGB)

    def _paint_mob(mob, color: np.ndarray) -> None:
        mask = np.asarray(mob.mask)
        pos = np.asarray(mob.position).astype(int)
        for i in np.where(mask)[0]:
            crop = _crop_world_to_fov(pr, pc, int(pos[i, 0]), int(pos[i, 1]))
            if crop is not None:
                cri, crj = crop
                rgb[cri, crj] = color

    _paint_mob(state.zombies, _MOB_RGB["zombie"])
    _paint_mob(state.skeletons, _MOB_RGB["skeleton"])
    _paint_mob(state.cows, _MOB_RGB["cow"])
    rgb[SYMBOLIC_FOV_PR, SYMBOLIC_FOV_PC] = _PLAYER_RGB
    return rgb


def _state_to_rgb_full(state) -> np.ndarray:
    sm = np.asarray(state.map)
    h, w = sm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for wr in range(h):
        for wc in range(w):
            cur = int(sm[wr, wc])
            rgb[wr, wc] = _BLOCK_RGB.get(cur, _DEFAULT_BLOCK_RGB)
    pr, pc = _player_pos(state)
    rgb[pr, pc] = _PLAYER_RGB

    def _paint_mob(mob, color: np.ndarray) -> None:
        mask = np.asarray(mob.mask)
        pos = np.asarray(mob.position).astype(int)
        for i in np.where(mask)[0]:
            rgb[int(pos[i, 0]), int(pos[i, 1])] = color

    _paint_mob(state.zombies, _MOB_RGB["zombie"])
    _paint_mob(state.skeletons, _MOB_RGB["skeleton"])
    _paint_mob(state.cows, _MOB_RGB["cow"])
    return rgb


def _make_ore_bootstrap(target_name: str, wall_count: int | None):
    target_id = jnp.int32(target_id_from_name(target_name))

    def apply(state, boot_rng, params):
        if wall_count is None:
            boot_rng, wall_rng = jax.random.split(boot_rng)
            wc = jax.random.randint(wall_rng, (), 1, 4)
        else:
            wc = jnp.int32(wall_count)
        return bootstrap_ore_state(
            state,
            boot_rng,
            params,
            target_id=target_id,
            wall_count=wc,
        )

    return apply


def _ore_panel_title(state, index: int, target_name: str) -> str:
    pr, pc = _player_pos(state)
    ore_val = int(TARGET_BLOCK_VALUES[target_id_from_name(target_name)])
    return (
        f"#{index + 1}  {target_name}={_count_block_in_fov(state, ore_val)}  "
        f"stone={_count_block_in_fov(state, int(BlockType.STONE.value))}  "
        f"player=({pr},{pc})"
    )


def _build_bootstrap_specs(
    ore_targets: list[str],
    wall_count: int | None,
) -> dict[str, BootstrapSpec]:
    specs: dict[str, BootstrapSpec] = {
        "find_wood": BootstrapSpec(
            key="find_wood",
            label="find_wood",
            make_env=make_env_find_wood,
            apply_bootstrap=bootstrap_wood_state,
            config_summary=(
                f"trees={cfg.FIND_WOOD_NUM_TREES_MIN}..{cfg.FIND_WOOD_NUM_TREES_MAX}  "
                f"fov_r={cfg.CHEBYSHEV_R}"
            ),
            panel_title=lambda s, i: (
                f"#{i + 1}  trees={_count_block_in_fov(s, int(BlockType.TREE.value))}  "
                f"player={_player_pos(s)}"
            ),
        ),
        "drink_water": BootstrapSpec(
            key="drink_water",
            label="drink_water",
            make_env=make_env_drink_water,
            apply_bootstrap=bootstrap_water_state,
            config_summary=(
                f"spawn_r={cfg.DRINK_WATER_SPAWN_MIN_CHEBYSHEV_R}.."
                f"{cfg.DRINK_WATER_SPAWN_MAX_CHEBYSHEV_R}  "
                f"patch_side={cfg.DRINK_WATER_WATER_PATCH_SIDE}"
            ),
            panel_title=lambda s, i: (
                f"#{i + 1}  water={_count_block_in_fov(s, int(BlockType.WATER.value))}  "
                f"player={_player_pos(s)}"
            ),
        ),
        "collect_stone": BootstrapSpec(
            key="collect_stone",
            label="collect_stone",
            make_env=make_env_collect_stone,
            apply_bootstrap=bootstrap_stone_state,
            config_summary=(
                f"stone={cfg.COLLECT_STONE_NUM_BLOCKS_MIN}.."
                f"{cfg.COLLECT_STONE_NUM_BLOCKS_MAX}  "
                f"fov_r={cfg.COLLECT_STONE_FOV_CHEBYSHEV_R}"
            ),
            panel_title=lambda s, i: (
                f"#{i + 1}  stone={_count_block_in_fov(s, int(BlockType.STONE.value))}  "
                f"player={_player_pos(s)}"
            ),
        ),
        "monster_fight": BootstrapSpec(
            key="monster_fight",
            label="monster_fight",
            make_env=make_env_monster_fight,
            apply_bootstrap=bootstrap_monster_fight_state,
            config_summary=(
                f"zombies={cfg.MONSTER_FIGHT_NUM_ZOMBIES_MIN}.."
                f"{cfg.MONSTER_FIGHT_NUM_ZOMBIES_MAX}  "
                f"skel_max={cfg.MONSTER_FIGHT_NUM_SKELETONS_MAX}  "
                f"trees={cfg.MONSTER_FIGHT_NUM_TREES_MIN}.."
                f"{cfg.MONSTER_FIGHT_NUM_TREES_MAX}  "
                f"fov_r={cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R}"
            ),
            panel_title=lambda s, i: (
                f"#{i + 1}  Z={_count_mobs(s, 'zombies')}  "
                f"Sk={_count_mobs(s, 'skeletons')}  "
                f"trees={_count_block_in_fov(s, int(BlockType.TREE.value))}  "
                f"player={_player_pos(s)}"
            ),
        ),
        "find_food": BootstrapSpec(
            key="find_food",
            label="find_food",
            make_env=make_env_find_food,
            apply_bootstrap=bootstrap_find_food_state,
            config_summary=(
                f"cows={cfg.FIND_FOOD_NUM_COWS_MIN}..{cfg.FIND_FOOD_NUM_COWS_MAX}  "
                f"trees={cfg.FIND_FOOD_NUM_TREES_MIN}..{cfg.FIND_FOOD_NUM_TREES_MAX}  "
                f"fov_r={cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R}"
            ),
            panel_title=lambda s, i: (
                f"#{i + 1}  cows={_count_mobs(s, 'cows')}  "
                f"trees={_count_block_in_fov(s, int(BlockType.TREE.value))}  "
                f"player={_player_pos(s)}"
            ),
        ),
    }

    for target_name in ore_targets:
        key = f"collect_ore_{target_name}"
        specs[key] = BootstrapSpec(
            key=key,
            label=f"collect_ore ({target_name})",
            make_env=make_env_collect_ore,
            apply_bootstrap=_make_ore_bootstrap(target_name, wall_count),
            config_summary=(
                f"target={target_name}  "
                f"ore_dist={cfg.COLLECT_ORE_ORE_DISTANCE_MIN}.."
                f"{cfg.COLLECT_ORE_ORE_DISTANCE_MAX}  "
                f"wall_count={'sample 1..3' if wall_count is None else wall_count}"
            ),
            panel_title=lambda s, i, tn=target_name: _ore_panel_title(s, i, tn),
        )
    return specs


def _resolve_selected_keys(
    bootstrap_args: list[str] | None,
    use_all: bool,
    ore_targets: list[str],
) -> list[str]:
    if use_all or bootstrap_args is None:
        keys = list(_BASE_BOOTSTRAP_KEYS)
        keys.extend(f"collect_ore_{t}" for t in ore_targets)
        return keys

    keys: list[str] = []
    for name in bootstrap_args:
        if name == "collect_ore":
            keys.extend(f"collect_ore_{t}" for t in ore_targets)
        elif name.startswith("collect_ore_"):
            keys.append(name)
        else:
            keys.append(name)
    return keys


def _sample_states(spec: BootstrapSpec, num: int, seed: int, *, pixel: bool = False):
    sym_env, env_params = spec.make_env()
    rng = jax.random.PRNGKey(int(seed))
    states = []
    pixel_frames: list[np.ndarray] = []

    pix_env = None
    pix_params = None
    if pixel:
        pix_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
        pix_params = env_params

    for _ in range(num):
        rng, reset_rng, boot_rng = jax.random.split(rng, 3)
        _, sym_state = sym_env.reset(reset_rng, env_params)
        sym_state = spec.apply_bootstrap(sym_state, boot_rng, env_params)
        sym_state = jax.device_get(sym_state)
        states.append(sym_state)

        if pixel:
            _, pix_state = pix_env.reset(reset_rng, pix_params)
            pix_state = sync_pixel_from_symbolic(sym_state, pix_state)
            frame = np.asarray(pix_env.get_obs(pix_state))
            pixel_frames.append((frame * 255).astype(np.uint8))

    if pixel:
        return states, pixel_frames
    return states


def _render_pixel_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return frame
    return upscale_frames([frame], scale=scale)[0]


def _render_grid_figure(
    spec: BootstrapSpec,
    states: list,
    *,
    seed: int,
    full_map: bool,
    pixel_frames: list[np.ndarray] | None = None,
    pixel_scale: int = 6,
) -> plt.Figure:
    n = len(states)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    if pixel_frames is not None:
        view = f"Craftax Classic pixel render (×{pixel_scale})"
    else:
        view = "full map" if full_map else "7×9 symbolic FOV"
    fig.suptitle(
        f"{spec.label} × {n} (seed={seed})  |  {view}\n{spec.config_summary}",
        fontsize=10,
    )
    rgb_fn = _state_to_rgb_full if full_map else _state_to_rgb_fov
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        state = states[k]
        if pixel_frames is not None:
            img = _render_pixel_frame(pixel_frames[k], pixel_scale)
            ax.imshow(img, origin="upper", interpolation="nearest")
        else:
            ax.imshow(rgb_fn(state), origin="upper", interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title(spec.panel_title(state, k), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def _show_figure_scrolled(fig, window_title: str, viewport_w: int, viewport_h: int) -> None:
    import tkinter as tk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title(window_title)

    outer = tk.Frame(root, width=viewport_w, height=viewport_h)
    outer.pack(fill=tk.BOTH, expand=True)
    outer.pack_propagate(False)

    scrollbar = tk.Scrollbar(outer, orient=tk.VERTICAL)
    yscroll = tk.Canvas(outer, highlightthickness=0, yscrollcommand=scrollbar.set)
    scrollbar.config(command=yscroll.yview)

    inner = tk.Frame(yscroll)
    win_id = yscroll.create_window((0, 0), window=inner, anchor="nw")

    def _inner_configure(_event=None):
        bbox = yscroll.bbox("all")
        if bbox is not None:
            yscroll.configure(scrollregion=bbox)

    def _yscroll_canvas_configure(ev):
        yscroll.itemconfig(win_id, width=ev.width)

    inner.bind("<Configure>", lambda _e: _inner_configure())
    yscroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    yscroll.bind("<Configure>", _yscroll_canvas_configure)

    canvas = FigureCanvasTkAgg(fig, master=inner)
    canvas.draw()
    mpl_widget = canvas.get_tk_widget()
    mpl_widget.pack(fill=tk.BOTH, expand=True)

    def _wheel(ev):
        d = getattr(ev, "delta", 0)
        if d:
            yscroll.yview_scroll(-int(round(d / 120)) or (-1 if d > 0 else 1), "units")
            return
        if ev.num == 4:
            yscroll.yview_scroll(-2, "units")
        elif ev.num == 5:
            yscroll.yview_scroll(2, "units")

    for w in (yscroll, mpl_widget):
        w.bind("<MouseWheel>", _wheel)
        w.bind("<Button-4>", _wheel)
        w.bind("<Button-5>", _wheel)

    root.geometry(f"{viewport_w}x{viewport_h}")
    root.after_idle(_inner_configure)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize symbolic FOV crops from bootstrap scripts."
    )
    parser.add_argument(
        "--bootstrap",
        nargs="+",
        choices=[
            *_BASE_BOOTSTRAP_KEYS,
            "collect_ore",
            *(f"collect_ore_{t}" for t in cfg.COLLECT_ORE_TARGET_NAMES),
        ],
        help="Bootstrap type(s) to visualize (default: all when --all or omitted with --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include every bootstrap type (and all ore targets unless --ore-target narrows).",
    )
    parser.add_argument("--num", type=int, default=10, help="Samples per bootstrap type.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ore-target",
        choices=[*cfg.COLLECT_ORE_TARGET_NAMES, "all"],
        default="all",
        help="Ore target(s) when visualizing collect_ore.",
    )
    parser.add_argument(
        "--wall-count",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Fixed collect_ore stone-shell depth (default: sample 1..3 per panel).",
    )
    parser.add_argument(
        "--full-map",
        action="store_true",
        help="Render the full world map instead of the 7×9 symbolic FOV crop.",
    )
    parser.add_argument(
        "--pixel",
        action="store_true",
        help="Render bootstrapped states with Craftax Classic pixel sprites (dual env sync).",
    )
    parser.add_argument(
        "--pixel-scale",
        type=int,
        default=6,
        help="Nearest-neighbor upscale for --pixel frames (default: 6).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for PNG files (default: runs/bootstrap_fov_viz/<timestamp>).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive Tk window for each bootstrap grid.",
    )
    parser.add_argument("--viewport-w", type=int, default=1200)
    parser.add_argument("--viewport-h", type=int, default=880)
    args = parser.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    if args.pixel and args.full_map:
        parser.error("--pixel and --full-map are mutually exclusive.")

    n = max(1, int(args.num))
    ore_targets = (
        list(cfg.COLLECT_ORE_TARGET_NAMES) if args.ore_target == "all" else [args.ore_target]
    )
    specs = _build_bootstrap_specs(ore_targets, args.wall_count)

    if args.all or not args.bootstrap:
        selected = _resolve_selected_keys(None, True, ore_targets)
    else:
        selected = _resolve_selected_keys(args.bootstrap, False, ore_targets)

    unknown = [k for k in selected if k not in specs]
    if unknown:
        parser.error(f"Unknown bootstrap keys: {unknown}")

    out_dir = args.out
    if out_dir is None and not args.show:
        stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
        out_dir = os.path.join(_ROOT, "runs", "bootstrap_fov_viz", stamp)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    saved_paths: list[str] = []
    for key in selected:
        spec = specs[key]
        if args.pixel:
            states, pixel_frames = _sample_states(spec, n, args.seed, pixel=True)
        else:
            states = _sample_states(spec, n, args.seed, pixel=False)
            pixel_frames = None
        fig = _render_grid_figure(
            spec,
            states,
            seed=args.seed,
            full_map=args.full_map,
            pixel_frames=pixel_frames,
            pixel_scale=max(1, int(args.pixel_scale)),
        )
        suffix = "_pixel" if args.pixel else ""
        if out_dir is not None:
            png_path = os.path.join(out_dir, f"{key}{suffix}_seed{args.seed}_n{n}.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            saved_paths.append(png_path)
            print(f"Saved {png_path}")
        if args.show:
            if out_dir is None:
                import matplotlib

                matplotlib.use("TkAgg")
            _show_figure_scrolled(
                fig, f"Bootstrap FOV: {spec.label}", args.viewport_w, args.viewport_h
            )
        else:
            plt.close(fig)

    if saved_paths:
        print(f"\nWrote {len(saved_paths)} PNG(s) under {out_dir}")
    elif args.show:
        print(f"Displayed {len(selected)} bootstrap grid(s).")
    else:
        parser.error("Nothing to do: pass --out and/or --show.")


if __name__ == "__main__":
    main()
