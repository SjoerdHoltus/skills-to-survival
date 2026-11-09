#!/usr/bin/env python3
"""Render a full natural Craftax Classic map with pixel block sprites."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


import craftax  # noqa: F401
import jax
import numpy as np
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_IMG, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_env():
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    params = env.default_params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
        mob_despawn_distance=10_000,
    )
    return env, params


def render_full_map(state, *, tile_size: int = BLOCK_PIXEL_SIZE_IMG) -> np.ndarray:
    textures = load_all_textures()[tile_size]
    block_textures = np.asarray(textures["block_textures"])
    player_textures = np.asarray(textures["player_textures"])

    sm = np.asarray(state.map, dtype=np.int32)
    h, w = sm.shape
    canvas = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            tex = block_textures[int(sm[r, c])].astype(np.float32)
            canvas[
                r * tile_size : (r + 1) * tile_size,
                c * tile_size : (c + 1) * tile_size,
            ] = tex

    pr, pc = map(int, np.asarray(state.player_position))
    dir_idx = int(state.player_direction) - 1
    if bool(state.is_sleeping):
        dir_idx = 4
    player = player_textures[dir_idx].astype(np.float32)
    if player.shape[-1] == 4:
        alpha = player[..., 3:4] / 255.0
        rgb = player[..., :3]
    else:
        alpha = np.ones((tile_size, tile_size, 1), dtype=np.float32)
        rgb = player

    y0, x0 = pr * tile_size, pc * tile_size
    region = canvas[y0 : y0 + tile_size, x0 : x0 + tile_size]
    canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = region * (1.0 - alpha) + rgb * alpha
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a pixel-rendered natural Classic map.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_ROOT, "figures", "output", "ore_heatmaps"),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=BLOCK_PIXEL_SIZE_IMG,
        help=f"Pixels per map tile (default: {BLOCK_PIXEL_SIZE_IMG}).",
    )
    args = parser.parse_args()

    env, params = _make_env()
    rng = jax.random.PRNGKey(int(args.seed))
    _, state = env.reset(rng, params)
    state = jax.device_get(state)

    frame = render_full_map(state, tile_size=int(args.tile_size))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"natural_map_seed{args.seed}_pixel.png")
    Image.fromarray(frame).save(out_path)
    print(f"Saved {out_path} ({frame.shape[1]}×{frame.shape[0]} px)")


if __name__ == "__main__":
    main()
