"""Shared helpers for primitive GIF eval and batch-eval JSON output."""

from __future__ import annotations

import os

import imageio.v3 as iio
import numpy as np

from training.run_utils import infer_run_dir_from_weight_path


def evals_dir_and_basename_for_weight(weight_path: str) -> tuple[str, str, str]:
    """Return ``(evals_dir, run_basename, weight_label_stem)``."""
    abs_path = os.path.abspath(weight_path)
    label_stem = os.path.splitext(os.path.basename(abs_path))[0]
    run_dir = infer_run_dir_from_weight_path(abs_path)
    if run_dir:
        return (
            os.path.join(run_dir, "evals"),
            os.path.basename(run_dir.rstrip(os.sep)),
            label_stem,
        )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "eval_weights", label_stem), label_stem, label_stem


def sync_pixel_from_symbolic(sym_state, pix_state):
    """Copy symbolic EnvState fields into the pixel env state for rendering."""
    return pix_state.replace(
        map=sym_state.map,
        mob_map=sym_state.mob_map,
        zombies=sym_state.zombies,
        skeletons=sym_state.skeletons,
        cows=sym_state.cows,
        arrows=sym_state.arrows,
        inventory=sym_state.inventory,
        player_position=sym_state.player_position,
        player_direction=sym_state.player_direction,
        player_health=sym_state.player_health,
        player_food=sym_state.player_food,
        player_drink=sym_state.player_drink,
        player_energy=sym_state.player_energy,
        player_recover=sym_state.player_recover,
        player_hunger=sym_state.player_hunger,
        player_thirst=sym_state.player_thirst,
        player_fatigue=sym_state.player_fatigue,
        is_sleeping=sym_state.is_sleeping,
        light_level=sym_state.light_level,
        achievements=sym_state.achievements,
        timestep=sym_state.timestep,
    )


def upscale_frames(frames, scale: int = 8):
    return [np.repeat(np.repeat(f, scale, axis=0), scale, axis=1) for f in frames]


def save_gif(frames, path: str, fps: int = 10) -> None:
    iio.imwrite(path, frames, loop=0, duration=int(1000 / fps))
    print(f"Saved GIF ({len(frames)} frames) to {path}")
