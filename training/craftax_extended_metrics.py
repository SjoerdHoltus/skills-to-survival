"""Craftax extended (full game) evaluation metrics.

Uses ``craftax.craftax.constants`` achievements and the weighted reward map
from ``ACHIEVEMENT_REWARD_MAP`` (+1 / +3 / +5 / +8 tiers via ``achievement_mapping``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement, achievement_mapping

from training.craftax_classic_metrics import (
    craftax_terminal_score,
    crafter_score_from_success_rates,
    mean_std_across_seeds,
)

CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS = len(Achievement)
CRAFTAX_EXTENDED_ACHIEVEMENT_NAMES = [
    Achievement(i).name for i in range(CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS)
]
CRAFTAX_EXTENDED_MAX_ACHIEVEMENT_REWARD = float(np.asarray(ACHIEVEMENT_REWARD_MAP).sum())

CRAFTAX_EXTENDED_ACHIEVEMENT_DEPTH: dict[str, int] = {
    name: int(achievement_mapping(i)) for i, name in enumerate(CRAFTAX_EXTENDED_ACHIEVEMENT_NAMES)
}

_ACHIEVEMENT_DEPTH_VEC = np.asarray(
    [CRAFTAX_EXTENDED_ACHIEVEMENT_DEPTH[name] for name in CRAFTAX_EXTENDED_ACHIEVEMENT_NAMES],
    dtype=np.int32,
)


def achievement_depth_from_vector(achievements: np.ndarray) -> int:
    ach = np.asarray(achievements).reshape(-1) > 0
    if not np.any(ach):
        return 0
    return int(_ACHIEVEMENT_DEPTH_VEC[ach].max())


def per_achievement_success_rates(achievement_matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(achievement_matrix) > 0
    if mat.size == 0:
        return np.zeros(CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    return mat.mean(axis=0).astype(np.float64)


def summarize_benchmark_episodes(
    *,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    achievement_matrix: np.ndarray,
    died: np.ndarray | None = None,
) -> dict[str, Any]:
    returns = np.asarray(episode_returns, dtype=np.float64).reshape(-1)
    lengths = np.asarray(episode_lengths, dtype=np.float64).reshape(-1)
    ach_mat = np.asarray(achievement_matrix, dtype=np.float64)
    if ach_mat.ndim == 1:
        ach_mat = ach_mat.reshape(1, -1)

    success_rates = per_achievement_success_rates(ach_mat)
    success_rates_pct = success_rates * 100.0
    ach_counts = ach_mat.sum(axis=-1)

    depths = np.asarray(
        [achievement_depth_from_vector(ach_mat[i]) for i in range(ach_mat.shape[0])],
        dtype=np.float64,
    )

    summary: dict[str, Any] = {
        "benchmark": "craftax_extended",
        "num_achievements": CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS,
        "max_achievement_reward_reference": CRAFTAX_EXTENDED_MAX_ACHIEVEMENT_REWARD,
        "num_episodes": int(returns.size),
        "episode_return": {
            "mean": float(returns.mean()) if returns.size else 0.0,
            "std": float(returns.std()) if returns.size else 0.0,
        },
        "return_pct_max_achievement_reward": {
            "mean": float((returns / CRAFTAX_EXTENDED_MAX_ACHIEVEMENT_REWARD * 100.0).mean())
            if returns.size
            else 0.0,
            "std": float((returns / CRAFTAX_EXTENDED_MAX_ACHIEVEMENT_REWARD * 100.0).std())
            if returns.size
            else 0.0,
        },
        "episode_length": {
            "mean": float(lengths.mean()) if lengths.size else 0.0,
            "std": float(lengths.std()) if lengths.size else 0.0,
        },
        "achievement_count": {
            "mean": float(ach_counts.mean()) if ach_counts.size else 0.0,
            "std": float(ach_counts.std()) if ach_counts.size else 0.0,
        },
        "achievement_depth": {
            "mean": float(depths.mean()) if depths.size else 0.0,
            "std": float(depths.std()) if depths.size else 0.0,
        },
        "craftax_extended_score_pct": crafter_score_from_success_rates(success_rates_pct),
        "mean_terminal_score_pct": float(
            np.mean([craftax_terminal_score(ach_mat[i]) for i in range(ach_mat.shape[0])])
        )
        if ach_mat.shape[0]
        else 0.0,
        "per_achievement_success_rate": {
            CRAFTAX_EXTENDED_ACHIEVEMENT_NAMES[i]: float(success_rates_pct[i])
            for i in range(CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS)
        },
    }

    if died is not None:
        d = np.asarray(died).reshape(-1) > 0.5
        summary["death_rate"] = float(d.mean()) if d.size else 0.0

    return summary


__all__ = [
    "CRAFTAX_EXTENDED_ACHIEVEMENT_DEPTH",
    "CRAFTAX_EXTENDED_ACHIEVEMENT_NAMES",
    "CRAFTAX_EXTENDED_MAX_ACHIEVEMENT_REWARD",
    "CRAFTAX_EXTENDED_NUM_ACHIEVEMENTS",
    "achievement_depth_from_vector",
    "mean_std_across_seeds",
    "per_achievement_success_rates",
    "summarize_benchmark_episodes",
]
