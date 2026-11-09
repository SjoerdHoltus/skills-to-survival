"""Official Craftax-Classic / Crafter-style evaluation metrics.

References:
- Crafter score: https://github.com/danijar/crafter (geometric mean of achievement success rates)
- Craftax Classic terminal info: ``craftax.craftax_classic.envs.common.compute_score``
- SGRL Appendix A.2: success rate in [0, 100], score = exp(mean(log(1 + rho_i))) - 1
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from craftax.craftax_classic.constants import Achievement

CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS = len(Achievement)
CRAFTAX_CLASSIC_ACHIEVEMENT_NAMES = [
    Achievement(i).name for i in range(CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS)
]

# Maximum extrinsic return if every achievement unlocks once (+1 each) with no net health change.
CRAFTAX_CLASSIC_MAX_ACHIEVEMENT_RETURN = float(CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS)

# Crafter / SGRL depth tiers (1 = earliest, 8 = COLLECT_DIAMOND / EAT_PLANT frontier).
CRAFTAX_CLASSIC_ACHIEVEMENT_DEPTH: dict[str, int] = {
    "WAKE_UP": 1,
    "COLLECT_WOOD": 1,
    "PLACE_TABLE": 2,
    "COLLECT_SAPLING": 2,
    "COLLECT_DRINK": 2,
    "MAKE_WOOD_PICKAXE": 2,
    "MAKE_WOOD_SWORD": 2,
    "PLACE_PLANT": 2,
    "EAT_COW": 3,
    "DEFEAT_ZOMBIE": 3,
    "COLLECT_STONE": 3,
    "PLACE_STONE": 3,
    "EAT_PLANT": 3,
    "DEFEAT_SKELETON": 4,
    "MAKE_STONE_PICKAXE": 4,
    "MAKE_STONE_SWORD": 4,
    "PLACE_FURNACE": 5,
    "COLLECT_COAL": 5,
    "COLLECT_IRON": 6,
    "MAKE_IRON_PICKAXE": 7,
    "MAKE_IRON_SWORD": 7,
    "COLLECT_DIAMOND": 8,
}

_ACHIEVEMENT_DEPTH_VEC = np.asarray(
    [CRAFTAX_CLASSIC_ACHIEVEMENT_DEPTH[name] for name in CRAFTAX_CLASSIC_ACHIEVEMENT_NAMES],
    dtype=np.int32,
)


def achievement_depth_from_vector(achievements: np.ndarray) -> int:
    """Furthest Crafter depth unlocked in ``achievements`` (0 if none)."""
    ach = np.asarray(achievements).reshape(-1) > 0
    if not np.any(ach):
        return 0
    return int(_ACHIEVEMENT_DEPTH_VEC[ach].max())


def crafter_score_from_success_rates(success_rates_pct: np.ndarray) -> float:
    """Geometric mean Crafter score with ``rho_i`` in [0, 100] (SGRL Appendix A.2)."""
    rho = np.clip(np.asarray(success_rates_pct, dtype=np.float64), 0.0, 100.0)
    if rho.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log1p(rho))) - 1.0)


def craftax_terminal_score(achievements: np.ndarray) -> float:
    """Per-episode score from ``craftax.craftax_classic.envs.common.compute_score``."""
    ach = np.asarray(achievements).reshape(-1)
    scaled = np.where(ach > 0, 100.0, 0.0)
    return float(np.exp(np.mean(np.log1p(scaled))) - 1.0)


def per_achievement_success_rates(achievement_matrix: np.ndarray) -> np.ndarray:
    """Fraction of episodes where each achievement was unlocked, shape ``(A,)``."""
    mat = np.asarray(achievement_matrix) > 0
    if mat.size == 0:
        return np.zeros(CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS, dtype=np.float64)
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
    """Aggregate official metrics over eval episodes."""
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
        "benchmark": "craftax_classic",
        "num_achievements": CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS,
        "max_achievement_reward_reference": CRAFTAX_CLASSIC_MAX_ACHIEVEMENT_RETURN,
        "num_episodes": int(returns.size),
        "episode_return": {
            "mean": float(returns.mean()) if returns.size else 0.0,
            "std": float(returns.std()) if returns.size else 0.0,
        },
        "return_pct_max_achievement_only": {
            "mean": float((returns / CRAFTAX_CLASSIC_MAX_ACHIEVEMENT_RETURN * 100.0).mean())
            if returns.size
            else 0.0,
            "std": float((returns / CRAFTAX_CLASSIC_MAX_ACHIEVEMENT_RETURN * 100.0).std())
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
        "crafter_score_pct": crafter_score_from_success_rates(success_rates_pct),
        "mean_terminal_score_pct": float(
            np.mean([craftax_terminal_score(ach_mat[i]) for i in range(ach_mat.shape[0])])
        )
        if ach_mat.shape[0]
        else 0.0,
        "per_achievement_success_rate": {
            CRAFTAX_CLASSIC_ACHIEVEMENT_NAMES[i]: float(success_rates_pct[i])
            for i in range(CRAFTAX_CLASSIC_NUM_ACHIEVEMENTS)
        },
    }

    # Per-episode series for thesis box / whisker plots.
    if returns.size:
        pct = returns / CRAFTAX_CLASSIC_MAX_ACHIEVEMENT_RETURN * 100.0
        summary["episode_returns"] = [float(x) for x in returns]
        summary["episode_return_pct_max"] = [float(x) for x in pct]
        summary["episode_lengths"] = [float(x) for x in lengths]
        q1, med, q3 = (float(x) for x in np.percentile(pct, [25, 50, 75]))
        iqr = q3 - q1
        summary["episode_return_pct_max_boxplot"] = {
            "min": float(pct.min()),
            "q1": q1,
            "median": med,
            "q3": q3,
            "max": float(pct.max()),
            "iqr": float(iqr),
            "whisker_low": float(max(pct.min(), q1 - 1.5 * iqr)),
            "whisker_high": float(min(pct.max(), q3 + 1.5 * iqr)),
        }

    if died is not None:
        d = np.asarray(died).reshape(-1) > 0.5
        summary["death_rate"] = float(d.mean()) if d.size else 0.0
        if d.size:
            summary["episode_died"] = [bool(x) for x in d]

    return summary


def mean_std_across_seeds(values: list[float]) -> dict[str, float]:
    """Mean and sample standard deviation (ddof=1) across independent seeds.

    With a single value, std is reported as 0.0 (ddof=0) so callers can format
    mean±std without special-casing n=1.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0 if arr.size == 1 else 1)),
        "n": int(arr.size),
    }
