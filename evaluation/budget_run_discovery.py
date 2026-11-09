#!/usr/bin/env python3
"""Discover budget-sweep runs that have saved checkpoints."""

from __future__ import annotations

import json
import os
import re
from glob import glob

from ppo.configs import CRAFTAX_BUDGET_LABELS

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

_BUDGET_SUFFIX = r"[a-z0-9]+"
POLICY_PATTERNS = {
    "extended_flat_512": (
        re.compile(rf"^baselines_extended_ppo_(?P<budget>{_BUDGET_SUFFIX})$"),
        "flat",
    ),
    "classic_flat": (
        re.compile(rf"^baselines_classic_flat_ppo_(?P<budget>{_BUDGET_SUFFIX})$"),
        "flat",
    ),
    "classic_meta": (
        re.compile(rf"^baselines_classic_meta_ppo_(?P<budget>{_BUDGET_SUFFIX})$"),
        "meta",
    ),
    "classic_meta_no_mask": (
        re.compile(rf"^baselines_classic_meta_ppo_no_mask_(?P<budget>{_BUDGET_SUFFIX})$"),
        "meta",
    ),
    "classic_meta_no_mask_random_skills": (
        re.compile(
            rf"^baselines_classic_meta_ppo_no_mask_random_skills_(?P<budget>{_BUDGET_SUFFIX})$"
        ),
        "meta",
    ),
    "classic_flat_rnn": (
        re.compile(rf"^baselines_classic_flat_ppo_rnn_(?P<budget>{_BUDGET_SUFFIX})$"),
        "flat_rnn",
    ),
    "classic_meta_rnn_no_mask": (
        re.compile(rf"^baselines_classic_meta_ppo_rnn_no_mask_(?P<budget>{_BUDGET_SUFFIX})$"),
        "meta_rnn",
    ),
}

GREEDY_EVAL_FILENAME = "greedy_eval_seed42_128ep.json"


def greedy_eval_filename(master_seed: int = 42, num_episodes: int = 128) -> str:
    return f"greedy_eval_seed{int(master_seed)}_{int(num_episodes)}ep.json"


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _has_checkpoint(run_dir: str) -> bool:
    weights = os.path.join(run_dir, "weights", "params.msgpack")
    if os.path.isfile(weights):
        return True
    ckpt = os.path.join(run_dir, "checkpoints")
    return os.path.isdir(ckpt) and bool(glob(os.path.join(ckpt, "*")))


def _runs_for_seed(policy_dir: str, seed: int) -> list[str]:
    out = []
    for run_dir in sorted(glob(os.path.join(policy_dir, "run_*"))):
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(cfg_path) or not _has_checkpoint(run_dir):
            continue
        cfg = _read_json(cfg_path)
        hyp = cfg.get("hyperparameters") or {}
        if int(hyp.get("seed", -1)) != seed:
            continue
        out.append(run_dir)
    return out


def discover_budget_runs(
    seed: int = 42,
    budgets: tuple[str, ...] | None = None,
) -> list[dict]:
    """Return eval jobs: policy_key, budget, run_dir, eval_policy, layer_size."""
    if budgets is None:
        budgets = CRAFTAX_BUDGET_LABELS
    budget_set = set(budgets)
    jobs: list[dict] = []

    for name in sorted(os.listdir(RUNS_ROOT)):
        policy_dir = os.path.join(RUNS_ROOT, name)
        if not os.path.isdir(policy_dir):
            continue
        for policy_key, (pattern, eval_policy) in POLICY_PATTERNS.items():
            m = pattern.match(name)
            if not m:
                continue
            budget = m.group("budget")
            if budget not in budget_set:
                continue

            if policy_key == "classic_flat":
                for run_dir in _runs_for_seed(policy_dir, seed):
                    cfg = _read_json(os.path.join(run_dir, "config.json"))
                    hyp = cfg.get("hyperparameters") or {}
                    width = int(hyp.get("layer_size", 512))
                    jobs.append(
                        {
                            "policy_key": f"flat_{width}",
                            "budget": budget,
                            "run_dir": run_dir,
                            "eval_policy": eval_policy,
                            "layer_size": width,
                        }
                    )
            else:
                run_dir = _runs_for_seed(policy_dir, seed)
                if not run_dir:
                    continue
                run_dir = run_dir[-1]
                cfg = _read_json(os.path.join(run_dir, "config.json"))
                hyp = cfg.get("hyperparameters") or {}
                key = policy_key.replace("extended_", "").replace("classic_", "")
                # Keep rnn suffixes: flat_rnn / meta_rnn_no_mask (not flat / meta_no_mask).
                jobs.append(
                    {
                        "policy_key": key,
                        "budget": budget,
                        "run_dir": run_dir,
                        "eval_policy": eval_policy,
                        "layer_size": int(hyp.get("layer_size", 512)),
                    }
                )

    jobs.sort(
        key=lambda j: (
            j["policy_key"],
            CRAFTAX_BUDGET_LABELS.index(j["budget"])
            if j["budget"] in CRAFTAX_BUDGET_LABELS
            else 999,
        )
    )
    return jobs


def greedy_eval_path(run_dir: str, *, master_seed: int = 42, num_episodes: int = 128) -> str:
    return os.path.join(run_dir, "evals", greedy_eval_filename(master_seed, num_episodes))


def checkpoint_exists(
    runs_policy: str,
    seed: int,
    *,
    layer_size: int | None = None,
) -> bool:
    """True if a checkpointed run exists for this policy dir + training seed."""
    policy_dir = os.path.join(RUNS_ROOT, runs_policy)
    for run_dir in _runs_for_seed(policy_dir, seed):
        cfg = _read_json(os.path.join(run_dir, "config.json"))
        hyp = cfg.get("hyperparameters") or {}
        if layer_size is not None and int(hyp.get("layer_size", -1)) != layer_size:
            continue
        return True
    return False


def train_use_action_mask(run_dir: str) -> bool | None:
    """Training mask flag from config, or None for non-meta / unknown."""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    cfg = _read_json(cfg_path)
    hyp = cfg.get("hyperparameters") or {}
    if "use_action_mask" in hyp:
        return bool(hyp["use_action_mask"])
    path = os.path.abspath(run_dir)
    if "no_mask" in path:
        return False
    if "meta_ppo_rnn" in path or "meta_ppo" in path:
        return True
    return None


def greedy_eval_matches_train(
    run_dir: str,
    eval_policy: str,
    *,
    master_seed: int = 42,
    num_episodes: int = 128,
) -> bool:
    """True if an existing greedy eval JSON matches train mask and has distributions."""
    path = greedy_eval_path(run_dir, master_seed=master_seed, num_episodes=num_episodes)
    if not os.path.isfile(path):
        # Backward-compat: default 128ep filename without explicit args.
        legacy = os.path.join(run_dir, "evals", GREEDY_EVAL_FILENAME)
        if num_episodes == 128 and master_seed == 42 and os.path.isfile(legacy):
            path = legacy
        else:
            return False
    ev = _read_json(path)
    if int(ev.get("num_episodes", -1)) != int(num_episodes):
        return False
    if num_episodes >= 256 and "episode_return_pct_max" not in ev:
        # Whisker-ready evals must store per-episode series.
        return False
    if eval_policy in ("meta", "meta_rnn"):
        train_mask = train_use_action_mask(run_dir)
        if train_mask is not None:
            if "eval_use_action_mask" not in ev:
                # Older evals always masked; stale when training was unmasked.
                if train_mask is not True:
                    return False
            elif bool(ev["eval_use_action_mask"]) != train_mask:
                return False
        if "meta_mode_distribution" not in ev:
            return False
    elif eval_policy in ("flat", "flat_rnn"):
        if "policy_action_distribution" not in ev:
            return False
    return True


def greedy_pct_from_eval(summary: dict, env_name: str) -> float:
    if "Classic" in env_name or "return_pct_max_achievement_only" in summary:
        return float(summary["return_pct_max_achievement_only"]["mean"])
    return float(summary["return_pct_max_achievement_reward"]["mean"])
