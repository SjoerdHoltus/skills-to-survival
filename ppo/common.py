"""Shared helpers for running official Craftax baselines."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import os
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from ppo.configs import CRAFTAX_BUDGET_LABELS


def namespace_from_preset(preset: dict, **overrides) -> SimpleNamespace:
    merged = dict(preset)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    if merged.get("seed") is None:
        merged["seed"] = int(np.random.randint(2**31))
    return SimpleNamespace(**merged)


def config_dict(ns: SimpleNamespace) -> dict:
    return {k.upper(): v for k, v in ns.__dict__.items()}


def get_run_dir(runs_policy: str) -> tuple[str, str]:

    from training.run_utils import get_run_dir as _get_run_dir

    run_dir, start_time = _get_run_dir(runs_policy)
    return os.path.abspath(run_dir), start_time


def save_train_state(train_state, run_dir: str, total_timesteps: int) -> str:
    run_dir = os.path.abspath(run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    from flax import serialization

    params_bytes = serialization.to_bytes(jax.device_get(train_state.params))
    params_path = os.path.join(weights_dir, "params.msgpack")
    with open(params_path, "wb") as f:
        f.write(params_bytes)

    checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=3, create=True)
    manager = CheckpointManager(ckpt_dir, checkpointer, options)
    save_args = orbax_utils.save_args_from_target(train_state)
    manager.save(
        int(total_timesteps),
        train_state,
        save_kwargs={"save_args": save_args},
    )
    return ckpt_dir


def load_params(run_dir: str):
    from flax import serialization

    params_path = os.path.join(os.path.abspath(run_dir), "weights", "params.msgpack")
    if os.path.isfile(params_path):
        with open(params_path, "rb") as f:
            return serialization.from_bytes(None, f.read())

    ckpt_dir = os.path.join(os.path.abspath(run_dir), "checkpoints")
    options = CheckpointManagerOptions(create=False)
    manager = CheckpointManager(ckpt_dir, PyTreeCheckpointer(), options)
    step = manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    restored = manager.restore(step)
    item = restored[0] if isinstance(restored, list) else restored
    if isinstance(item, dict) and "params" in item:
        return item["params"]
    if hasattr(item, "params"):
        return item.params
    raise TypeError(f"Unexpected checkpoint payload type: {type(item)}")


def extract_final_train_state(out: dict):
    """Last train state from official PPO / meta ``make_train`` output."""
    state = out["runner_state"][0]

    def pick(x):
        x = jnp.asarray(x)
        if x.ndim >= 1 and x.shape[0] == 1:
            return x[0]
        return x

    return jax.tree.map(pick, state)


def _metric_series(metrics: dict, key: str) -> np.ndarray:
    """Per-update series for repeat 0 (official NUM_REPEATS=1)."""
    raw = metrics[key][0]
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float64, copy=False)
    else:
        arr = np.asarray(jax.device_get(raw), dtype=np.float64)
    return arr.reshape(-1)


def _achievement_metric_keys(metrics: dict) -> list[str]:
    return sorted(k for k in metrics if k.lower().startswith("achievements/"))


def _achievement_name_from_metric_key(key: str) -> str:
    return key.split("/", 1)[-1].upper()


def _per_achievement_success_rates(metrics: dict, idx: slice) -> dict[str, float]:
    """Mean training success rate (0–100) per achievement over selected updates."""
    out: dict[str, float] = {}
    for key in _achievement_metric_keys(metrics):
        series = _metric_series(metrics, key)[idx]
        out[_achievement_name_from_metric_key(key)] = float(series.mean()) if series.size else 0.0
    return out


def _per_update_achievement_success_rates(metrics: dict) -> dict[str, list[float]]:
    """Full per-update curves keyed by achievement name (values in [0, 100])."""
    keys = _achievement_metric_keys(metrics)
    if not keys:
        return {}
    n_updates = _metric_series(metrics, keys[0]).size
    out: dict[str, list[float]] = {}
    for key in keys:
        series = _metric_series(metrics, key)
        if series.size != n_updates:
            continue
        out[_achievement_name_from_metric_key(key)] = [float(x) for x in series]
    return out


def _save_training_return_metrics(
    out: dict, run_dir: str, cfg: dict, *, elapsed: float, tail_updates: int = 10
) -> None:
    metrics = out["metrics"] if "metrics" in out else out["metric"]
    returns = _metric_series(metrics, "returned_episode_returns")
    lengths = _metric_series(metrics, "returned_episode_lengths")
    max_reward = 22.0 if "Classic" in cfg["ENV_NAME"] else 226.0
    last_ret = float(returns[-1])
    tail = max(1, min(tail_updates, returns.size))
    tail_ret = float(returns[-tail:].mean())
    last_ach = _per_achievement_success_rates(metrics, slice(-1, None))
    tail_ach = _per_achievement_success_rates(metrics, slice(-tail, None))
    per_update_ach = _per_update_achievement_success_rates(metrics)
    payload = {
        "run_dir": os.path.abspath(run_dir),
        "env_name": cfg["ENV_NAME"],
        "seed": int(cfg["SEED"]),
        "total_timesteps": int(cfg["TOTAL_TIMESTEPS"]),
        "num_updates": int(returns.size),
        "max_achievement_reward_reference": max_reward,
        "training_elapsed_sec": elapsed,
        "last_update": {
            "episode_return": last_ret,
            "return_pct_max": last_ret / max_reward * 100.0,
            "episode_length": float(lengths[-1]),
            "per_achievement_success_rate": last_ach,
        },
        "tail_mean": {
            "updates": tail,
            "episode_return": tail_ret,
            "return_pct_max": tail_ret / max_reward * 100.0,
            "per_achievement_success_rate": tail_ach,
        },
    }
    if per_update_ach:
        payload["per_update_achievement_success_rate"] = per_update_ach
    path = os.path.join(run_dir, "training_return_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_official_train(make_train_fn, ns: SimpleNamespace, run_dir: str) -> dict:
    """JIT train via an official ``make_train`` factory and save checkpoints locally."""
    cfg = config_dict(ns)
    rng = jax.random.PRNGKey(int(cfg["SEED"]))
    rngs = jax.random.split(rng, int(cfg["NUM_REPEATS"]))

    train_jit = jax.jit(make_train_fn(cfg))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    elapsed = time.time() - t0
    sps = float(cfg["TOTAL_TIMESTEPS"]) / max(elapsed, 1e-6)
    print(f"Training finished in {elapsed:.1f}s ({sps:.0f} SPS)")

    train_state = extract_final_train_state(out)
    ckpt_dir = save_train_state(train_state, run_dir, cfg["TOTAL_TIMESTEPS"])

    summary = {
        "elapsed_sec": elapsed,
        "steps_per_sec": sps,
        "checkpoint_dir": ckpt_dir,
        "num_updates": int(cfg["NUM_UPDATES"]),
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _save_training_return_metrics(out, run_dir, cfg, elapsed=elapsed)
    return {"out": out, "train_state": train_state, "summary": summary}


def add_shared_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budget",
        choices=CRAFTAX_BUDGET_LABELS,
        default=None,
        help="Training budget label (env steps: 100k, 500k, 1m, …)",
    )
    parser.add_argument("--total-timesteps", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--layer-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-dir", default=None, help="Override run directory")


def preset_overrides(args, preset: dict) -> dict:
    return {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "layer_size": args.layer_size,
        "lr": args.lr,
    }


def save_run_config(run_dir: str, payload: dict) -> None:
    from training.run_utils import save_config

    save_config(run_dir, payload)
