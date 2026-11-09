"""Train once to the final budget, checkpointing at each intermediate budget."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from ppo.common import (
    _save_training_return_metrics,
    config_dict,
    extract_final_train_state,
    save_run_config,
    save_train_state,
)
from ppo.configs import (
    CRAFTAX_BUDGET_LABELS,
    budget_update_chunks,
    num_updates_for_budget,
)


def _concat_metrics(chunks: list) -> dict:
    if not chunks:
        raise ValueError("No metric chunks")
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *chunks)


def _runs_policy_for_budget(base_runs_policy: str, budget: str) -> str:
    """Swap trailing budget label on a runs_policy name."""
    parts = base_runs_policy.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in CRAFTAX_BUDGET_LABELS:
        return f"{parts[0]}_{budget}"
    return f"{base_runs_policy}_{budget}"


def run_chunked_budget_train(
    make_train_fn,
    ns: SimpleNamespace,
    *,
    start_time: str,
    budgets: tuple[str, ...] | list[str] | None = None,
    config_extra: dict | None = None,
    algorithm: str,
) -> dict:
    """One continuous run: scan deltas between budgets, save weights at each.

    Materializes ``runs/<policy>_<budget>/run_<start_time>/`` per budget so
    existing discovery/eval scripts keep working.
    """
    labels = list(budgets) if budgets is not None else list(CRAFTAX_BUDGET_LABELS)
    chunks = budget_update_chunks(labels, num_envs=int(ns.num_envs), num_steps=int(ns.num_steps))
    final_label, final_steps, _, final_updates = chunks[-1]

    ns = SimpleNamespace(**{**ns.__dict__, "total_timesteps": final_steps})
    cfg = config_dict(ns)
    cfg["ANNEAL_NUM_UPDATES"] = final_updates
    cfg["TOTAL_TIMESTEPS"] = final_steps
    # NUM_UPDATES inside make_train is overwritten from TOTAL; set final for anneal fallback.
    cfg["NUM_UPDATES"] = final_updates

    train = make_train_fn(cfg)
    if not hasattr(train, "init_runner") or not hasattr(train, "scan_updates"):
        raise TypeError("make_train must expose init_runner/scan_updates for chunked training")

    init_vmap = jax.jit(jax.vmap(train.init_runner))
    scan_cache: dict[int, object] = {}

    def scan_vmap(delta: int):
        if delta not in scan_cache:

            def _scan_one(rs):
                return train.scan_updates(rs, delta)

            scan_cache[delta] = jax.jit(jax.vmap(_scan_one))
        return scan_cache[delta]

    rng = jax.random.PRNGKey(int(cfg["SEED"]))
    rngs = jax.random.split(rng, int(cfg["NUM_REPEATS"]))

    t0 = time.time()
    runner_state = init_vmap(rngs)
    metric_chunks: list = []
    budget_dirs: dict[str, str] = {}

    base_policy = str(ns.runs_policy)
    for label, steps, delta, cum_updates in chunks:
        print(
            f"Chunk {label}: +{delta} updates -> {cum_updates} total ({steps:,} env steps)",
            flush=True,
        )
        runner_state, metric = scan_vmap(delta)(runner_state)
        jax.block_until_ready(runner_state)
        metric_chunks.append(metric)

        runs_policy = _runs_policy_for_budget(base_policy, label)
        run_dir = os.path.abspath(os.path.join("runs", runs_policy, f"run_{start_time}"))
        os.makedirs(run_dir, exist_ok=True)
        budget_dirs[label] = run_dir

        hyp = dict(ns.__dict__)
        hyp["total_timesteps"] = steps
        hyp["budget_label"] = label
        hyp["target_num_updates"] = cum_updates
        hyp["anneal_num_updates"] = final_updates
        hyp["chunked_budget_train"] = True
        payload = {
            "task": runs_policy,
            "algorithm": algorithm,
            "hyperparameters": hyp,
            "start_time": start_time,
            "chunked_from": base_policy,
            "budget_label": label,
        }
        if config_extra:
            payload.update(config_extra)
        save_run_config(run_dir, payload)

        train_state = extract_final_train_state({"runner_state": runner_state})
        save_train_state(train_state, run_dir, steps)

        metrics_so_far = _concat_metrics(metric_chunks)
        elapsed_so_far = time.time() - t0
        chunk_cfg = dict(cfg)
        chunk_cfg["TOTAL_TIMESTEPS"] = steps
        chunk_cfg["NUM_UPDATES"] = cum_updates
        _save_training_return_metrics(
            {"metrics": metrics_so_far, "runner_state": runner_state},
            run_dir,
            chunk_cfg,
            elapsed=elapsed_so_far,
        )
        summary = {
            "elapsed_sec": elapsed_so_far,
            "steps_per_sec": float(steps) / max(elapsed_so_far, 1e-6),
            "num_updates": cum_updates,
            "budget_label": label,
            "chunked_budget_train": True,
            "final_budget": final_label,
        }
        with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  saved {run_dir}", flush=True)

    elapsed = time.time() - t0
    print(
        f"Chunked training finished in {elapsed:.1f}s ({final_steps / max(elapsed, 1e-6):.0f} SPS)",
        flush=True,
    )
    return {
        "runner_state": runner_state,
        "budget_dirs": budget_dirs,
        "elapsed_sec": elapsed,
        "final_updates": final_updates,
    }


def check_budget_chunks_self() -> None:
    """ponytail: assert deltas sum to 20m updates; fails if budget table drifts."""
    chunks = budget_update_chunks()
    assert chunks[0][0] == "100k" and chunks[-1][0] == "20m"
    assert sum(c[2] for c in chunks) == chunks[-1][3]
    assert chunks[-1][3] == num_updates_for_budget(20_000_000)
    # Known table with default 256 envs × 16 steps.
    assert [c[2] for c in chunks] == [24, 98, 122, 244, 732, 1221, 2441]
    assert _runs_policy_for_budget("baselines_classic_flat_ppo_rnn_20m", "1m") == (
        "baselines_classic_flat_ppo_rnn_1m"
    )
    assert (
        _runs_policy_for_budget("baselines_classic_meta_ppo_rnn_no_mask_20m", "500k")
        == "baselines_classic_meta_ppo_rnn_no_mask_500k"
    )


if __name__ == "__main__":
    check_budget_chunks_self()
    print("chunked_train self-check ok")
