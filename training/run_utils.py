"""Run directory setup, checkpointing, and config saving.

All artifacts (weights, tensorboard event files, training log) live only under
``<run_dir>/``. Nothing is mirrored to a global folder.
"""

import datetime as dt
import json
import os

import jax.numpy as jnp

RUNS_DIR = "runs"


def _sanitize_policy_name(policy: str) -> str:
    """Safe folder name under runs/<policy>/ (no path separators)."""
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in (policy or "").strip())
    s = s.strip("_") or "misc"
    return s[:64]


def get_run_dir(policy: str = "misc"):
    """Return run directory path under runs/<policy>/run_<timestamp>/ and start_time string.

    Timestamp uses ``%m%d%y_%H%M%S`` (month, day, 2-digit year, wall time) so lexical
    order matches chronological order within the same century.

    Example: ``runs/monster_fight/run_050326_140021``
    """
    start_time = dt.datetime.now().strftime("%m%d%y_%H%M%S")
    policy_key = _sanitize_policy_name(policy)
    return os.path.join(RUNS_DIR, policy_key, f"run_{start_time}"), start_time


def setup_run_dir(run_dir):
    """Create run directory structure. Returns (run_weights_dir, run_tb_dir)."""
    run_weights_dir = os.path.join(run_dir, "weights")
    run_tb_dir = os.path.join(run_dir, "tb_logs")
    os.makedirs(run_weights_dir, exist_ok=True)
    os.makedirs(run_tb_dir, exist_ok=True)
    return run_weights_dir, run_tb_dir


def save_config(run_dir, config):
    """Write config dict to run_dir/config.json."""
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def infer_run_dir_from_weight_path(weight_path: str) -> str | None:
    """Return ``runs/<policy>/run_*/`` when *weight_path* lives under ``.../weights/``."""
    abs_path = os.path.abspath(weight_path)
    if not os.path.isfile(abs_path):
        return None
    weights_parent = os.path.dirname(abs_path)
    if os.path.basename(weights_parent) != "weights":
        return None
    run_dir = os.path.dirname(weights_parent)
    return run_dir if os.path.isdir(run_dir) else None


def save_checkpoint(es_mean, start_time, gen, mean_fitness, run_weights_dir):
    """Save rolling checkpoint to run_dir/weights/.

    The filename includes the generation number. The previous rolling file for
    this run is removed so only the latest is kept. Every 100 generations a
    permanent snapshot is also written (not removed on advance).
    """
    base_name = f"checkpoint_{start_time}_gen_{gen:04d}.npy"
    run_checkpoint_path = os.path.join(run_weights_dir, base_name)

    # Remove previous rolling checkpoints from this run (keep only latest).
    prefix = f"checkpoint_{start_time}_gen_"
    if os.path.isdir(run_weights_dir):
        for f in os.listdir(run_weights_dir):
            if f.startswith(prefix) and f != base_name:
                try:
                    os.remove(os.path.join(run_weights_dir, f))
                except OSError:
                    pass

    jnp.save(run_checkpoint_path, es_mean)

    if gen % 100 == 0:
        perm_name = f"checkpoint_{start_time}_{gen:04d}_{mean_fitness:.2f}.npy"
        jnp.save(os.path.join(run_weights_dir, perm_name), es_mean)

    return run_checkpoint_path


def save_best_es_weights(es_mean, start_time, run_weights_dir):
    """Save best-so-far ES mean vector when population mean fitness improves."""
    base_name = f"best_{start_time}.npy"
    run_path = os.path.join(run_weights_dir, base_name)
    jnp.save(run_path, es_mean)
    return run_path


def save_final(es_mean, start_time, mean_fitness, run_weights_dir):
    """Save final ES mean weights to run_dir/weights/. Returns path."""
    save_path = os.path.join(run_weights_dir, f"final_{start_time}_{mean_fitness:.2f}.npy")
    jnp.save(save_path, es_mean)
    return save_path


TRAINING_LOG_NAME = "training_log.txt"


def append_training_log_line(run_dir, line):
    """Append one line to training_log.txt (UTF-8). Creates file if needed."""
    path = os.path.join(run_dir, TRAINING_LOG_NAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def finalize_run(run_dir, run_tb_dir, start_time, log_lines, mean_fitness=None):
    """Rewrite training log from memory and optionally rename run dir with final fitness.

    ``run_tb_dir`` and ``start_time`` are kept as positional parameters for
    call-site backwards compatibility; nothing is copied outside ``run_dir``.
    """
    del run_tb_dir, start_time
    with open(os.path.join(run_dir, TRAINING_LOG_NAME), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    if mean_fitness is not None:
        new_run_dir = f"{run_dir}_fitness_{mean_fitness:.2f}"
        os.rename(run_dir, new_run_dir)
