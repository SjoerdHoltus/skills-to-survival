"""Hyperparameter presets for official Craftax baseline runs."""

from __future__ import annotations

import copy
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ENV_EXTENDED = "Craftax-Symbolic-v1"
ENV_CLASSIC = "Craftax-Classic-Symbolic-v1"

META_NUM_ACTIONS = 24
META_LAYER_SIZE_DEFAULT = 128
META_LAYER_SIZE_MAX = 256
FLAT_LAYER_SIZE = 512

SHARED_1M = {
    "total_timesteps": 1_000_000,
    "num_envs": 256,
    "num_steps": 16,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.8,
    "update_epochs": 4,
    "num_minibatches": 8,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    "activation": "tanh",
    "anneal_lr": True,
    "use_optimistic_resets": True,
    "optimistic_reset_ratio": 16,
    "num_repeats": 1,
    "debug": True,
    "jit": True,
    "use_wandb": False,
    "save_policy": True,
    "train_icm": False,
    "use_e3b": False,
}


def _merge(**overrides):
    cfg = copy.deepcopy(SHARED_1M)
    cfg.update(overrides)
    return cfg


SHARED_10M = copy.deepcopy(SHARED_1M)
SHARED_10M["total_timesteps"] = 10_000_000


CRAFTAX_1M_FLAT_RNN_CLASSIC_PRESET = _merge(
    env_name=ENV_CLASSIC,
    layer_size=FLAT_LAYER_SIZE,
    runs_policy="baselines_classic_flat_ppo_rnn_1m",
)

CRAFTAX_1M_FLAT_RNN_PRESET = _merge(
    env_name=ENV_EXTENDED,
    layer_size=FLAT_LAYER_SIZE,
    runs_policy="baselines_extended_flat_ppo_rnn_1m",
)

CRAFTAX_1M_META_RNN_PRESET = _merge(
    env_name=ENV_CLASSIC,
    layer_size=META_LAYER_SIZE_DEFAULT,
    runs_policy="baselines_classic_meta_ppo_rnn_1m",
    use_action_mask=True,
    meta_num_actions=META_NUM_ACTIONS,
)

DEFAULT_NUM_ENVS = SHARED_1M["num_envs"]
DEFAULT_NUM_STEPS = SHARED_1M["num_steps"]


def num_updates_for_budget(
    total_timesteps: int, *, num_envs: int = DEFAULT_NUM_ENVS, num_steps: int = DEFAULT_NUM_STEPS
) -> int:
    return int(total_timesteps) // num_steps // num_envs


# Env-step budgets (k = thousands, m = millions of env steps).
CRAFTAX_BUDGET_STEPS: dict[str, int] = {
    "100k": 100_000,
    "500k": 500_000,
    "1m": 1_000_000,
    "2m": 2_000_000,
    "4m": 4_000_000,  # update-matched H=4 vs H=1@1M (244 meta updates)
    "5m": 5_000_000,
    "10m": 10_000_000,
    "20m": 20_000_000,
}

CRAFTAX_BUDGET_LABELS: tuple[str, ...] = tuple(CRAFTAX_BUDGET_STEPS.keys())


def resolve_budget(
    budget: str | None = None, total_timesteps: int | None = None
) -> tuple[str, int]:
    """Return (budget_label, total_timesteps). Default budget is 1m."""
    if budget is not None:
        label = budget.lower()
        if label.isdigit():  # raw step count, as returned for non-standard --total-timesteps
            return label, int(label)
        if label not in CRAFTAX_BUDGET_STEPS:
            raise ValueError(f"Unknown budget {budget!r}; choose from {list(CRAFTAX_BUDGET_STEPS)}")
        return label, CRAFTAX_BUDGET_STEPS[label]
    if total_timesteps is not None:
        steps = int(total_timesteps)
        for label, known in CRAFTAX_BUDGET_STEPS.items():
            if steps == known:
                return label, known
        if steps % 1_000_000 == 0:
            return f"{steps // 1_000_000}m", steps
        if steps % 1_000 == 0 and steps < 1_000_000:
            return f"{steps // 1_000}k", steps
        return str(steps), steps
    return "1m", CRAFTAX_BUDGET_STEPS["1m"]


def _merge_budget(budget: str, **overrides) -> dict:
    label, steps = resolve_budget(budget)
    cfg = copy.deepcopy(SHARED_1M)
    cfg["total_timesteps"] = steps
    cfg.update(overrides)
    cfg["budget_label"] = label
    cfg["target_num_updates"] = num_updates_for_budget(steps)
    return cfg


def flat_ppo_preset(
    *, env: str, budget: str | None = None, total_timesteps: int | None = None
) -> dict:
    label, _ = resolve_budget(budget, total_timesteps)
    suffix = label
    if env == "extended":
        return _merge_budget(
            suffix,
            env_name=ENV_EXTENDED,
            layer_size=FLAT_LAYER_SIZE,
            runs_policy=f"baselines_extended_ppo_{suffix}",
        )
    return _merge_budget(
        suffix,
        env_name=ENV_CLASSIC,
        layer_size=FLAT_LAYER_SIZE,
        runs_policy=f"baselines_classic_flat_ppo_{suffix}",
    )


def meta_ppo_preset(
    *,
    use_action_mask: bool,
    budget: str | None = None,
    total_timesteps: int | None = None,
    skill_horizon: int = 1,
    random_skills: bool = False,
) -> dict:
    label, _ = resolve_budget(budget, total_timesteps)
    horizon = max(1, int(skill_horizon))
    h_tag = "" if horizon == 1 else f"_h{horizon}"
    rs_tag = "_random_skills" if random_skills else ""
    if use_action_mask:
        return _merge_budget(
            label,
            env_name=ENV_CLASSIC,
            layer_size=META_LAYER_SIZE_DEFAULT,
            runs_policy=f"baselines_classic_meta_ppo{rs_tag}{h_tag}_{label}",
            use_action_mask=True,
            meta_num_actions=META_NUM_ACTIONS,
            skill_horizon=horizon,
            random_skills=bool(random_skills),
        )
    return _merge_budget(
        label,
        env_name=ENV_CLASSIC,
        layer_size=META_LAYER_SIZE_DEFAULT,
        runs_policy=f"baselines_classic_meta_ppo_no_mask{rs_tag}{h_tag}_{label}",
        use_action_mask=False,
        meta_num_actions=META_NUM_ACTIONS,
        skill_horizon=horizon,
        random_skills=bool(random_skills),
    )


def flat_rnn_preset(
    *, env: str = "classic", budget: str | None = None, total_timesteps: int | None = None
) -> dict:
    label, _ = resolve_budget(budget, total_timesteps)
    if env == "extended":
        return _merge_budget(
            label,
            env_name=ENV_EXTENDED,
            layer_size=FLAT_LAYER_SIZE,
            runs_policy=f"baselines_extended_flat_ppo_rnn_{label}",
        )
    return _merge_budget(
        label,
        env_name=ENV_CLASSIC,
        layer_size=FLAT_LAYER_SIZE,
        runs_policy=f"baselines_classic_flat_ppo_rnn_{label}",
    )


def meta_rnn_preset(
    *,
    use_action_mask: bool = False,
    budget: str | None = None,
    total_timesteps: int | None = None,
) -> dict:
    """Classic meta PPO-RNN. Default is unmasked (thesis comparison variant)."""
    label, _ = resolve_budget(budget, total_timesteps)
    if use_action_mask:
        return _merge_budget(
            label,
            env_name=ENV_CLASSIC,
            layer_size=META_LAYER_SIZE_DEFAULT,
            runs_policy=f"baselines_classic_meta_ppo_rnn_{label}",
            use_action_mask=True,
            meta_num_actions=META_NUM_ACTIONS,
        )
    return _merge_budget(
        label,
        env_name=ENV_CLASSIC,
        layer_size=META_LAYER_SIZE_DEFAULT,
        runs_policy=f"baselines_classic_meta_ppo_rnn_no_mask_{label}",
        use_action_mask=False,
        meta_num_actions=META_NUM_ACTIONS,
    )


def budget_update_chunks(
    budgets: tuple[str, ...] | list[str] | None = None,
    *,
    num_envs: int = DEFAULT_NUM_ENVS,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> list[tuple[str, int, int, int]]:
    """Ordered (label, cumulative_steps, delta_updates, cumulative_updates)."""
    labels = list(budgets) if budgets is not None else list(CRAFTAX_BUDGET_LABELS)
    out: list[tuple[str, int, int, int]] = []
    prev_updates = 0
    prev_steps = 0
    for label in labels:
        if label not in CRAFTAX_BUDGET_STEPS:
            raise ValueError(f"Unknown budget {label!r}")
        steps = CRAFTAX_BUDGET_STEPS[label]
        if steps <= prev_steps:
            raise ValueError(f"Budgets must be strictly increasing; got {label} after {prev_steps}")
        updates = num_updates_for_budget(steps, num_envs=num_envs, num_steps=num_steps)
        delta = updates - prev_updates
        if delta <= 0:
            raise ValueError(f"Non-positive update delta at {label}: {delta}")
        out.append((label, steps, delta, updates))
        prev_updates = updates
        prev_steps = steps
    return out


# Canonical frozen skill weights from the thesis meta run (062226).
DEFAULT_SKILL_WEIGHTS = {
    "find_wood": "checkpoints/skills/find_wood.npy",
    "drink_water": "checkpoints/skills/drink_water.npy",
    "find_food": "checkpoints/skills/find_food.npy",
    "monster_fight": "checkpoints/skills/monster_fight.npy",
    "collect_stone": "checkpoints/skills/collect_stone.npy",
    "collect_ore": "checkpoints/skills/collect_ore.npy",
}


def resolve_skill_paths(paths: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_SKILL_WEIGHTS)
    if paths:
        merged.update({k: v for k, v in paths.items() if v})
    out = {}
    for key, rel in merged.items():
        abs_path = rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Missing skill weights for {key}: {abs_path}")
        out[key] = abs_path
    return out


def clamp_meta_layer_size(layer_size: int) -> int:
    if layer_size > META_LAYER_SIZE_MAX:
        raise ValueError(f"Meta layer_size={layer_size} exceeds cap {META_LAYER_SIZE_MAX}")
    return int(layer_size)
