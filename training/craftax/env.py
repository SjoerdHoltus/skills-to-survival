"""Shared full-Craftax env factory for OpenES skill training."""

from __future__ import annotations

from craftax.craftax_env import make_craftax_env_from_name

from configs.craftax_base import CRAFTAX_ENV_NAME


def make_skill_env():
    """Natural full Craftax symbolic env; mobs cleared in bootstraps instead of spawn params."""
    env = make_craftax_env_from_name(CRAFTAX_ENV_NAME, auto_reset=False)
    env_params = env.default_params.replace(mob_despawn_distance=10_000)
    return env, env_params
