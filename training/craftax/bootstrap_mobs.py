"""Mob injection bootstraps for full Craftax monster_fight / find_food training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax.constants import BlockType

import configs as _cfg
from training.craftax.bootstrap_target import symbolic_fov_world_mask
from training.craftax.state_helpers import clear_mobs_on_level, level_map, set_level_map

_GRASS_V = jnp.int32(BlockType.GRASS.value)


def _grass_arena(state):
    state = clear_mobs_on_level(state)
    mmap = jnp.full_like(level_map(state), _GRASS_V)
    return set_level_map(state, mmap)


def _sample_hostile_positions(rng, player, num_slots, h, w):
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    fov = symbolic_fov_world_mask((h, w), player)
    cheb = jnp.maximum(jnp.abs(rr - player[0]), jnp.abs(cc - player[1]))
    valid = fov & (cheb >= 1) & (cheb <= jnp.int32(_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R))
    scores = jax.random.uniform(rng, (h, w))
    scores = jnp.where(valid, scores, jnp.float32(1e9))
    flat = scores.reshape(-1)
    order = jnp.argsort(flat)
    idx = order[:num_slots]
    rows = idx // w
    cols = idx % w
    pos = jnp.stack([rows, cols], axis=-1)
    valid_slot = flat[idx] < jnp.float32(1e8)
    return pos, valid_slot


def apply_melee_bootstrap(
    state,
    boot_rng: jax.Array,
    env_params,
    *,
    static_positions: bool = False,
    equip_sword: bool = True,
    melee_type_id: int = 0,
):
    """Inject melee hostiles. ``melee_type_id`` is Craftax Extended type (0=zombie)."""
    import numpy as np
    from craftax.craftax.constants import MOB_TYPE_HEALTH_MAPPING, MobType

    state = _grass_arena(state)
    lvl = state.player_level.astype(jnp.int32)
    player = state.player_position.astype(jnp.int32)
    h, w = level_map(state).shape

    num_min = int(_cfg.MONSTER_FIGHT_NUM_ZOMBIES_MIN)
    num_max = int(_cfg.MONSTER_FIGHT_NUM_ZOMBIES_MAX)
    rng_count, rng_pos = jax.random.split(boot_rng)
    count = jax.random.randint(rng_count, (), minval=num_min, maxval=num_max + 1)
    max_slots = state.melee_mobs.mask.shape[1]
    count = jnp.minimum(count, jnp.int32(max_slots))

    positions, valids = _sample_hostile_positions(rng_pos, player, max_slots, h, w)
    slot_idx = jnp.arange(max_slots, dtype=jnp.int32)
    active = valids & (slot_idx < count)

    # Python ints only — must stay concrete under jit (no int(traced HP)).
    tid = int(melee_type_id)
    health_table = np.asarray(MOB_TYPE_HEALTH_MAPPING)
    if tid < 0 or tid >= health_table.shape[0]:
        raise ValueError(f"melee_type_id={tid} out of range")
    hp = int(health_table[tid, int(MobType.MELEE.value)])
    health = jnp.where(active, jnp.int32(hp), jnp.int32(0)).astype(state.melee_mobs.health.dtype)
    cooldown = jnp.where(
        active & jnp.asarray(static_positions, dtype=jnp.bool_),
        jnp.int32(1000),
        jnp.int32(0),
    )
    type_ids = jnp.where(
        active,
        jnp.int32(tid),
        jnp.int32(0),
    ).astype(state.melee_mobs.type_id.dtype)

    melee = state.melee_mobs
    melee = melee.replace(
        position=melee.position.at[lvl].set(positions.astype(melee.position.dtype)),
        mask=melee.mask.at[lvl].set(active),
        health=melee.health.at[lvl].set(health),
        type_id=melee.type_id.at[lvl].set(type_ids),
        attack_cooldown=melee.attack_cooldown.at[lvl].set(
            jnp.full((max_slots,), cooldown, dtype=melee.attack_cooldown.dtype)
        ),
    )
    state = state.replace(melee_mobs=melee)
    if equip_sword:
        state = state.replace(inventory=state.inventory.replace(sword=jnp.int32(1)))
    return state


def apply_passive_bootstrap(
    state,
    boot_rng: jax.Array,
    env_params,
    *,
    static_positions: bool = False,
    equip_sword: bool = True,
):
    state = _grass_arena(state)
    lvl = state.player_level.astype(jnp.int32)
    player = state.player_position.astype(jnp.int32)
    h, w = level_map(state).shape

    num_min = int(_cfg.FIND_FOOD_NUM_COWS_MIN)
    num_max = int(_cfg.FIND_FOOD_NUM_COWS_MAX)
    rng_count, rng_pos = jax.random.split(boot_rng)
    count = jax.random.randint(rng_count, (), minval=num_min, maxval=num_max + 1)
    max_slots = state.passive_mobs.mask.shape[1]
    count = jnp.minimum(count, jnp.int32(max_slots))

    positions, valids = _sample_hostile_positions(rng_pos, player, max_slots, h, w)
    slot_idx = jnp.arange(max_slots, dtype=jnp.int32)
    active = valids & (slot_idx < count)

    c_health = getattr(env_params, "cow_health", jnp.int32(3))
    health = jnp.where(
        active, jnp.asarray(c_health, dtype=state.passive_mobs.health.dtype), jnp.int32(0)
    )
    cooldown = jnp.where(
        active & jnp.asarray(static_positions, dtype=jnp.bool_),
        jnp.int32(1000),
        jnp.int32(0),
    )

    passive = state.passive_mobs
    passive = passive.replace(
        position=passive.position.at[lvl].set(positions.astype(passive.position.dtype)),
        mask=passive.mask.at[lvl].set(active),
        health=passive.health.at[lvl].set(health),
        type_id=passive.type_id.at[lvl].set(jnp.zeros((max_slots,), dtype=passive.type_id.dtype)),
        attack_cooldown=passive.attack_cooldown.at[lvl].set(
            jnp.full((max_slots,), cooldown, dtype=passive.attack_cooldown.dtype)
        ),
    )
    state = state.replace(passive_mobs=passive)
    if equip_sword:
        state = state.replace(inventory=state.inventory.replace(sword=jnp.int32(1)))
    return state
