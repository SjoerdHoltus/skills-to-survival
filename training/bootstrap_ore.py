"""Collect-ore bootstrap (walled ore) and 15-D obs with stone blocker features.

``target_id`` selects block type and inventory channel at bootstrap/eval; training
fixes one target (see ``COLLECT_ORE_TRAINING_TARGET`` in config).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType

import configs as _cfg
from configs import COLLECT_ORE_OBS_SIZE, COLLECT_ORE_TARGET_FEATURES, OBS_SIZE
from training.bootstrap_target import (
    _CARDINAL_DELTAS,
    INV_IDX_COAL,
    INV_IDX_DIAMOND,
    INV_IDX_IRON,
    _strip_natural_ores,
    collect_target_features_from_symbolic_obs,
)
from training.monster_world_bootstrap import rebuild_mob_map_from_state

assert COLLECT_ORE_OBS_SIZE == COLLECT_ORE_TARGET_FEATURES + 4

_STONE_V = jnp.int32(BlockType.STONE.value)
_GRASS_V = jnp.int32(BlockType.GRASS.value)

TARGET_BLOCK_VALUES = jnp.asarray(
    [BlockType.COAL.value, BlockType.IRON.value, BlockType.DIAMOND.value],
    dtype=jnp.int32,
)
TARGET_INV_INDICES = jnp.asarray(
    [INV_IDX_COAL, INV_IDX_IRON, INV_IDX_DIAMOND],
    dtype=jnp.int32,
)
TARGET_CAPS = jnp.asarray(
    [
        int(_cfg.COLLECT_ORE_MAX_COAL),
        int(_cfg.COLLECT_ORE_MAX_IRON),
        int(_cfg.COLLECT_ORE_MAX_DIAMOND),
    ],
    dtype=jnp.int32,
)

_CARDINAL_STEPS = jnp.asarray(
    [
        [0, -1],
        [0, +1],
        [-1, 0],
        [+1, 0],
    ],
    dtype=jnp.int32,
)


def target_id_from_name(name: str) -> int:
    names = _cfg.COLLECT_ORE_TARGET_NAMES
    try:
        return names.index(name.lower())
    except ValueError as exc:
        raise ValueError(f"Unknown ore target {name!r}; expected one of {names}") from exc


DIAMOND_TARGET_ID = jnp.int32(target_id_from_name("diamond"))


def wall_count_for_curriculum(
    rng: jax.Array, gen: jnp.ndarray, num_gens: jnp.ndarray
) -> jnp.ndarray:
    """Curriculum over target shell depth: 1 layer early, deeper shells later."""
    gen_f = gen.astype(jnp.float32)
    ng = jnp.maximum(num_gens.astype(jnp.float32), 1.0)
    progress = gen_f / jnp.maximum(ng - 1.0, 1.0)

    p1 = jnp.float32(_cfg.COLLECT_ORE_WALL_CURRICULUM_PHASE1_FRACTION)
    p2_width = jnp.float32(_cfg.COLLECT_ORE_WALL_CURRICULUM_PHASE2_FRACTION)
    p2 = jnp.clip((progress - p1) / jnp.maximum(p2_width, jnp.float32(1e-6)), 0.0, 1.0) * 0.45
    late_width = jnp.maximum(1.0 - p1 - p2_width, jnp.float32(1e-6))
    p3 = jnp.clip((progress - p1 - p2_width) / late_width, 0.0, 1.0) * 0.30
    u = jax.random.uniform(rng, (), dtype=jnp.float32)
    return jnp.where(u < p3, jnp.int32(3), jnp.where(u < p3 + p2, jnp.int32(2), jnp.int32(1)))


def _stone_blocker_features_from_map(
    blocks: jnp.ndarray,
    pr_i: jnp.ndarray,
    pc_i: jnp.ndarray,
    pr: jnp.float32,
    pc: jnp.float32,
    ore_r: jnp.ndarray,
    ore_c: jnp.ndarray,
    has_target: jnp.ndarray,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """4-D stone blocker slice: block_dx, block_dy, adj_stone, face_stone."""
    stone_here = blocks[..., _STONE_V.astype(jnp.int32)] > jnp.float32(0.5)

    rows_f = jnp.arange(blocks.shape[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(blocks.shape[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)

    ore_manh = manh[ore_r, ore_c]
    between = jnp.logical_and(stone_here, jnp.logical_and(manh > 0.0, manh < ore_manh))
    block_key = jnp.where(between, manh, jnp.float32(1e9))
    flat_b = jnp.argmin(block_key.reshape(-1))
    br = flat_b // jnp.int32(blocks.shape[1])
    bc = flat_b % jnp.int32(blocks.shape[1])
    has_block = jnp.any(between).astype(jnp.float32)

    fov_r = jnp.float32(_cfg.COLLECT_ORE_FOV_NORM_CHEBYSHEV_R)
    block_dx = jnp.clip((br.astype(jnp.float32) - pr) / fov_r, -1.0, 1.0) * has_block * has_target
    block_dy = jnp.clip((bc.astype(jnp.float32) - pc) / fov_r, -1.0, 1.0) * has_block * has_target

    adj_stone = jnp.any((manh == 1.0) & stone_here).astype(jnp.float32)

    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_stone = stone_here[fr, fc].astype(jnp.float32)

    return jnp.stack([block_dx, block_dy, adj_stone, face_stone], axis=0)


def collect_ore_features_from_symbolic_obs(
    symbolic_obs: jnp.ndarray, target_id: jnp.ndarray
) -> jnp.ndarray:
    """15-D: 11-D target ore features + 4-D stone blocker features."""
    tid = target_id.astype(jnp.int32)
    block_v = TARGET_BLOCK_VALUES[tid]
    inv_idx = TARGET_INV_INDICES[tid]
    cap = TARGET_CAPS[tid].astype(jnp.float32)

    base = collect_target_features_from_symbolic_obs(
        symbolic_obs,
        block_v,
        inv_idx,
        cap,
        float(_cfg.COLLECT_ORE_FOV_NORM_CHEBYSHEV_R),
    )

    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    _NUM_BLOCK_TYPES = int(len(BlockType))
    _NUM_MOB_CHANNELS = 4
    _SYMBOLIC_CHANNELS_PER_CELL = _NUM_BLOCK_TYPES + _NUM_MOB_CHANNELS
    _SYMBOLIC_MAP_FLAT = int(OBS_DIM[0] * OBS_DIM[1] * _SYMBOLIC_CHANNELS_PER_CELL)
    _SYMBOLIC_DIR_START = _SYMBOLIC_MAP_FLAT + 12 + 4

    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    target_here = blocks[..., block_v.astype(jnp.int32)] > jnp.float32(0.5)

    pr_i = jnp.asarray(OBS_DIM[0] // 2, dtype=jnp.int32)
    pc_i = jnp.asarray(OBS_DIM[1] // 2, dtype=jnp.int32)
    pr = jnp.float32(OBS_DIM[0] // 2)
    pc = jnp.float32(OBS_DIM[1] // 2)

    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    big = jnp.float32(1e9)
    target_key = jnp.where(target_here, manh, big)
    flat_t = jnp.argmin(target_key.reshape(-1))
    t_r = flat_t // jnp.int32(OBS_DIM[1])
    t_c = flat_t % jnp.int32(OBS_DIM[1])
    has_target = jnp.any(target_here).astype(jnp.float32)

    dir_oh = sym[_SYMBOLIC_DIR_START : _SYMBOLIC_DIR_START + 4]
    direction = jnp.argmax(dir_oh) + 1

    blockers = _stone_blocker_features_from_map(
        blocks, pr_i, pc_i, pr, pc, t_r, t_c, has_target, direction
    )
    return jnp.concatenate([base, blockers], axis=0)


assert (
    int(collect_ore_features_from_symbolic_obs(jnp.zeros(OBS_SIZE), jnp.int32(0)).shape[0])
    == COLLECT_ORE_OBS_SIZE
)


def collect_ore_has_target(obs: jnp.ndarray) -> jnp.ndarray:
    """Target ore visible in symbolic FOV (15-D obs dim 2)."""
    return (obs[..., int(_cfg.COLLECT_ORE_FEAT_HAS_TARGET)] >= jnp.float32(0.5)).astype(jnp.float32)


def make_collect_ore_obs(env, state, target_id: jnp.ndarray) -> jnp.ndarray:
    return collect_ore_features_from_symbolic_obs(env.get_obs(state), target_id)


def _strip_ambient_stone(state_map: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(state_map == _STONE_V, _GRASS_V, state_map)


def _clear_hostiles(state):
    zombies = state.zombies.replace(
        position=jnp.zeros_like(state.zombies.position),
        health=jnp.zeros_like(state.zombies.health),
        mask=jnp.zeros_like(state.zombies.mask),
        attack_cooldown=jnp.zeros_like(state.zombies.attack_cooldown),
    )
    skeletons = state.skeletons.replace(
        position=jnp.zeros_like(state.skeletons.position),
        health=jnp.zeros_like(state.skeletons.health),
        mask=jnp.zeros_like(state.skeletons.mask),
        attack_cooldown=jnp.zeros_like(state.skeletons.attack_cooldown),
    )
    cows = state.cows.replace(
        position=jnp.zeros_like(state.cows.position),
        health=jnp.zeros_like(state.cows.health),
        mask=jnp.zeros_like(state.cows.mask),
        attack_cooldown=jnp.zeros_like(state.cows.attack_cooldown),
    )
    arrows = state.arrows.replace(
        position=jnp.zeros_like(state.arrows.position),
        health=jnp.zeros_like(state.arrows.health),
        mask=jnp.zeros_like(state.arrows.mask),
    )
    return state.replace(zombies=zombies, skeletons=skeletons, cows=cows, arrows=arrows)


def _delta_to_direction(delta: jnp.ndarray) -> jnp.ndarray:
    """Map [dr, dc] in {-1,0,1} to Craftax direction index 1..4."""
    dr, dc = delta[0], delta[1]
    return jnp.where(
        dr < 0,
        jnp.int32(3),
        jnp.where(
            dr > 0,
            jnp.int32(4),
            jnp.where(dc < 0, jnp.int32(1), jnp.int32(2)),
        ),
    )


def _symbolic_fov_max_cardinal_dist(delta: jnp.ndarray) -> jnp.ndarray:
    """Max cardinal steps from crop center (player) that remain inside ``OBS_DIM``."""
    max_row = jnp.int32((OBS_DIM[0] - 1) // 2)
    max_col = jnp.int32((OBS_DIM[1] - 1) // 2)
    is_vertical = jnp.abs(delta[0]) > 0
    return jnp.where(is_vertical, max_row, max_col)


def _map_max_cardinal_dist(
    player: jnp.ndarray, delta: jnp.ndarray, h: jnp.ndarray, w: jnp.ndarray
) -> jnp.ndarray:
    """Max cardinal steps from ``player`` before leaving the world map."""
    return jnp.where(
        delta[0] < 0,
        player[0],
        jnp.where(
            delta[0] > 0,
            h - 1 - player[0],
            jnp.where(
                delta[1] < 0,
                player[1],
                jnp.where(delta[1] > 0, w - 1 - player[1], jnp.int32(0)),
            ),
        ),
    )


def _place_layered_ore_layout(
    state_map: jnp.ndarray,
    player: jnp.ndarray,
    rng: jax.Array,
    ore_value: jnp.ndarray,
    shell_count: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Place ore within symbolic FOV and wrap it in stone shells.

    Distance is sampled along a cardinal from the player. It must fit both the
    symbolic crop (``OBS_DIM``) and the world map; otherwise the ore block never
    appears in ``has_target`` (policy FOV) even though shells may be placed.
    """
    rng_dir, rng_dist = jax.random.split(rng)
    # Train on left/right ore only (matches natural side-mining; avoids vertical
    # FOV-edge layouts that never appear in procedural worlds).
    dir_idx = jax.random.randint(rng_dir, (), 0, 2)
    delta = _CARDINAL_STEPS[dir_idx]
    layers = jnp.clip(shell_count.astype(jnp.int32), 1, 3)

    h, w = state_map.shape
    fov_cap = _symbolic_fov_max_cardinal_dist(delta)
    map_cap = _map_max_cardinal_dist(player, delta, jnp.int32(h), jnp.int32(w))
    usable_cap = jnp.minimum(fov_cap, map_cap)

    dist_lo_cfg = jnp.int32(_cfg.COLLECT_ORE_ORE_DISTANCE_MIN)
    dist_hi_cfg = jnp.int32(_cfg.COLLECT_ORE_ORE_DISTANCE_MAX)
    dist_lo = jnp.minimum(dist_lo_cfg, usable_cap)
    dist_hi = jnp.minimum(dist_hi_cfg, usable_cap)
    dist_lo = jnp.minimum(dist_lo, dist_hi)

    dist = jax.random.randint(rng_dist, (), minval=dist_lo, maxval=dist_hi + 1)

    ore_r = player[0] + delta[0] * dist
    ore_c = player[1] + delta[1] * dist
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    cheb = jnp.maximum(jnp.abs(rr - ore_r), jnp.abs(cc - ore_c))
    shell = jnp.logical_and(cheb > 0, cheb <= layers)
    player_cell = jnp.logical_and(rr == player[0], cc == player[1])
    shell = jnp.logical_and(shell, jnp.logical_not(player_cell))
    new_map = jnp.where(shell, _STONE_V.astype(state_map.dtype), state_map)
    # ``usable_cap`` guarantees in-bounds cardinal placement; always paint ore.
    new_map = new_map.at[ore_r, ore_c].set(ore_value.astype(new_map.dtype))
    face_dir = _delta_to_direction(delta)
    return new_map, face_dir


def bootstrap_ore_state(
    state,
    boot_rng: jax.Array,
    _params,
    target_id: jnp.ndarray,
    wall_count: jnp.ndarray | None = None,
    curriculum_gen: jnp.ndarray | None = None,
    curriculum_num_gens: jnp.ndarray | None = None,
) -> object:
    """Clear hostiles, strip ores/stone, dual pickaxes, layered ore layout."""
    boot_rng_layout, boot_rng_wall = jax.random.split(boot_rng)

    if wall_count is None:
        assert curriculum_gen is not None and curriculum_num_gens is not None
        wall_count = wall_count_for_curriculum(boot_rng_wall, curriculum_gen, curriculum_num_gens)
    else:
        wall_count = wall_count.astype(jnp.int32)

    staged = _clear_hostiles(state)
    player = staged.player_position.astype(jnp.int32)
    ore_value = TARGET_BLOCK_VALUES[target_id.astype(jnp.int32)]

    cleared = _strip_natural_ores(staged.map)
    cleared = _strip_ambient_stone(cleared)
    new_map, face_dir = _place_layered_ore_layout(
        cleared, player, boot_rng_layout, ore_value, wall_count
    )

    staged_map = staged.replace(map=new_map, player_direction=face_dir)
    mob_map = rebuild_mob_map_from_state(staged_map)

    z = jnp.int32(0)
    tid = target_id.astype(jnp.int32)
    is_diamond = tid == jnp.int32(2)
    inventory = staged_map.inventory.replace(
        wood_pickaxe=jnp.int32(1),
        stone_pickaxe=jnp.int32(1),
        iron_pickaxe=jnp.where(is_diamond, jnp.int32(1), jnp.int32(0)),
        coal=z,
        iron=z,
        diamond=z,
    )
    return staged_map.replace(mob_map=mob_map, inventory=inventory)
