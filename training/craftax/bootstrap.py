"""Frozen-skill observation builders for full Craftax symbolic env."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax.constants import OBS_DIM, BlockType

import configs as _cfg
from training.craftax.symbolic_layout import (
    _MELEE_MOB_REL,
    _MOB_SLICE_START,
    _NUM_BLOCK_TYPES,
    _PASSIVE_MOB_REL,
    _PROJECTILE_MOB_REL,
    _SYMBOLIC_CHANNELS_PER_CELL,
    _SYMBOLIC_MAP_FLAT,
    _SYMBOLIC_PLAYER_COL,
    _SYMBOLIC_PLAYER_ROW,
)

_CARDINAL_DELTAS = jnp.asarray(
    [[0, 0], [0, -1], [0, +1], [-1, 0], [+1, 0]],
    dtype=jnp.int32,
)

_WALKABLE_BLOCKS: tuple[int, ...] = (
    BlockType.GRASS.value,
    BlockType.PATH.value,
    BlockType.SAND.value,
    BlockType.ICE_GRASS.value,
    BlockType.FIRE_GRASS.value,
)


def _blocks_from_symbolic(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    return map3[..., :_NUM_BLOCK_TYPES]


def _mob_layers_from_symbolic(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    return map3[..., _MOB_SLICE_START:]


def _symbolic_walkable_dot_vec() -> jnp.ndarray:
    v = jnp.zeros((_NUM_BLOCK_TYPES,), dtype=jnp.float32)
    for b in _WALKABLE_BLOCKS:
        v = v.at[b].set(1.0)
    return v


def _cardinal_blocked(blocks: jnp.ndarray) -> jnp.ndarray:
    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    wv = _symbolic_walkable_dot_vec()

    def _blocked(dr: int, dc: int) -> jnp.ndarray:
        r = pr_i + jnp.int32(dr)
        c = pc_i + jnp.int32(dc)
        oh = blocks[r, c, :]
        return (jnp.dot(oh, wv) < 0.5).astype(jnp.float32)

    return jnp.stack(
        [_blocked(0, -1), _blocked(0, 1), _blocked(-1, 0), _blocked(1, 0)],
        axis=0,
    )


def _direction_one_hot(state) -> jnp.ndarray:
    return jax.nn.one_hot(state.player_direction - 1, num_classes=4)


def _target_gather_features(
    symbolic_obs: jnp.ndarray,
    state,
    target_block_value: jnp.ndarray,
    inv_count: jnp.ndarray,
    inv_norm_scale: float,
    fov_norm_chebyshev_r: float,
    *,
    include_health: bool = False,
) -> jnp.ndarray:
    blocks = _blocks_from_symbolic(symbolic_obs)
    bv = target_block_value.astype(jnp.int32)
    target_here = blocks[..., bv] > jnp.float32(0.5)

    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)

    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)

    big = jnp.float32(1e9)
    target_key = jnp.where(target_here, manh, big)
    flat_t = jnp.argmin(target_key.reshape(-1))
    t_r = flat_t // jnp.int32(OBS_DIM[1])
    t_c = flat_t % jnp.int32(OBS_DIM[1])
    has_target = jnp.any(target_here).astype(jnp.float32)
    nearest_manh_crop = manh[t_r, t_c]
    d_row = (t_r.astype(jnp.float32) - pr) * has_target
    d_col = (t_c.astype(jnp.float32) - pc) * has_target
    fov_r = jnp.float32(fov_norm_chebyshev_r)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    dist_sq_norm = jnp.where(
        has_target > 0.5,
        (nearest_manh_crop / jnp.maximum(crop_max_manh, 1.0)) ** 2,
        jnp.float32(1.0),
    )

    blocked = _cardinal_blocked(blocks)
    adj_target = jnp.any((manh == 1.0) & target_here).astype(jnp.float32)

    dir_oh = _direction_one_hot(state)
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_target = target_here[fr, fc].astype(jnp.float32)

    scale = jnp.maximum(jnp.float32(inv_norm_scale), jnp.float32(1.0))
    inv_norm = jnp.clip(inv_count.astype(jnp.float32) / scale, 0.0, 1.0)

    feats = [
        dx_norm,
        dy_norm,
        has_target,
        blocked[0],
        blocked[1],
        blocked[2],
        blocked[3],
        adj_target,
        face_target,
        dist_sq_norm,
        inv_norm,
    ]
    if include_health:
        health_norm = jnp.clip(
            state.player_health.astype(jnp.float32) / jnp.float32(10.0),
            0.0,
            1.0,
        )
        feats.append(health_norm)
    return jnp.stack(feats, axis=0)


def find_wood_features_from_symbolic_obs(symbolic_obs: jnp.ndarray, state) -> jnp.ndarray:
    blocks = _blocks_from_symbolic(symbolic_obs)
    tree_here = blocks[..., BlockType.TREE.value] > jnp.float32(0.5)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)
    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    big = jnp.float32(1e9)
    tree_key = jnp.where(tree_here, manh, big)
    flat_t = jnp.argmin(tree_key.reshape(-1))
    t_r = flat_t // jnp.int32(OBS_DIM[1])
    t_c = flat_t % jnp.int32(OBS_DIM[1])
    has_tree = jnp.any(tree_here).astype(jnp.float32)
    nearest_manh_crop = manh[t_r, t_c]
    d_row = (t_r.astype(jnp.float32) - pr) * has_tree
    d_col = (t_c.astype(jnp.float32) - pc) * has_tree
    fov_r = jnp.float32(_cfg.CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)
    dist_sq_norm = jnp.where(
        has_tree > 0.5,
        (nearest_manh_crop / jnp.maximum(crop_max_manh, 1.0)) ** 2,
        jnp.float32(1.0),
    )
    blocked = _cardinal_blocked(blocks)
    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    dir_oh = _direction_one_hot(state)
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_tree = tree_here[fr, fc].astype(jnp.float32)
    adj_tree = jnp.any((manh == 1.0) & tree_here).astype(jnp.float32)
    wood_norm = jnp.clip(
        state.inventory.wood.astype(jnp.float32) / jnp.float32(_cfg.MAX_RESOURCES),
        0.0,
        1.0,
    )
    return jnp.stack(
        [
            dx_norm,
            dy_norm,
            has_tree,
            blocked[0],
            blocked[1],
            blocked[2],
            blocked[3],
            adj_tree,
            face_tree,
            dist_sq_norm,
            wood_norm,
        ],
        axis=0,
    )


def collect_stone_features_from_symbolic_obs(symbolic_obs: jnp.ndarray, state) -> jnp.ndarray:
    return _target_gather_features(
        symbolic_obs,
        state,
        jnp.int32(BlockType.STONE.value),
        state.inventory.stone,
        float(_cfg.COLLECT_STONE_MAX_STONE),
        float(_cfg.COLLECT_STONE_FOV_CHEBYSHEV_R),
    )


def _stone_blocker_features_from_map(
    blocks, pr_i, pc_i, pr, pc, t_r, t_c, has_target, direction
) -> jnp.ndarray:
    stone_here = blocks[..., BlockType.STONE.value] > jnp.float32(0.5)
    big = jnp.float32(1e9)
    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    stone_key = jnp.where(stone_here, manh, big)
    flat_s = jnp.argmin(stone_key.reshape(-1))
    s_r = flat_s // jnp.int32(OBS_DIM[1])
    s_c = flat_s % jnp.int32(OBS_DIM[1])
    has_stone = jnp.any(stone_here).astype(jnp.float32)
    block_dx = jnp.clip((s_c.astype(jnp.float32) - pc) / jnp.float32(4.0), -1.0, 1.0) * has_stone
    block_dy = jnp.clip((s_r.astype(jnp.float32) - pr) / jnp.float32(4.0), -1.0, 1.0) * has_stone
    adj_stone = jnp.any((manh == 1.0) & stone_here).astype(jnp.float32)
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_stone = stone_here[fr, fc].astype(jnp.float32)
    return jnp.stack([block_dx, block_dy, adj_stone, face_stone], axis=0)


def collect_ore_features_from_symbolic_obs(
    symbolic_obs: jnp.ndarray, state, target_id: jnp.ndarray
) -> jnp.ndarray:
    targets = jnp.asarray(
        [BlockType.COAL.value, BlockType.IRON.value, BlockType.DIAMOND.value],
        dtype=jnp.int32,
    )
    inv_map = jnp.stack(
        [state.inventory.coal, state.inventory.iron, state.inventory.diamond],
        axis=0,
    )
    cap_map = jnp.asarray(
        [
            _cfg.COLLECT_ORE_MAX_COAL,
            _cfg.COLLECT_ORE_MAX_IRON,
            _cfg.COLLECT_ORE_MAX_DIAMOND,
        ],
        dtype=jnp.float32,
    )
    tid = target_id.astype(jnp.int32)
    base = _target_gather_features(
        symbolic_obs,
        state,
        targets[tid],
        inv_map[tid],
        cap_map[tid].astype(jnp.float32),
        jnp.float32(_cfg.COLLECT_ORE_FOV_NORM_CHEBYSHEV_R),
    )
    blocks = _blocks_from_symbolic(symbolic_obs)
    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)
    target_here = blocks[..., targets[tid]] > jnp.float32(0.5)
    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    big = jnp.float32(1e9)
    target_key = jnp.where(target_here, manh, big)
    flat_t = jnp.argmin(target_key.reshape(-1))
    t_r = flat_t // jnp.int32(OBS_DIM[1])
    t_c = flat_t % jnp.int32(OBS_DIM[1])
    has_target = jnp.any(target_here).astype(jnp.float32)
    dir_oh = _direction_one_hot(state)
    direction = jnp.argmax(dir_oh) + 1
    blockers = _stone_blocker_features_from_map(
        blocks, pr_i, pc_i, pr, pc, t_r, t_c, has_target, direction
    )
    return jnp.concatenate([base, blockers], axis=0)


def monster_fight_features_from_symbolic_obs(symbolic_obs: jnp.ndarray, state) -> jnp.ndarray:
    blocks = _blocks_from_symbolic(symbolic_obs)
    mob = _mob_layers_from_symbolic(symbolic_obs)
    melee = mob[..., _MELEE_MOB_REL].sum(axis=-1)
    projectile = mob[..., _PROJECTILE_MOB_REL].sum(axis=-1)
    hostile = melee > jnp.float32(0.5)
    arr_here = projectile > jnp.float32(0.5)

    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)
    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)

    big = jnp.float32(1e9)
    host_key = jnp.where(hostile, manh, big)
    flat_h = jnp.argmin(host_key.reshape(-1))
    h_r = flat_h // jnp.int32(OBS_DIM[1])
    h_c = flat_h % jnp.int32(OBS_DIM[1])
    has_hostile = jnp.any(hostile).astype(jnp.float32)
    nearest_dist_h = manh[h_r, h_c]
    d_row = (h_r.astype(jnp.float32) - pr) * has_hostile
    d_col = (h_c.astype(jnp.float32) - pc) * has_hostile
    fov_r = jnp.float32(_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    inv_host = (1.0 / (nearest_dist_h + 1.0)) * has_hostile

    blocked = _cardinal_blocked(blocks)
    adj_hostile = jnp.any((manh == 1.0) & hostile).astype(jnp.float32)
    dir_oh = _direction_one_hot(state)
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_host = hostile[fr, fc].astype(jnp.float32)
    arrow_on_face = arr_here[fr, fc].astype(jnp.float32)
    health_norm = jnp.clip(
        state.player_health.astype(jnp.float32) / jnp.float32(_cfg.MONSTER_FIGHT_PLAYER_MAX_HEALTH),
        0.0,
        1.0,
    )

    has_arrow = jnp.any(arr_here).astype(jnp.float32)
    arr_key = jnp.where(arr_here, manh, big)
    flat_a = jnp.argmin(arr_key.reshape(-1))
    a_r = flat_a // jnp.int32(OBS_DIM[1])
    a_c = flat_a % jnp.int32(OBS_DIM[1])
    arr_dist = manh[a_r, a_c]
    dx_a = jnp.clip((a_r.astype(jnp.float32) - pr) / fov_r, -1.0, 1.0) * has_arrow
    dy_a = jnp.clip((a_c.astype(jnp.float32) - pc) / fov_r, -1.0, 1.0) * has_arrow
    inv_a = (1.0 / (arr_dist + 1.0)) * has_arrow
    n_arr_slots = jnp.float32(3.0)
    num_norm = jnp.minimum(arr_here.sum().astype(jnp.float32), n_arr_slots) / n_arr_slots

    return jnp.concatenate(
        [
            jnp.stack(
                [
                    dx_norm,
                    dy_norm,
                    inv_host,
                    has_hostile,
                    blocked[0],
                    blocked[1],
                    blocked[2],
                    blocked[3],
                    adj_hostile,
                    face_host,
                    health_norm,
                    jnp.float32(0.0),
                ],
                axis=0,
            ),
            jnp.stack(
                [has_arrow, dx_a, dy_a, inv_a, num_norm, arrow_on_face],
                axis=0,
            ),
        ],
        axis=0,
    )


def find_food_features_from_symbolic_obs(symbolic_obs: jnp.ndarray, state) -> jnp.ndarray:
    """10-D find-food features aligned with Classic (slot 9 = dist_sq_norm)."""
    mob = _mob_layers_from_symbolic(symbolic_obs)
    passive = mob[..., _PASSIVE_MOB_REL].sum(axis=-1)
    cow_here = passive > jnp.float32(0.5)
    blocks = _blocks_from_symbolic(symbolic_obs)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)
    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)
    big = jnp.float32(1e9)
    cow_key = jnp.where(cow_here, manh, big)
    flat_c = jnp.argmin(cow_key.reshape(-1))
    c_r = flat_c // jnp.int32(OBS_DIM[1])
    c_c = flat_c % jnp.int32(OBS_DIM[1])
    has_cow = jnp.any(cow_here).astype(jnp.float32)
    nearest_manh = manh[c_r, c_c]
    d_row = (c_r.astype(jnp.float32) - pr) * has_cow
    d_col = (c_c.astype(jnp.float32) - pc) * has_cow
    fov_r = jnp.float32(_cfg.CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    dist_sq_norm = jnp.where(
        has_cow > 0.5,
        (nearest_manh / jnp.maximum(crop_max_manh, 1.0)) ** 2,
        jnp.float32(1.0),
    )
    blocked = _cardinal_blocked(blocks)
    adj_cow = jnp.any((manh == 1.0) & cow_here).astype(jnp.float32)
    dir_oh = _direction_one_hot(state)
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_cow = cow_here[fr, fc].astype(jnp.float32)
    return jnp.stack(
        [
            dx_norm,
            dy_norm,
            has_cow,
            blocked[0],
            blocked[1],
            blocked[2],
            blocked[3],
            adj_cow,
            face_cow,
            dist_sq_norm,
        ],
        axis=0,
    )


def make_find_wood_obs(env, state) -> jnp.ndarray:
    return find_wood_features_from_symbolic_obs(env.get_obs(state), state)


def make_drink_water_obs(env, state) -> jnp.ndarray:
    sym = env.get_obs(state)
    blocks = _blocks_from_symbolic(sym)
    water_here = blocks[..., BlockType.WATER.value] > jnp.float32(0.5)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)
    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    big = jnp.float32(1e9)
    water_key = jnp.where(water_here, manh, big)
    flat_t = jnp.argmin(water_key.reshape(-1))
    t_r = flat_t // jnp.int32(OBS_DIM[1])
    t_c = flat_t % jnp.int32(OBS_DIM[1])
    has_water = jnp.any(water_here).astype(jnp.float32)
    nearest_manh_crop = manh[t_r, t_c]
    d_row = (t_r.astype(jnp.float32) - pr) * has_water
    d_col = (t_c.astype(jnp.float32) - pc) * has_water
    fov_r = jnp.float32(_cfg.CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)
    dist_sq_norm = jnp.where(
        has_water > 0.5,
        (nearest_manh_crop / jnp.maximum(crop_max_manh, 1.0)) ** 2,
        jnp.float32(1.0),
    )
    blocked = _cardinal_blocked(blocks)
    pr_i = jnp.int32(_SYMBOLIC_PLAYER_ROW)
    pc_i = jnp.int32(_SYMBOLIC_PLAYER_COL)
    dir_oh = _direction_one_hot(state)
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_water = water_here[fr, fc].astype(jnp.float32)
    adj_water = jnp.any((manh == 1.0) & water_here).astype(jnp.float32)
    drink_norm = jnp.clip(
        state.player_drink.astype(jnp.float32) / jnp.float32(_cfg.MAX_RESOURCES),
        0.0,
        1.0,
    )
    return jnp.stack(
        [
            dx_norm,
            dy_norm,
            has_water,
            blocked[0],
            blocked[1],
            blocked[2],
            blocked[3],
            adj_water,
            face_water,
            dist_sq_norm,
            drink_norm,
        ],
        axis=0,
    )


def make_find_food_obs(env, state) -> jnp.ndarray:
    return find_food_features_from_symbolic_obs(env.get_obs(state), state)


def make_monster_fight_obs(env, state) -> jnp.ndarray:
    return monster_fight_features_from_symbolic_obs(env.get_obs(state), state)


def make_collect_stone_obs(env, state) -> jnp.ndarray:
    return collect_stone_features_from_symbolic_obs(env.get_obs(state), state)


def make_collect_ore_obs(env, state, target_id: jnp.ndarray) -> jnp.ndarray:
    return collect_ore_features_from_symbolic_obs(env.get_obs(state), state, target_id)


def collect_ore_has_target(obs: jnp.ndarray) -> jnp.ndarray:
    """Target ore visible in skill obs (dim 2 = has_target)."""
    return (obs[..., 2] >= jnp.float32(0.5)).astype(jnp.float32)
