"""Shared target-block features and FOV map helpers for stone / ore bootstraps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType

from configs import OBS_SIZE

_NUM_BLOCK_TYPES = int(len(BlockType))
_NUM_MOB_CHANNELS = 4
_SYMBOLIC_CHANNELS_PER_CELL = _NUM_BLOCK_TYPES + _NUM_MOB_CHANNELS
_SYMBOLIC_MAP_FLAT = int(OBS_DIM[0] * OBS_DIM[1] * _SYMBOLIC_CHANNELS_PER_CELL)
_SYMBOLIC_TAIL = OBS_SIZE - _SYMBOLIC_MAP_FLAT
assert _SYMBOLIC_TAIL == 22, (_SYMBOLIC_MAP_FLAT, OBS_SIZE)

_SYMBOLIC_INVENTORY_START = _SYMBOLIC_MAP_FLAT
_SYMBOLIC_HEALTH_IDX = _SYMBOLIC_MAP_FLAT + 12
_SYMBOLIC_DIR_START = _SYMBOLIC_MAP_FLAT + 12 + 4

# Inventory tail order matches Craftax symbolic layout: wood, stone, coal, iron, ...
INV_IDX_STONE = jnp.int32(1)
INV_IDX_COAL = jnp.int32(2)
INV_IDX_IRON = jnp.int32(3)
INV_IDX_DIAMOND = jnp.int32(4)

_ORE_BLOCK_VALUES = jnp.asarray(
    [
        BlockType.STONE.value,
        BlockType.COAL.value,
        BlockType.IRON.value,
        BlockType.DIAMOND.value,
    ],
    dtype=jnp.int32,
)

# Gather-style primitive layout: dx, dy, has, blocked[4], adj, face, dist_sq, inv_norm.
GATHER_OBS_BLOCKED_START = 3
GATHER_OBS_BLOCKED_LEN = 4

_CARDINAL_DELTAS = jnp.asarray(
    [
        [0, 0],
        [0, -1],
        [0, +1],
        [-1, 0],
        [+1, 0],
    ],
    dtype=jnp.int32,
)


def _symbolic_walkable_dot_vec() -> jnp.ndarray:
    v = jnp.zeros((_NUM_BLOCK_TYPES,), dtype=jnp.float32)
    v = v.at[BlockType.GRASS.value].set(1.0)
    v = v.at[BlockType.PATH.value].set(1.0)
    v = v.at[BlockType.SAND.value].set(1.0)
    return v


def cardinal_blocked_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """4-D walkability blockers for the player's cardinal neighbors (shared across skills)."""
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    pr_i = jnp.asarray(OBS_DIM[0] // 2, dtype=jnp.int32)
    pc_i = jnp.asarray(OBS_DIM[1] // 2, dtype=jnp.int32)
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


def gather_obs_meta_slice(gather_obs: jnp.ndarray) -> jnp.ndarray:
    """Drop shared ``blocked[4]`` (indices 3..6) from an 11-D (or 10-D find-food) gather vector."""
    end = GATHER_OBS_BLOCKED_START + GATHER_OBS_BLOCKED_LEN
    return jnp.concatenate([gather_obs[:GATHER_OBS_BLOCKED_START], gather_obs[end:]], axis=0)


def ore_obs_meta_slice(ore_obs: jnp.ndarray) -> jnp.ndarray:
    """15-D collect-ore obs without duplicating gather ``blocked[4]`` in the meta stack."""
    base = gather_obs_meta_slice(ore_obs[:11])
    return jnp.concatenate([base, ore_obs[11:]], axis=0)


def collect_target_features_from_symbolic_obs(
    symbolic_obs: jnp.ndarray,
    target_block_value: jnp.ndarray,
    inv_channel_idx: jnp.ndarray,
    inv_norm_scale: jnp.ndarray,
    fov_norm_chebyshev_r: float,
    *,
    include_health: bool = False,
    player_max_health: float = 9.0,
) -> jnp.ndarray:
    """Handcrafted target features (11-D default; 12-D with ``include_health=True``)."""
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    target_here = blocks[..., target_block_value.astype(jnp.int32)] > jnp.float32(0.5)

    pr_i = jnp.asarray(OBS_DIM[0] // 2, dtype=jnp.int32)
    pc_i = jnp.asarray(OBS_DIM[1] // 2, dtype=jnp.int32)
    pr = jnp.float32(OBS_DIM[0] // 2)
    pc = jnp.float32(OBS_DIM[1] // 2)

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

    wv = _symbolic_walkable_dot_vec()

    def _blocked(dr: int, dc: int) -> jnp.ndarray:
        r = pr_i + jnp.int32(dr)
        c = pc_i + jnp.int32(dc)
        oh = blocks[r, c, :]
        return (jnp.dot(oh, wv) < 0.5).astype(jnp.float32)

    blocked = jnp.stack(
        [_blocked(0, -1), _blocked(0, 1), _blocked(-1, 0), _blocked(1, 0)],
        axis=0,
    )

    adj_target = jnp.any((manh == 1.0) & target_here).astype(jnp.float32)

    dir_oh = sym[_SYMBOLIC_DIR_START : _SYMBOLIC_DIR_START + 4]
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_target = target_here[fr, fc].astype(jnp.float32)

    inv_ch = sym[_SYMBOLIC_INVENTORY_START + inv_channel_idx.astype(jnp.int32)]
    scale = jnp.maximum(inv_norm_scale.astype(jnp.float32), jnp.float32(1.0))
    inv_norm = jnp.clip(inv_ch * jnp.float32(10.0) / scale, 0.0, 1.0)

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
        health_scaled = sym[_SYMBOLIC_HEALTH_IDX]
        health_norm = jnp.clip(
            health_scaled * jnp.float32(10.0) / jnp.float32(player_max_health),
            0.0,
            1.0,
        )
        feats.append(health_norm)
    return jnp.stack(feats, axis=0)


# Player sits at the center of the 7×9 symbolic crop (matches Craftax renderer).
SYMBOLIC_FOV_PR = int(OBS_DIM[0] // 2)
SYMBOLIC_FOV_PC = int(OBS_DIM[1] // 2)


def symbolic_fov_world_mask(
    map_shape: tuple[int, int] | list[int],
    player: jnp.ndarray,
) -> jnp.ndarray:
    """Boolean (H, W) mask of world tiles visible in the policy's symbolic crop."""
    h, w = int(map_shape[0]), int(map_shape[1])
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    pp = player.astype(jnp.int32)
    in_rows = jnp.abs(rr - pp[0]) <= jnp.int32(SYMBOLIC_FOV_PR)
    in_cols = jnp.abs(cc - pp[1]) <= jnp.int32(SYMBOLIC_FOV_PC)
    return in_rows & in_cols


def _sample_block_positions_in_fov(
    state_map: jnp.ndarray,
    player: jnp.ndarray,
    mob_positions: jnp.ndarray,
    mob_valids: jnp.ndarray,
    rng: jax.Array,
    num_blocks: jnp.ndarray,
    block_value: jnp.ndarray,
    fov_r: int,
) -> jnp.ndarray:
    """Place up to ``num_blocks`` cells of ``block_value`` in the player FOV annulus."""
    nb = jnp.asarray(num_blocks, dtype=jnp.int32).reshape(())
    h, w = state_map.shape
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]

    cheb_player = jnp.maximum(jnp.abs(rr - player[0]), jnp.abs(cc - player[1]))
    in_symbolic = symbolic_fov_world_mask((h, w), player)
    in_fov = cheb_player <= jnp.asarray(fov_r, dtype=jnp.int32)
    not_player_neighbourhood = cheb_player > 1

    def per_mob_mask(m_pos, m_valid):
        cheb_m = jnp.maximum(jnp.abs(rr - m_pos[0]), jnp.abs(cc - m_pos[1]))
        return jnp.logical_or(jnp.logical_not(m_valid), cheb_m > 1)

    per_mob = jax.vmap(per_mob_mask)(mob_positions, mob_valids)
    not_mob_neighbourhood = jnp.all(per_mob, axis=0)

    candidate = in_symbolic & in_fov & not_player_neighbourhood & not_mob_neighbourhood

    scores = jax.random.uniform(rng, (h, w), dtype=jnp.float32)
    scores = jnp.where(candidate, scores, jnp.float32(1e9))

    flat_scores = scores.reshape(-1)
    sorted_scores = jnp.sort(flat_scores)
    n_cells = jnp.int32(flat_scores.shape[0])
    nb_eff = jnp.clip(nb, jnp.int32(0), n_cells)
    threshold = jnp.where(
        nb_eff <= 0,
        jnp.float32(-jnp.inf),
        sorted_scores[nb_eff - 1],
    )
    is_block = (scores <= threshold) & candidate

    return jnp.where(
        is_block,
        block_value.astype(state_map.dtype),
        state_map,
    )


def _strip_natural_ores(state_map: jnp.ndarray) -> jnp.ndarray:
    g = jnp.asarray(BlockType.GRASS.value, dtype=state_map.dtype)
    out = state_map
    for bv in _ORE_BLOCK_VALUES:
        out = jnp.where(out == bv, g, out)
    return out
