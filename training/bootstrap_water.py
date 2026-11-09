"""Water-task bootstrap: remove all natural WATER, then paint a single training patch.

Call after ``env.reset``: every ``BlockType.WATER`` cell is replaced with ``GRASS`` so
no reset-seed water remains, then a square of WATER is written at a sampled offset.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType

import configs as _cfg
from configs import DRINK_WATER_TARGET_FEATURES, OBS_SIZE
from training.bootstrap_target import SYMBOLIC_FOV_PC, SYMBOLIC_FOV_PR
from training.monster_world_bootstrap import rebuild_mob_map_from_state

_NUM_BLOCK_TYPES = int(len(BlockType))
_NUM_MOB_CHANNELS = 4
_SYMBOLIC_CHANNELS_PER_CELL = _NUM_BLOCK_TYPES + _NUM_MOB_CHANNELS
_SYMBOLIC_MAP_FLAT = int(OBS_DIM[0] * OBS_DIM[1] * _SYMBOLIC_CHANNELS_PER_CELL)
_SYMBOLIC_TAIL = OBS_SIZE - _SYMBOLIC_MAP_FLAT
assert _SYMBOLIC_TAIL == 22, (_SYMBOLIC_MAP_FLAT, OBS_SIZE)

_SYMBOLIC_DRINK_IDX = _SYMBOLIC_MAP_FLAT + 14
_SYMBOLIC_DIR_START = _SYMBOLIC_MAP_FLAT + 12 + 4

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

_WALKABLE_BLOCKS: tuple[int, ...] = (
    BlockType.GRASS.value,
    BlockType.PATH.value,
    BlockType.SAND.value,
)


def _allowed_block_mask(state_map: jnp.ndarray, allowed_blocks: tuple[int, ...]) -> jnp.ndarray:
    """``state_map in allowed_blocks`` without ``jnp.isin`` (portable)."""
    allowed = jnp.zeros_like(state_map, dtype=bool)
    for b in allowed_blocks:
        allowed = allowed | (state_map == b)
    return allowed


def _symbolic_walkable_dot_vec() -> jnp.ndarray:
    """Grass / path / sand count as walkable (matches `_WALKABLE_BLOCKS`)."""
    v = jnp.zeros((_NUM_BLOCK_TYPES,), dtype=jnp.float32)
    v = v.at[BlockType.GRASS.value].set(1.0)
    v = v.at[BlockType.PATH.value].set(1.0)
    v = v.at[BlockType.SAND.value].set(1.0)
    return v


def drink_water_features_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """11-D handcrafted features from flat Craftax-Classic symbolic obs."""
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    water_here = blocks[..., BlockType.WATER.value] > jnp.float32(0.5)

    pr_i = jnp.asarray(OBS_DIM[0] // 2, dtype=jnp.int32)
    pc_i = jnp.asarray(OBS_DIM[1] // 2, dtype=jnp.int32)
    pr = jnp.float32(OBS_DIM[0] // 2)
    pc = jnp.float32(OBS_DIM[1] // 2)

    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)

    big = jnp.float32(1e9)
    water_key = jnp.where(water_here, manh, big)
    flat_w = jnp.argmin(water_key.reshape(-1))
    w_r = flat_w // jnp.int32(OBS_DIM[1])
    w_c = flat_w % jnp.int32(OBS_DIM[1])
    has_water = jnp.any(water_here).astype(jnp.float32)
    nearest_manh_crop = manh[w_r, w_c]
    d_row = (w_r.astype(jnp.float32) - pr) * has_water
    d_col = (w_c.astype(jnp.float32) - pc) * has_water
    fov_r = jnp.float32(_cfg.DRINK_WATER_FOV_NORM_CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    dist_sq_norm = jnp.where(
        has_water > 0.5,
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

    adj_water = jnp.any((manh == 1.0) & water_here).astype(jnp.float32)

    dir_oh = sym[_SYMBOLIC_DIR_START : _SYMBOLIC_DIR_START + 4]
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_water = water_here[fr, fc].astype(jnp.float32)

    drink_ch = sym[_SYMBOLIC_DRINK_IDX]
    drink_norm = jnp.clip(drink_ch * jnp.float32(10.0) / jnp.float32(9.0), 0.0, 1.0)

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


_EXPECTED_FEATS = 11
assert DRINK_WATER_TARGET_FEATURES == _EXPECTED_FEATS, (
    f"config.DRINK_WATER_TARGET_FEATURES={DRINK_WATER_TARGET_FEATURES} vs {_EXPECTED_FEATS}"
)


def make_drink_water_obs(env, state) -> jnp.ndarray:
    return drink_water_features_from_symbolic_obs(env.get_obs(state))


def strip_all_water_from_map(mmap: jnp.ndarray) -> jnp.ndarray:
    """Replace every WATER tile with GRASS so the map has no water before our patch."""
    w = jnp.asarray(BlockType.WATER.value, dtype=mmap.dtype)
    g = jnp.asarray(BlockType.GRASS.value, dtype=mmap.dtype)
    return jnp.where(mmap == w, g, mmap)


def _place_water_patch(
    mmap: jnp.ndarray, player_position: jnp.ndarray, patch_rng: jax.Array
) -> jnp.ndarray:
    """Overwrite a small WATER square; center sampled in a Chebyshev annulus around the player."""
    side = int(_cfg.DRINK_WATER_WATER_PATCH_SIDE)
    if side % 2 != 1 or side < 1:
        raise ValueError(f"DRINK_WATER_WATER_PATCH_SIDE must be a positive odd int, got {side}")
    half_w = side // 2
    max_r_scan = max(int(_cfg.DRINK_WATER_SPAWN_MAX_CHEBYSHEV_R), half_w)

    rr = jnp.arange(-max_r_scan, max_r_scan + 1, dtype=jnp.int32)
    cc = jnp.arange(-max_r_scan, max_r_scan + 1, dtype=jnp.int32)
    grid_r, grid_c = jnp.meshgrid(rr, cc, indexing="ij")
    dr = grid_r.reshape(-1)
    dc = grid_c.reshape(-1)
    cheb = jnp.maximum(jnp.abs(dr), jnp.abs(dc))

    lo = jnp.int32(_cfg.DRINK_WATER_SPAWN_MIN_CHEBYSHEV_R)
    hi = jnp.int32(_cfg.DRINK_WATER_SPAWN_MAX_CHEBYSHEV_R)
    in_symbolic = (jnp.abs(dr) <= jnp.int32(SYMBOLIC_FOV_PR)) & (
        jnp.abs(dc) <= jnp.int32(SYMBOLIC_FOV_PC)
    )
    valid = jnp.logical_and(cheb >= lo, cheb <= hi) & in_symbolic
    fallback = in_symbolic & (cheb >= jnp.int32(1))
    valid = jnp.where(jnp.any(valid), valid, fallback)
    noise = jax.random.uniform(patch_rng, shape=valid.shape)
    pick_i = jnp.argmax(noise + valid.astype(jnp.float32) * jnp.float32(1.0))

    pp = player_position.astype(jnp.int32)
    cr = jnp.clip(
        pp[0] + dr[pick_i],
        jnp.int32(1),
        jnp.int32(mmap.shape[0] - 2),
    )
    cl = jnp.clip(
        pp[1] + dc[pick_i],
        jnp.int32(1),
        jnp.int32(mmap.shape[1] - 2),
    )

    ij = jnp.arange(-half_w, half_w + 1, dtype=jnp.int32)
    ddi, ddj = jnp.meshgrid(ij, ij, indexing="ij")
    tr = jnp.clip(cr + ddi, jnp.int32(1), jnp.int32(mmap.shape[0] - 2)).ravel()
    tc = jnp.clip(cl + ddj, jnp.int32(1), jnp.int32(mmap.shape[1] - 2)).ravel()
    wt = jnp.asarray(BlockType.WATER.value, dtype=mmap.dtype)
    return mmap.at[tr, tc].set(wt)


def bootstrap_water_state(state, boot_rng: jax.Array, _params) -> object:
    """Clear hostiles; strip all WATER from the map; write one WATER patch; reset thirst baseline."""
    patch_rng = boot_rng

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

    staged = state.replace(zombies=zombies, skeletons=skeletons, cows=cows, arrows=arrows)
    cleared = strip_all_water_from_map(staged.map)
    new_map = _place_water_patch(cleared, staged.player_position.astype(jnp.int32), patch_rng)
    staged_map = staged.replace(map=new_map)
    mob_map = rebuild_mob_map_from_state(staged_map)

    thirst0 = jnp.asarray(
        float(_cfg.DRINK_WATER_PLAYER_START_THIRST), dtype=staged_map.player_thirst.dtype
    )
    drink0 = jnp.asarray(
        int(_cfg.DRINK_WATER_PLAYER_START_DRINK), dtype=staged_map.player_drink.dtype
    )

    return staged_map.replace(
        mob_map=mob_map,
        player_thirst=thirst0,
        player_drink=drink0,
        is_sleeping=jnp.asarray(False, dtype=staged_map.is_sleeping.dtype),
    )
