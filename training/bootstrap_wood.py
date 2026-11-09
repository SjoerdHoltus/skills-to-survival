"""Find-wood bootstrap and 11-D features from Craftax-Classic symbolic obs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType

import configs as _cfg
from configs import CHEBYSHEV_R, OBS_SIZE
from training.monster_world_bootstrap import (
    _sample_tree_positions_in_fov,
    rebuild_mob_map_from_state,
)

_NUM_BLOCK_TYPES = int(len(BlockType))
_NUM_MOB_CHANNELS = 4
_SYMBOLIC_CHANNELS_PER_CELL = _NUM_BLOCK_TYPES + _NUM_MOB_CHANNELS
_SYMBOLIC_MAP_FLAT = int(OBS_DIM[0] * OBS_DIM[1] * _SYMBOLIC_CHANNELS_PER_CELL)
_SYMBOLIC_TAIL = OBS_SIZE - _SYMBOLIC_MAP_FLAT
assert _SYMBOLIC_TAIL == 22, (_SYMBOLIC_MAP_FLAT, OBS_SIZE)

_SYMBOLIC_WOOD_IDX = _SYMBOLIC_MAP_FLAT + 0
_SYMBOLIC_HEALTH_IDX = _SYMBOLIC_MAP_FLAT + 12
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


def _symbolic_walkable_dot_vec() -> jnp.ndarray:
    v = jnp.zeros((_NUM_BLOCK_TYPES,), dtype=jnp.float32)
    v = v.at[BlockType.GRASS.value].set(1.0)
    v = v.at[BlockType.PATH.value].set(1.0)
    v = v.at[BlockType.SAND.value].set(1.0)
    return v


def find_wood_features_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """11-D handcrafted features (parallel layout to drink-water)."""
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    tree_here = blocks[..., BlockType.TREE.value] > jnp.float32(0.5)

    pr_i = jnp.asarray(OBS_DIM[0] // 2, dtype=jnp.int32)
    pc_i = jnp.asarray(OBS_DIM[1] // 2, dtype=jnp.int32)
    pr = jnp.float32(OBS_DIM[0] // 2)
    pc = jnp.float32(OBS_DIM[1] // 2)

    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)

    big = jnp.float32(1e9)
    tree_key = jnp.where(tree_here, manh, big)
    flat_t = jnp.argmin(tree_key.reshape(-1))
    t_r = flat_t // jnp.int32(OBS_DIM[1])
    t_c = flat_t % jnp.int32(OBS_DIM[1])
    has_tree = jnp.any(tree_here).astype(jnp.float32)
    nearest_manh_crop = manh[t_r, t_c]
    d_row = (t_r.astype(jnp.float32) - pr) * has_tree
    d_col = (t_c.astype(jnp.float32) - pc) * has_tree
    fov_r = jnp.float32(CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    dist_sq_norm = jnp.where(
        has_tree > 0.5,
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

    adj_tree = jnp.any((manh == 1.0) & tree_here).astype(jnp.float32)

    dir_oh = sym[_SYMBOLIC_DIR_START : _SYMBOLIC_DIR_START + 4]
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_tree = tree_here[fr, fc].astype(jnp.float32)

    wood_ch = sym[_SYMBOLIC_WOOD_IDX]
    wood_norm = jnp.clip(wood_ch * jnp.float32(10.0) / jnp.float32(9.0), 0.0, 1.0)

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


def make_find_wood_obs(env, state) -> jnp.ndarray:
    return find_wood_features_from_symbolic_obs(env.get_obs(state))


def bootstrap_wood_state(state, boot_rng: jax.Array, _params) -> object:
    """Clear hostiles, zero wood, add trees in the FOV annulus."""
    boot_rng_tree_count, boot_rng_trees = jax.random.split(boot_rng)

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
    player = staged.player_position.astype(jnp.int32)

    tree_count = jax.random.randint(
        boot_rng_tree_count,
        (),
        minval=jnp.int32(_cfg.FIND_WOOD_NUM_TREES_MIN),
        maxval=jnp.int32(_cfg.FIND_WOOD_NUM_TREES_MAX + 1),
    )

    mob_positions = jnp.zeros((1, 2), dtype=jnp.int32)
    mob_valids = jnp.zeros((1,), dtype=bool)
    new_map = _sample_tree_positions_in_fov(
        staged.map,
        player,
        mob_positions,
        mob_valids,
        boot_rng_trees,
        tree_count,
        fov_r=int(CHEBYSHEV_R),
    )
    staged_map = staged.replace(map=new_map)
    mob_map = rebuild_mob_map_from_state(staged_map)

    wood0 = jnp.asarray(jnp.int32(0), dtype=staged_map.inventory.wood.dtype)
    inventory = staged_map.inventory.replace(wood=wood0)

    return staged_map.replace(mob_map=mob_map, inventory=inventory)
