"""
Monster-world bootstrapping for Craftax Classic.

Bootstrap and spawning logic use the full ``EnvState`` (positions, masks, map,
etc.) as needed—that is **not** cheating because the **policy** never receives
that state: `make_monster_fight_obs` builds its input **only** from
``env.get_obs(state)`` (the 1345-dim symbolic observation).

For monster_fight training/eval we intentionally use a very simple reset:
- clear all mobs and arrows
- equip the player with a wooden sword
- spawn mobs at random valid tiles inside the **symbolic 7×9 crop** (not merely
  Chebyshev‑4 on the world map, which extends one row past the crop vertically)

This keeps early learning focused and removes multi-mob scenario variance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType

import configs as _mb_cfg
from configs import FIND_FOOD_TARGET_FEATURES, MONSTER_FIGHT_TARGET_FEATURES, OBS_SIZE

# Flat `render_craftax_symbolic` layout: per cell = one-hot(block type, |BlockType|)
# + 4 uint8 mob channels (zombie, cow, skeleton, arrow), then inventory + intrinsics + …
_NUM_BLOCK_TYPES = int(len(BlockType))
_NUM_MOB_CHANNELS = 4
_SYMBOLIC_CHANNELS_PER_CELL = _NUM_BLOCK_TYPES + _NUM_MOB_CHANNELS
_SYMBOLIC_MAP_FLAT = int(OBS_DIM[0] * OBS_DIM[1] * _SYMBOLIC_CHANNELS_PER_CELL)
_SYMBOLIC_TAIL = OBS_SIZE - _SYMBOLIC_MAP_FLAT  # 22 = inv(12)+intrinsics(4)+dir(4)+2
assert _SYMBOLIC_TAIL == 22, (_SYMBOLIC_MAP_FLAT, OBS_SIZE)
# Player sits at the center of the 7×9 crop (same as renderer local grid).
_SYMBOLIC_PLAYER_ROW = OBS_DIM[0] // 2
_SYMBOLIC_PLAYER_COL = OBS_DIM[1] // 2
# Intrinsics start after map (1323) + inventory (12); first slot is player_health/10.
_SYMBOLIC_HEALTH_IDX = _SYMBOLIC_MAP_FLAT + 12
_SYMBOLIC_DIR_START = _SYMBOLIC_MAP_FLAT + 12 + 4

from training.bootstrap_target import symbolic_fov_world_mask

_WALKABLE_BLOCKS: tuple[int, ...] = (
    BlockType.GRASS.value,
    BlockType.PATH.value,
    BlockType.SAND.value,
)


def _equip_wooden_sword(inventory):
    """Set wooden sword in inventory when the field exists."""
    if hasattr(inventory, "wood_sword"):
        wood_sword = jnp.asarray(1, dtype=inventory.wood_sword.dtype)
        return inventory.replace(wood_sword=wood_sword)
    # Fallback used by some Craftax inventory variants where sword is tier-coded.
    if hasattr(inventory, "sword"):
        sword = jnp.asarray(1, dtype=inventory.sword.dtype)
        return inventory.replace(sword=sword)
    return inventory


def _allowed_block_mask(state_map: jnp.ndarray, allowed_blocks: tuple[int, ...]) -> jnp.ndarray:
    """
    Returns a boolean mask `state_map in allowed_blocks` with same shape as state_map.
    JAX doesn't have `isin` for integers in all versions, so we OR comparisons.
    """
    allowed = jnp.zeros_like(state_map, dtype=bool)
    for b in allowed_blocks:
        allowed = allowed | (state_map == b)
    return allowed


def _sample_zombies_in_fov(
    state_map: jnp.ndarray,
    player: jnp.ndarray,
    rng: jax.Array,
    num_zombies: int,
    fov_r: int | None = None,
    min_cheb_dist: int | None = None,
    allowed_blocks: tuple[int, ...] = _WALKABLE_BLOCKS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample `num_zombies` distinct zombie spawn tiles in the player's FOV.

    Returns
        positions  — int32 array of shape (num_zombies, 2), rows/cols
        valid      — bool   array of shape (num_zombies,), False if the
                     corresponding slot could not be filled (fewer valid
                     tiles in the FOV than slots requested).

    The selection is "score-and-take-N-smallest": each tile gets a uniform
    [0, 1) score (1e9 if the tile is invalid) and we pick the N smallest.
    This guarantees N distinct positions (no duplicates by construction) and
    is fully JIT-compatible. If the FOV annulus is too cramped, falls back
    to "any valid tile" — same fallback policy as the previous single-zombie
    sampler.

    ``fov_r`` / ``min_cheb_dist`` default from ``config`` at call time (so
    eval can match a run config loaded after import).
    """
    if fov_r is None:
        fov_r = int(_mb_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R)
    if min_cheb_dist is None:
        min_cheb_dist = int(_mb_cfg.MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST)
    h, w = state_map.shape
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    allowed = _allowed_block_mask(state_map, allowed_blocks)
    not_player = jnp.logical_not((rr == player[0]) & (cc == player[1]))
    cheb = jnp.maximum(jnp.abs(rr - player[0]), jnp.abs(cc - player[1]))
    in_symbolic = symbolic_fov_world_mask((h, w), player)
    in_annulus = (cheb <= jnp.asarray(fov_r, dtype=jnp.int32)) & (
        cheb >= jnp.asarray(min_cheb_dist, dtype=jnp.int32)
    )
    valid_annulus = allowed & not_player & in_symbolic & in_annulus
    valid_in_crop = allowed & not_player & in_symbolic
    annulus_count = valid_annulus.sum()
    use_annulus = annulus_count >= jnp.asarray(num_zombies, dtype=annulus_count.dtype)
    valid = jnp.where(use_annulus, valid_annulus, valid_in_crop)

    scores = jax.random.uniform(rng, (h, w), dtype=jnp.float32)
    scores = jnp.where(valid, scores, jnp.float32(1e9))

    flat_scores = scores.reshape(-1)
    flat_indices = jnp.argsort(flat_scores)[: int(num_zombies)]
    rows = flat_indices // jnp.int32(w)
    cols = flat_indices % jnp.int32(w)
    positions = jnp.stack([rows, cols], axis=-1)  # (N, 2)

    chosen_scores = flat_scores[flat_indices]
    valids = chosen_scores < jnp.float32(1e9)
    return positions, valids


def _sample_skeletons_in_fov(
    state_map: jnp.ndarray,
    player: jnp.ndarray,
    zombie_positions: jnp.ndarray,
    zombie_valids: jnp.ndarray,
    rng: jax.Array,
    num_skeletons: int,
    fov_r: int | None = None,
    min_cheb_dist: int | None = None,
    allowed_blocks: tuple[int, ...] = _WALKABLE_BLOCKS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample `num_skeletons` distinct skeleton spawn tiles in the player's FOV
    that do not collide with any active zombie tile.

    Returns (positions, valids) with the same semantics as
    `_sample_zombies_in_fov`. Reuses the same "score-and-take-N-smallest"
    approach: each tile gets a uniform score, invalid tiles get a sentinel
    of 1e9, the N smallest are picked, and `valids[i]` flags whether slot
    i found a valid tile (else the slot stays inactive in the bootstrap).

    ``fov_r`` / ``min_cheb_dist`` default from ``config`` at call time.
    """
    if fov_r is None:
        fov_r = int(_mb_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R)
    if min_cheb_dist is None:
        min_cheb_dist = int(_mb_cfg.MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST)
    h, w = state_map.shape
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    allowed = _allowed_block_mask(state_map, allowed_blocks)
    not_player = jnp.logical_not((rr == player[0]) & (cc == player[1]))
    cheb = jnp.maximum(jnp.abs(rr - player[0]), jnp.abs(cc - player[1]))
    in_symbolic = symbolic_fov_world_mask((h, w), player)
    in_annulus = (cheb <= jnp.asarray(fov_r, dtype=jnp.int32)) & (
        cheb >= jnp.asarray(min_cheb_dist, dtype=jnp.int32)
    )

    # "Not on top of an active zombie": for each (r, c) check no zombie is at
    # exactly that cell. Inactive zombie slots impose no constraint.
    def per_zombie(z_pos, z_valid):
        same = (rr == z_pos[0]) & (cc == z_pos[1])
        return jnp.logical_or(jnp.logical_not(z_valid), jnp.logical_not(same))

    per_zomb_mask = jax.vmap(per_zombie)(zombie_positions, zombie_valids)  # (Nz, h, w)
    not_on_zombie = jnp.all(per_zomb_mask, axis=0)

    valid_annulus = allowed & not_player & in_symbolic & in_annulus & not_on_zombie
    valid_in_crop = allowed & not_player & in_symbolic & not_on_zombie
    annulus_count = valid_annulus.sum()
    use_annulus = annulus_count >= jnp.asarray(num_skeletons, dtype=annulus_count.dtype)
    valid = jnp.where(use_annulus, valid_annulus, valid_in_crop)

    scores = jax.random.uniform(rng, (h, w), dtype=jnp.float32)
    scores = jnp.where(valid, scores, jnp.float32(1e9))
    flat_scores = scores.reshape(-1)
    flat_indices = jnp.argsort(flat_scores)[: int(num_skeletons)]
    rows = flat_indices // jnp.int32(w)
    cols = flat_indices % jnp.int32(w)
    positions = jnp.stack([rows, cols], axis=-1)
    chosen_scores = flat_scores[flat_indices]
    valids = chosen_scores < jnp.float32(1e9)
    return positions, valids


def _sample_tree_positions_in_fov(
    state_map: jnp.ndarray,
    player: jnp.ndarray,
    mob_positions: jnp.ndarray,
    mob_valids: jnp.ndarray,
    rng: jax.Array,
    num_trees: int | jax.Array,
    fov_r: int | None = None,
) -> jnp.ndarray:
    """Return a copy of `state_map` with up to `num_trees` TREE blocks added.

    `mob_positions` has shape `(N, 2)` and `mob_valids` has shape `(N,)` —
    the caller stacks both zombie AND skeleton spawn slots into this array
    so trees never pin any active hostile. Trees are sampled uniformly at
    random from tiles that satisfy ALL of:
      - inside the player's FOV (Chebyshev distance <= `fov_r`)
      - Chebyshev distance > 1 from the player (so all 4 cardinal moves are
        unblocked at reset)
      - Chebyshev distance > 1 from EVERY active mob (so each hostile can
        always be reached from at least one cardinal neighbour, regardless
        of how many are spawned)

    `num_trees` may be a Python int or a scalar JAX array (e.g. per-rollout
    count from ``randint``). If it is 0, no trees are added. If fewer valid
    candidates exist than `num_trees`, the function silently places fewer
    trees — this is fine for training.

    ``fov_r`` defaults from ``config`` at call time.
    """
    if fov_r is None:
        fov_r = int(_mb_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R)

    nt = jnp.asarray(num_trees, dtype=jnp.int32).reshape(())
    h, w = state_map.shape
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]

    cheb_player = jnp.maximum(jnp.abs(rr - player[0]), jnp.abs(cc - player[1]))
    in_symbolic = symbolic_fov_world_mask((h, w), player)
    in_fov = cheb_player <= jnp.asarray(fov_r, dtype=jnp.int32)
    not_player_neighbourhood = cheb_player > 1

    # Per-mob "tile is far enough from this mob" mask, OR'd with "this mob
    # wasn't validly spawned" so failed slots impose no constraint. Then
    # AND across all N to get tiles safe from every mob.
    def per_mob_mask(m_pos, m_valid):
        cheb_m = jnp.maximum(jnp.abs(rr - m_pos[0]), jnp.abs(cc - m_pos[1]))
        return jnp.logical_or(jnp.logical_not(m_valid), cheb_m > 1)

    per_mob = jax.vmap(per_mob_mask)(mob_positions, mob_valids)  # (N, h, w)
    not_mob_neighbourhood = jnp.all(per_mob, axis=0)  # (h, w)

    candidate = in_symbolic & in_fov & not_player_neighbourhood & not_mob_neighbourhood

    scores = jax.random.uniform(rng, (h, w), dtype=jnp.float32)
    scores = jnp.where(candidate, scores, jnp.float32(1e9))

    flat_scores = scores.reshape(-1)
    sorted_scores = jnp.sort(flat_scores)
    n_cells = jnp.int32(flat_scores.shape[0])
    nt_eff = jnp.clip(nt, jnp.int32(0), n_cells)
    threshold = jnp.where(
        nt_eff <= 0,
        jnp.float32(-jnp.inf),
        sorted_scores[nt_eff - 1],
    )
    is_tree = (scores <= threshold) & candidate

    return jnp.where(
        is_tree,
        jnp.asarray(BlockType.TREE.value, dtype=state_map.dtype),
        state_map,
    )


# Cardinal-direction lookup matching `craftax_classic.constants.DIRECTIONS`:
# index 0 = NOOP, 1 = LEFT, 2 = RIGHT, 3 = UP, 4 = DOWN.
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
    """Length-|BlockType| vector: 1 on walkable block one-hot indices (GRASS, PATH)."""
    v = jnp.zeros((_NUM_BLOCK_TYPES,), dtype=jnp.float32)
    v = v.at[BlockType.GRASS.value].set(1.0)
    v = v.at[BlockType.PATH.value].set(1.0)
    return v


def monster_fight_features_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """18-D monster-fight features derived **only** from flat symbolic obs (1345,).

    Layout matches ``render_craftax_symbolic`` in Craftax: a 7××9××(``len(BlockType)``+4)
    map tensor (block one-hot + zombie/cow/skeleton/arrow flags), then inventory,
    intrinsics (health first, each scaled ``/10`` in the obs), direction one-hot, etc.

    Zombies+cows+skeletons+arrows outside the 7×9 crop are invisible (same as the agent).
    Equidistant hostiles: zombie is preferred over skeleton via a +1 tie-break penalty on
    skeleton-only cells in the argmin key.
    """
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    mob = map3[..., _NUM_BLOCK_TYPES:]

    pr_i = jnp.asarray(_SYMBOLIC_PLAYER_ROW, dtype=jnp.int32)
    pc_i = jnp.asarray(_SYMBOLIC_PLAYER_COL, dtype=jnp.int32)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)

    zombie_here = mob[..., 0] > 0.5
    skel_here = mob[..., 2] > 0.5
    ar = mob[..., 3]
    hostile = zombie_here | skel_here

    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)

    skel_only = skel_here & jnp.logical_not(zombie_here)
    big = jnp.float32(1e9)
    host_key = jnp.where(
        hostile,
        manh * 2.0 + jnp.where(skel_only, jnp.float32(1.0), jnp.float32(0.0)),
        big,
    )
    flat_h = jnp.argmin(host_key.reshape(-1))
    h_r = flat_h // jnp.int32(OBS_DIM[1])
    h_c = flat_h % jnp.int32(OBS_DIM[1])
    has_hostile = jnp.any(hostile).astype(jnp.float32)
    nearest_dist_h = manh[h_r, h_c]
    d_row = (h_r.astype(jnp.float32) - pr) * has_hostile
    d_col = (h_c.astype(jnp.float32) - pc) * has_hostile
    fov_r = jnp.float32(_mb_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    inv_host = (1.0 / (nearest_dist_h + 1.0)) * has_hostile
    is_skeleton = (skel_here[h_r, h_c] & jnp.logical_not(zombie_here[h_r, h_c])).astype(
        jnp.float32
    ) * has_hostile

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

    adj_hostile = jnp.any((manh == 1.0) & hostile).astype(jnp.float32)

    dir_oh = sym[_SYMBOLIC_DIR_START : _SYMBOLIC_DIR_START + 4]
    direction = jnp.argmax(dir_oh) + 1
    delta = _CARDINAL_DELTAS[direction]
    fr = pr_i + delta[0]
    fc = pc_i + delta[1]
    face_host = (zombie_here[fr, fc] | skel_here[fr, fc]).astype(jnp.float32)
    arrow_on_face = (ar[fr, fc] > 0.5).astype(jnp.float32)

    health_scaled = sym[_SYMBOLIC_HEALTH_IDX]
    health_norm = jnp.clip(
        health_scaled * jnp.float32(10.0) / jnp.float32(_mb_cfg.MONSTER_FIGHT_PLAYER_MAX_HEALTH),
        0.0,
        1.0,
    )

    arr_here = ar > 0.5
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

    arrow_feats = jnp.stack(
        [
            has_arrow,
            dx_a,
            dy_a,
            inv_a,
            num_norm,
            arrow_on_face,
        ],
        axis=0,
    )

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
                    is_skeleton,
                ],
                axis=0,
            ),
            arrow_feats,
        ],
        axis=0,
    )


MONSTER_OBS_BLOCKED_START = 4
MONSTER_OBS_BLOCKED_LEN = 4


def monster_obs_meta_slice(monster_obs: jnp.ndarray) -> jnp.ndarray:
    """18-D monster-fight obs without duplicating shared cardinal ``blocked[4]``."""
    end = MONSTER_OBS_BLOCKED_START + MONSTER_OBS_BLOCKED_LEN
    return jnp.concatenate(
        [
            monster_obs[:MONSTER_OBS_BLOCKED_START],
            monster_obs[end : MONSTER_OBS_BLOCKED_START + 8],
            monster_obs[MONSTER_OBS_BLOCKED_START + 8 :],
        ],
        axis=0,
    )


def find_food_features_from_symbolic_obs(symbolic_obs: jnp.ndarray) -> jnp.ndarray:
    """10-D find-food features (subset of the 11-D block-gather layout).

    Same navigation core as drink/wood/stone; slot 9 is ``dist_sq_norm`` only (no
    inventory norm — fitness is kill-based, not fill-a-meter).
    """
    sym = symbolic_obs.astype(jnp.float32).reshape(-1)
    map3 = sym[:_SYMBOLIC_MAP_FLAT].reshape(OBS_DIM[0], OBS_DIM[1], _SYMBOLIC_CHANNELS_PER_CELL)
    blocks = map3[..., :_NUM_BLOCK_TYPES]
    mob = map3[..., _NUM_BLOCK_TYPES:]

    pr_i = jnp.asarray(_SYMBOLIC_PLAYER_ROW, dtype=jnp.int32)
    pc_i = jnp.asarray(_SYMBOLIC_PLAYER_COL, dtype=jnp.int32)
    pr = jnp.float32(_SYMBOLIC_PLAYER_ROW)
    pc = jnp.float32(_SYMBOLIC_PLAYER_COL)

    cow_here = mob[..., 1] > 0.5

    rows_f = jnp.arange(OBS_DIM[0], dtype=jnp.float32)[:, None]
    cols_f = jnp.arange(OBS_DIM[1], dtype=jnp.float32)[None, :]
    manh = jnp.abs(rows_f - pr) + jnp.abs(cols_f - pc)
    crop_max_manh = jnp.float32(OBS_DIM[0] + OBS_DIM[1] - 2)

    big = jnp.float32(1e9)
    host_key = jnp.where(cow_here, manh * 2.0, big)
    flat_h = jnp.argmin(host_key.reshape(-1))
    h_r = flat_h // jnp.int32(OBS_DIM[1])
    h_c = flat_h % jnp.int32(OBS_DIM[1])
    has_cow = jnp.any(cow_here).astype(jnp.float32)
    nearest_dist_h = manh[h_r, h_c]
    d_row = (h_r.astype(jnp.float32) - pr) * has_cow
    d_col = (h_c.astype(jnp.float32) - pc) * has_cow
    fov_r = jnp.float32(_mb_cfg.MONSTER_FIGHT_FOV_CHEBYSHEV_R)
    dx_norm = jnp.clip(d_row / fov_r, -1.0, 1.0)
    dy_norm = jnp.clip(d_col / fov_r, -1.0, 1.0)
    dist_sq_norm = jnp.where(
        has_cow > 0.5,
        (nearest_dist_h / jnp.maximum(crop_max_manh, 1.0)) ** 2,
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

    adj_cow = jnp.any((manh == 1.0) & cow_here).astype(jnp.float32)

    dir_oh = sym[_SYMBOLIC_DIR_START : _SYMBOLIC_DIR_START + 4]
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


# Compile-time check: bootstrap's feature builder must agree with config.
_EXPECTED_MONSTER_FEATS = 18
_EXPECTED_FIND_FOOD_FEATS = 10
assert MONSTER_FIGHT_TARGET_FEATURES == _EXPECTED_MONSTER_FEATS, (
    f"config.MONSTER_FIGHT_TARGET_FEATURES={MONSTER_FIGHT_TARGET_FEATURES} but "
    f"monster_fight_features_from_symbolic_obs returns {_EXPECTED_MONSTER_FEATS} floats; keep them in sync."
)
assert FIND_FOOD_TARGET_FEATURES == _EXPECTED_FIND_FOOD_FEATS, (
    f"config.FIND_FOOD_TARGET_FEATURES={FIND_FOOD_TARGET_FEATURES} but "
    f"find_food_features_from_symbolic_obs returns {_EXPECTED_FIND_FOOD_FEATS} floats; keep them in sync."
)


def make_monster_fight_obs(env, state) -> jnp.ndarray:
    """Policy observation: 18-D features from ``env.get_obs(state)`` only (no raw state)."""
    raw = env.get_obs(state)
    return monster_fight_features_from_symbolic_obs(raw)


def make_find_food_obs(env, state) -> jnp.ndarray:
    raw = env.get_obs(state)
    return find_food_features_from_symbolic_obs(raw)


def rebuild_mob_map_from_state(state) -> jnp.ndarray:
    """Rebuild boolean mob occupancy map from active mob positions/masks."""
    h, w = state.map.shape

    def _accumulate(counts: jnp.ndarray, positions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        rows = positions[:, 0].astype(jnp.int32)
        cols = positions[:, 1].astype(jnp.int32)
        contrib = jnp.zeros((h, w), dtype=jnp.int32).at[rows, cols].add(mask.astype(jnp.int32))
        return counts + contrib

    counts = jnp.zeros((h, w), dtype=jnp.int32)
    counts = _accumulate(counts, state.zombies.position, state.zombies.mask)
    counts = _accumulate(counts, state.skeletons.position, state.skeletons.mask)
    counts = _accumulate(counts, state.cows.position, state.cows.mask)
    return counts > 0


def bootstrap_monster_fight_state(state, boot_rng: jax.Array, params) -> object:
    """
    Rewrite `state` for monster-fight training/eval with variable zombie +
    variable skeleton starts.

    Spawning protocol (in order):
      1. Player is left at its env-reset position.
      2. A *variable* number of zombies is sampled per rollout from
         Uniform({MIN, ..., MAX}) with
         ``MONSTER_FIGHT_NUM_ZOMBIES_MIN`` and ``MONSTER_FIGHT_NUM_ZOMBIES_MAX``.
         We always sample MAX candidate positions, then mask the tail (same
         pattern as skeletons). Remaining zombie slots stay inactive.
      3. A *variable* number of skeletons is sampled per rollout from
         `Uniform({0, 1, ..., MONSTER_FIGHT_NUM_SKELETONS_MAX})`. The
         requested count is then placed at distinct tiles that don't
         overlap with any *active* zombie tile. Remaining skeleton slots stay
         inactive.
      4. A *variable* number of trees is sampled per rollout from
         `Uniform({MIN, ..., MAX})` using `MONSTER_FIGHT_NUM_TREES_MIN` /
         `MONSTER_FIGHT_NUM_TREES_MAX`. Trees never
         spawn on the player tile, on any tile cardinally adjacent to the
         player, on any active hostile tile, or adjacent to any active
         hostile (so every hostile is reachable from at least one cardinal
         step at reset).

    Variable mob counts are JIT-friendly: sample MAX positions, then apply
    per-rollout masks so tail slots stay health=0 / mask=False.

    Cooldowns:
      - Static zombie (`MONSTER_FIGHT_STATIC_ZOMBIE`): zombie cooldown starts
        at 1000 (inert within typical horizons).
      - Static skeleton (`MONSTER_FIGHT_STATIC_SKELETON`): same cooldown
        treatment for active skeleton slots.
      - Otherwise both use cooldown 0 — natural chase / ranged behaviour.
    """
    num_z_min = int(_mb_cfg.MONSTER_FIGHT_NUM_ZOMBIES_MIN)
    num_z_max = int(_mb_cfg.MONSTER_FIGHT_NUM_ZOMBIES_MAX)
    num_skel_max = int(_mb_cfg.MONSTER_FIGHT_NUM_SKELETONS_MAX)
    if num_z_min < 1:
        raise ValueError("MONSTER_FIGHT_NUM_ZOMBIES_MIN must be >= 1")
    if num_z_max < num_z_min:
        raise ValueError("MONSTER_FIGHT_NUM_ZOMBIES_MAX must be >= MONSTER_FIGHT_NUM_ZOMBIES_MIN")
    if num_skel_max < 0:
        raise ValueError("MONSTER_FIGHT_NUM_SKELETONS_MAX must be >= 0")

    max_z_slots = state.zombies.position.shape[0]
    if num_z_max > max_z_slots:
        raise ValueError(
            f"MONSTER_FIGHT_NUM_ZOMBIES_MAX={num_z_max} > available zombie slots "
            f"({max_z_slots}); raise static_params.max_zombies in the env."
        )
    max_s_slots = state.skeletons.position.shape[0]
    if num_skel_max > max_s_slots:
        raise ValueError(
            f"MONSTER_FIGHT_NUM_SKELETONS_MAX={num_skel_max} > available skeleton slots "
            f"({max_s_slots}); raise static_params.max_skeletons in the env."
        )

    num_trees_min = int(_mb_cfg.MONSTER_FIGHT_NUM_TREES_MIN)
    num_trees_max = int(_mb_cfg.MONSTER_FIGHT_NUM_TREES_MAX)
    if num_trees_min < 0:
        raise ValueError("MONSTER_FIGHT_NUM_TREES_MIN must be >= 0")
    if num_trees_max < num_trees_min:
        raise ValueError("MONSTER_FIGHT_NUM_TREES_MAX must be >= MONSTER_FIGHT_NUM_TREES_MIN")

    state_map = jnp.full_like(state.map, jnp.asarray(BlockType.GRASS.value, dtype=state.map.dtype))
    player_position = state.player_position.astype(jnp.int32)

    (
        boot_rng_z_pos,
        boot_rng_z_count,
        boot_rng_skel_count,
        boot_rng_skel_pos,
        boot_rng_tree_count,
        boot_rng_trees,
    ) = jax.random.split(boot_rng, 6)

    # 1) Zombies: sample MAX positions, then mask to a rollout count in [MIN, MAX].
    spawn_positions, spawn_valids = _sample_zombies_in_fov(
        state_map, player_position, boot_rng_z_pos, num_zombies=num_z_max
    )
    zombie_count = jax.random.randint(
        boot_rng_z_count,
        shape=(),
        minval=jnp.int32(num_z_min),
        maxval=jnp.int32(num_z_max + 1),
    )
    slot_idx_z = jnp.arange(num_z_max, dtype=jnp.int32)
    z_valids = spawn_valids & (slot_idx_z < zombie_count)

    # 2) Skeletons. Sample a count k uniformly in [0, MAX]; always sample
    # MAX positions, then mask the tail based on k.
    if num_skel_max > 0:
        skel_count = jax.random.randint(
            boot_rng_skel_count,
            shape=(),
            minval=jnp.int32(0),
            maxval=jnp.int32(num_skel_max + 1),
        )  # int32 in [0, MAX]
        skel_positions, skel_pos_valids = _sample_skeletons_in_fov(
            state_map,
            player_position,
            spawn_positions,
            z_valids,
            boot_rng_skel_pos,
            num_skeletons=num_skel_max,
        )
        # Drop slots beyond `skel_count` even if a position was sampled
        # successfully — this is what gives us the per-rollout variable
        # count.
        slot_idx = jnp.arange(num_skel_max, dtype=jnp.int32)
        within_count = slot_idx < skel_count
        skel_valids = skel_pos_valids & within_count
    else:
        # Static-empty arrays (still get materialised so the rest of the
        # function is dimension-agnostic).
        skel_positions = jnp.zeros((0, 2), dtype=jnp.int32)
        skel_valids = jnp.zeros((0,), dtype=jnp.bool_)

    # 3) Trees: avoid player + every active hostile (zombies AND skeletons).
    if num_skel_max > 0:
        all_mob_positions = jnp.concatenate([spawn_positions, skel_positions], axis=0)
        all_mob_valids = jnp.concatenate([z_valids, skel_valids], axis=0)
    else:
        all_mob_positions = spawn_positions
        all_mob_valids = z_valids
    tree_count = jax.random.randint(
        boot_rng_tree_count,
        shape=(),
        minval=jnp.int32(num_trees_min),
        maxval=jnp.int32(num_trees_max + 1),
    )
    state_map = _sample_tree_positions_in_fov(
        state_map,
        player_position,
        all_mob_positions,
        all_mob_valids,
        boot_rng_trees,
        tree_count,
    )

    zeros_z_cool = jnp.zeros_like(state.zombies.attack_cooldown)
    zeros_s_cool = jnp.zeros_like(state.skeletons.attack_cooldown)
    zeros_c_cool = jnp.zeros_like(state.cows.attack_cooldown)

    # Zombies: place into the first MAX slots; mask from `z_valids`.
    z_pos = jnp.zeros_like(state.zombies.position)
    z_health = jnp.zeros_like(state.zombies.health)
    z_mask = jnp.zeros_like(state.zombies.mask)
    z_cool = zeros_z_cool
    z_pos = z_pos.at[:num_z_max].set(spawn_positions.astype(z_pos.dtype))
    z_health = z_health.at[:num_z_max].set(
        jnp.where(
            z_valids,
            jnp.asarray(params.zombie_health, dtype=z_health.dtype),
            jnp.asarray(0, dtype=z_health.dtype),
        )
    )
    z_mask = z_mask.at[:num_z_max].set(z_valids.astype(z_mask.dtype))
    initial_z_cool = jnp.where(
        jnp.asarray(_mb_cfg.MONSTER_FIGHT_STATIC_ZOMBIE, dtype=jnp.bool_),
        jnp.asarray(1000, dtype=z_cool.dtype),
        jnp.asarray(0, dtype=z_cool.dtype),
    )
    z_cool_slots = jnp.where(
        z_valids,
        initial_z_cool,
        jnp.asarray(0, dtype=z_cool.dtype),
    )
    z_cool = z_cool.at[:num_z_max].set(z_cool_slots)

    # Skeletons: place into the first `num_skel_max` slots; only those with
    # `skel_valids[i]==True` end up active.
    s_pos = jnp.zeros_like(state.skeletons.position)
    s_health = jnp.zeros_like(state.skeletons.health)
    s_mask = jnp.zeros_like(state.skeletons.mask)
    s_cool = zeros_s_cool
    if num_skel_max > 0:
        s_pos = s_pos.at[:num_skel_max].set(skel_positions.astype(s_pos.dtype))
        s_health = s_health.at[:num_skel_max].set(
            jnp.where(
                skel_valids,
                jnp.asarray(params.skeleton_health, dtype=s_health.dtype),
                jnp.asarray(0, dtype=s_health.dtype),
            )
        )
        s_mask = s_mask.at[:num_skel_max].set(skel_valids.astype(s_mask.dtype))
        initial_s_cool = jnp.where(
            jnp.asarray(_mb_cfg.MONSTER_FIGHT_STATIC_SKELETON, dtype=jnp.bool_),
            jnp.asarray(1000, dtype=s_cool.dtype),
            jnp.asarray(0, dtype=s_cool.dtype),
        )
        s_cool_slots = jnp.where(
            skel_valids,
            initial_s_cool,
            jnp.asarray(0, dtype=s_cool.dtype),
        )
        s_cool = s_cool.at[:num_skel_max].set(s_cool_slots)

    # Cows always inactive.
    c_pos = jnp.zeros_like(state.cows.position)
    c_health = jnp.zeros_like(state.cows.health)
    c_mask = jnp.zeros_like(state.cows.mask)

    zombies = state.zombies.replace(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=z_cool,
    )
    skeletons = state.skeletons.replace(
        position=s_pos,
        health=s_health,
        mask=s_mask,
        attack_cooldown=s_cool,
    )
    cows = state.cows.replace(
        position=c_pos,
        health=c_health,
        mask=c_mask,
        attack_cooldown=zeros_c_cool,
    )
    staged_state = state.replace(map=state_map, zombies=zombies, skeletons=skeletons, cows=cows)
    mob_map = rebuild_mob_map_from_state(staged_state)

    arrows = state.arrows.replace(
        position=jnp.zeros_like(state.arrows.position),
        health=jnp.zeros_like(state.arrows.health),
        mask=jnp.zeros_like(state.arrows.mask),
    )

    inventory = _equip_wooden_sword(state.inventory)

    state = state.replace(
        map=state_map,
        mob_map=mob_map,
        zombies=zombies,
        skeletons=skeletons,
        cows=cows,
        arrows=arrows,
        inventory=inventory,
    )

    return state


def bootstrap_find_food_state(state, boot_rng: jax.Array, params) -> object:
    """Clears zombies/skeletons/arrows; spawns cows (+ trees). Uses MONSTER_FIGHT_FOV_* annulus."""

    num_c_min = int(_mb_cfg.FIND_FOOD_NUM_COWS_MIN)
    num_c_max = int(_mb_cfg.FIND_FOOD_NUM_COWS_MAX)
    if num_c_min < 1:
        raise ValueError("FIND_FOOD_NUM_COWS_MIN must be >= 1")
    if num_c_max < num_c_min:
        raise ValueError("FIND_FOOD_NUM_COWS_MAX must be >= FIND_FOOD_NUM_COWS_MIN")

    max_c_slots = state.cows.position.shape[0]
    if num_c_max > max_c_slots:
        raise ValueError(
            f"FIND_FOOD_NUM_COWS_MAX={num_c_max} > available cow slots ({max_c_slots})"
        )

    num_trees_min = int(_mb_cfg.FIND_FOOD_NUM_TREES_MIN)
    num_trees_max = int(_mb_cfg.FIND_FOOD_NUM_TREES_MAX)
    if num_trees_min < 0:
        raise ValueError("FIND_FOOD_NUM_TREES_MIN must be >= 0")
    if num_trees_max < num_trees_min:
        raise ValueError("FIND_FOOD_NUM_TREES_MAX must be >= FIND_FOOD_NUM_TREES_MIN")

    state_map = jnp.full_like(state.map, jnp.asarray(BlockType.GRASS.value, dtype=state.map.dtype))
    player_position = state.player_position.astype(jnp.int32)

    boot_rng_pos, boot_rng_count, boot_rng_tree_count, boot_rng_trees = jax.random.split(
        boot_rng, 4
    )

    spawn_positions, spawn_valids = _sample_zombies_in_fov(
        state_map, player_position, boot_rng_pos, num_zombies=num_c_max
    )
    cow_count = jax.random.randint(
        boot_rng_count,
        shape=(),
        minval=jnp.int32(num_c_min),
        maxval=jnp.int32(num_c_max + 1),
    )
    slot_idx_c = jnp.arange(num_c_max, dtype=jnp.int32)
    c_valids = spawn_valids & (slot_idx_c < cow_count)

    tree_count = jax.random.randint(
        boot_rng_tree_count,
        shape=(),
        minval=jnp.int32(num_trees_min),
        maxval=jnp.int32(num_trees_max + 1),
    )
    state_map = _sample_tree_positions_in_fov(
        state_map,
        player_position,
        spawn_positions,
        c_valids,
        boot_rng_trees,
        tree_count,
    )

    zeros_z_cool = jnp.zeros_like(state.zombies.attack_cooldown)
    zeros_s_cool = jnp.zeros_like(state.skeletons.attack_cooldown)
    zeros_c_cool = jnp.zeros_like(state.cows.attack_cooldown)

    z_pos = jnp.zeros_like(state.zombies.position)
    z_health = jnp.zeros_like(state.zombies.health)
    z_mask = jnp.zeros_like(state.zombies.mask)
    z_cool = zeros_z_cool

    s_pos = jnp.zeros_like(state.skeletons.position)
    s_health = jnp.zeros_like(state.skeletons.health)
    s_mask = jnp.zeros_like(state.skeletons.mask)
    s_cool = zeros_s_cool

    c_pos = jnp.zeros_like(state.cows.position)
    c_health = jnp.zeros_like(state.cows.health)
    c_mask = jnp.zeros_like(state.cows.mask)

    c_pos = c_pos.at[:num_c_max].set(spawn_positions.astype(c_pos.dtype))
    c_health = c_health.at[:num_c_max].set(
        jnp.where(
            c_valids,
            jnp.asarray(params.cow_health, dtype=c_health.dtype),
            jnp.asarray(0, dtype=c_health.dtype),
        )
    )
    c_mask = c_mask.at[:num_c_max].set(c_valids.astype(c_mask.dtype))
    initial_c_cool = jnp.where(
        jnp.asarray(_mb_cfg.FIND_FOOD_STATIC_COW, dtype=jnp.bool_),
        jnp.asarray(1000, dtype=zeros_c_cool.dtype),
        jnp.asarray(0, dtype=zeros_c_cool.dtype),
    )
    cool_slots_c = jnp.where(
        c_valids,
        initial_c_cool,
        jnp.asarray(0, dtype=zeros_c_cool.dtype),
    )
    c_cool = zeros_c_cool.at[:num_c_max].set(cool_slots_c)

    zombies = state.zombies.replace(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=z_cool,
    )
    skeletons = state.skeletons.replace(
        position=s_pos,
        health=s_health,
        mask=s_mask,
        attack_cooldown=s_cool,
    )
    cows = state.cows.replace(
        position=c_pos,
        health=c_health,
        mask=c_mask,
        attack_cooldown=c_cool,
    )
    staged_state = state.replace(
        map=state_map,
        zombies=zombies,
        skeletons=skeletons,
        cows=cows,
    )
    mob_map = rebuild_mob_map_from_state(staged_state)

    arrows = state.arrows.replace(
        position=jnp.zeros_like(state.arrows.position),
        health=jnp.zeros_like(state.arrows.health),
        mask=jnp.zeros_like(state.arrows.mask),
    )
    inventory = _equip_wooden_sword(state.inventory)

    return state.replace(
        map=state_map,
        mob_map=mob_map,
        zombies=zombies,
        skeletons=skeletons,
        cows=cows,
        arrows=arrows,
        inventory=inventory,
    )
