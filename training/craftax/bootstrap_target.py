"""FOV map helpers for full Craftax skill bootstraps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax.constants import OBS_DIM, BlockType

SYMBOLIC_FOV_PR = int(OBS_DIM[0] // 2)
SYMBOLIC_FOV_PC = int(OBS_DIM[1] // 2)

_ORE_BLOCK_VALUES = jnp.asarray(
    [BlockType.STONE.value, BlockType.COAL.value, BlockType.IRON.value, BlockType.DIAMOND.value],
    dtype=jnp.int32,
)


def symbolic_fov_world_mask(
    map_shape: tuple[int, int] | list[int], player: jnp.ndarray
) -> jnp.ndarray:
    h, w = int(map_shape[0]), int(map_shape[1])
    rr = jnp.arange(h, dtype=jnp.int32)[:, None]
    cc = jnp.arange(w, dtype=jnp.int32)[None, :]
    pp = player.astype(jnp.int32)
    return (jnp.abs(rr - pp[0]) <= jnp.int32(SYMBOLIC_FOV_PR)) & (
        jnp.abs(cc - pp[1]) <= jnp.int32(SYMBOLIC_FOV_PC)
    )


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
    threshold = jnp.where(nb_eff <= 0, jnp.float32(-jnp.inf), sorted_scores[nb_eff - 1])
    is_block = (scores <= threshold) & candidate
    return jnp.where(is_block, block_value.astype(state_map.dtype), state_map)


def strip_natural_ores(state_map: jnp.ndarray) -> jnp.ndarray:
    g = jnp.asarray(BlockType.GRASS.value, dtype=state_map.dtype)
    out = state_map
    for bv in _ORE_BLOCK_VALUES:
        out = jnp.where(out == bv, g, out)
    return out
