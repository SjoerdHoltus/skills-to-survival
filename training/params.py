"""Parameter conversion between flat arrays and Flax pytrees."""

import jax
import jax.numpy as jnp


def pytree_to_flat_params(params):
    """Flatten a pytree of parameters into a 1D array."""
    leaves, tree_def = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    flat = jnp.concatenate([l.ravel() for l in leaves])
    return flat, shapes, tree_def


def flat_params_to_pytree(flat_params, param_shapes, param_tree_def):
    """Unflatten a 1D array back into a pytree of parameters."""
    params_list = []
    offset = 0
    for shape in param_shapes:
        size = 1
        for s in shape:
            size *= s
        params_list.append(flat_params[offset : offset + size].reshape(shape))
        offset += size
    return jax.tree_util.tree_unflatten(param_tree_def, params_list)
