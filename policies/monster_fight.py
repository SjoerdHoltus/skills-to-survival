"""Tiny policy for the monster-fight task on Craftax-Classic.

The policy consumes the compact feature vector from
`training.monster_world_bootstrap.make_monster_fight_obs` (derived only from the
flat 1345-dim symbolic observation).

Setting `hidden_size = 0` collapses the network to a single Dense(num_actions)
head: ``obs_dim * 5 + 5`` parameters (``obs_dim`` = ``MONSTER_FIGHT_OBS_SIZE``).
ES can exploit tiny obs when every weight ties directly to the visible symbols.

For values >0 we keep the original 2-hidden-layer MLP, which is useful for
later curriculum stages that need non-linear features (e.g. an "DO if
adjacent" rule on top of the direction features).
"""

from __future__ import annotations

from flax import linen as nn


class MonsterFightPolicy(nn.Module):
    """obs -> action logits (narrow 0..4 -> env actions 1..5).

    When `hidden_size <= 0`, the policy is a single linear layer (no hidden
    units, no ReLU). Otherwise it's a 2-layer MLP with ReLU activations.
    """

    hidden_size: int = 12
    num_actions: int = 5

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits
