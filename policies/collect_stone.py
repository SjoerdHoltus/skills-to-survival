"""Small MLP for collect-stone observations (movement + DO)."""

from flax import linen as nn

from configs import COLLECT_STONE_HIDDEN_SIZE, COLLECT_STONE_NUM_ACTIONS


class CollectStonePolicy(nn.Module):
    hidden_size: int = COLLECT_STONE_HIDDEN_SIZE
    num_actions: int = COLLECT_STONE_NUM_ACTIONS

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        return nn.Dense(self.num_actions)(x)
