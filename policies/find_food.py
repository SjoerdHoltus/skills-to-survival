"""Policy for hunt-cow / find-food on Craftax-Classic."""

from __future__ import annotations

from flax import linen as nn

from configs import FIND_FOOD_HIDDEN_SIZE, FIND_WOOD_NUM_ACTIONS


class FindFoodPolicy(nn.Module):
    """obs -> logits for narrow LEFT/RIGHT/UP/DOWN/DO (Craftax indices 1..5)."""

    hidden_size: int = FIND_FOOD_HIDDEN_SIZE
    num_actions: int = FIND_WOOD_NUM_ACTIONS

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits
