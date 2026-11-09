from flax import linen as nn


class DrinkWaterPolicy(nn.Module):
    hidden_size: int = 12
    num_actions: int = 5

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits
