"""Value network for hierarchical meta PPO (critic on meta obs)."""

from flax import linen as nn

from configs import OBS_SIZE, SURVIVAL_PPO_CRITIC_HIDDEN


class HierarchicalCritic(nn.Module):
    """Two-layer MLP: obs -> scalar V(s)."""

    hidden_size: int = SURVIVAL_PPO_CRITIC_HIDDEN
    obs_size: int = OBS_SIZE

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)
