"""Meta-policy for survival hierarchy: logits over frozen skills + primitive moves."""

from flax import linen as nn

from configs import SURVIVAL_META_HIDDEN, SURVIVAL_META_NUM_MODES, SURVIVAL_META_OBS_SIZE


class HierarchicalSurvivalMetaPolicy(nn.Module):
    """Two-layer MLP: full symbolic obs -> logits over wood / drink / food / fight / craft / primitives / SLEEP."""

    hidden_size: int = SURVIVAL_META_HIDDEN
    num_modes: int = SURVIVAL_META_NUM_MODES
    obs_size: int = SURVIVAL_META_OBS_SIZE

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_modes)(x)
