"""Monster-fight policy and world bootstrap."""

# Monster-fight **policy** observation: a small vector built **only** from the
# raw 1345-dim Craftax-Classic symbolic obs (`env.get_obs` / `render_craftax_symbolic`).
# See `training.monster_world_bootstrap.monster_fight_features_from_symbolic_obs`.
#
# Feature layout (18 floats — indices 12..17 are arrow-related in the crop):
#   0..3 — direction-to-nearest-LIVE-HOSTILE in the 7×9 symbolic crop
#   4..7 — blocked_LEFT/RIGHT/UP/DOWN
#   8    — is_adjacent_hostile
#   9    — is_facing_hostile
#  10    — player_health_norm
#  11    — is_skeleton
#  12..17 — arrow features from symbolic mob channel

MONSTER_FIGHT_TARGET_FEATURES = 18
MONSTER_FIGHT_OBS_SIZE = MONSTER_FIGHT_TARGET_FEATURES

MONSTER_FIGHT_ROLLOUT_STEPS = 90
MONSTER_FIGHT_HIDDEN_SIZE = 12

# Per active step
MONSTER_FIGHT_STEP_PENALTY = 0.05

# Reward for killing the zombie
MONSTER_FIGHT_KILL_REWARD = 10.0

MONSTER_FIGHT_DO_FACING_HOSTILE_BONUS = 0.5

# Small penalty for DO when it does not produce immediate combat effect.
MONSTER_FIGHT_WASTED_DO_PENALTY = 0.02

MONSTER_FIGHT_LEARNING_RATE = 0.05
MONSTER_FIGHT_STD_START = 0.2
MONSTER_FIGHT_STD_END = 0.05

MONSTER_FIGHT_POLICY_TEMPERATURE_START = 2.0
MONSTER_FIGHT_POLICY_TEMPERATURE_END = 0.5
MONSTER_FIGHT_POLICY_TEMPERATURE_ANNEAL_FRACTION = 1.0
MONSTER_FIGHT_POLICY_TEMPERATURE = MONSTER_FIGHT_POLICY_TEMPERATURE_START
MONSTER_FIGHT_GREEDY_FINETUNE_FRACTION = 0.2

# Monster fight world bootstrap
MONSTER_FIGHT_NUM_ZOMBIES_MIN = 1
MONSTER_FIGHT_NUM_ZOMBIES_MAX = 3
MONSTER_FIGHT_STATIC_ZOMBIE = False
MONSTER_FIGHT_NUM_SKELETONS_MAX = 2
MONSTER_FIGHT_STATIC_SKELETON = False
MONSTER_FIGHT_NUM_TREES_MIN = 0
MONSTER_FIGHT_NUM_TREES_MAX = 9

# Monster fight FOV
MONSTER_FIGHT_FOV_MIN_CHEBYSHEV_DIST = 2
MONSTER_FIGHT_FOV_CHEBYSHEV_R = 4

MONSTER_FIGHT_PLAYER_MAX_HEALTH = 9

MONSTER_TOUCH_PROGRESS_ALPHA = 5.0

MONSTER_FIGHT_ROLLOUT_REFRESH_EVERY = 12

MONSTER_FIGHT_NUM_ROLLOUTS_START = 8
MONSTER_FIGHT_NUM_ROLLOUTS_ANNEAL_FRACTION = 1.0
