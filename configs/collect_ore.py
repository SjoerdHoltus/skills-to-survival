"""Collect-ore policy (target block behind stone walls; obs encodes which ore).

Train on one ``COLLECT_ORE_TRAINING_TARGET`` (default coal). The same weights apply at
eval/meta for other deploy targets (iron, diamond): only ``target_id`` / bootstrap block
type and inventory channel change in the 15-D obs.
"""

COLLECT_ORE_TARGET_FEATURES = 11
COLLECT_ORE_BLOCKER_FEATURES = 4
COLLECT_ORE_OBS_SIZE = COLLECT_ORE_TARGET_FEATURES + COLLECT_ORE_BLOCKER_FEATURES

# Deploy-time target ids (eval, batch eval, survival meta). Train on coal only.
COLLECT_ORE_TARGET_NAMES = ("coal", "iron", "diamond")
COLLECT_ORE_TRAINING_TARGET = "coal"
COLLECT_ORE_TRAINING_TARGET_ID = COLLECT_ORE_TARGET_NAMES.index(COLLECT_ORE_TRAINING_TARGET)

COLLECT_ORE_ROLLOUT_STEPS = 10
COLLECT_ORE_HIDDEN_SIZE = 12
COLLECT_ORE_NUM_ACTIONS = 5

COLLECT_ORE_MAX_COAL = 9  # Classic inventory cap
COLLECT_ORE_MAX_IRON = 9
COLLECT_ORE_MAX_DIAMOND = 1  # achievement / single-gem eval cap

COLLECT_ORE_BONUS = 10.0
COLLECT_ORE_STEP_PENALTY = 0.008
COLLECT_ORE_DO_FACING_STONE_BONUS = 0.5  # dense signal to break stone walls
COLLECT_ORE_DO_FACING_ORE_BONUS = 0.5  # dense signal to mine visible ore
COLLECT_ORE_WALL_BREAK_BONUS = 0.5  # stone tile cleared from the map
COLLECT_ORE_REACH_MANHATTAN_THRESHOLD = 1
COLLECT_ORE_FOV_NORM_CHEBYSHEV_R = 4
# Cardinal distance from player to ore (bootstrap). Max 4 only applies horizontally;
# vertical placement is capped at (OBS_DIM[0]-1)//2 == 3 so ore stays in symbolic FOV.
COLLECT_ORE_ORE_DISTANCE_MIN = 3
COLLECT_ORE_ORE_DISTANCE_MAX = 4

# Wall-shell depth curriculum during training (not target switching).
COLLECT_ORE_WALL_CURRICULUM_PHASE1_FRACTION = 0.40  # 1-tile shell
COLLECT_ORE_WALL_CURRICULUM_PHASE2_FRACTION = 0.30  # 2-tile shell (cumulative 0.70)
# remainder of training: 3-tile shells

COLLECT_ORE_LEARNING_RATE = 0.05
COLLECT_ORE_STD_START = 0.2
COLLECT_ORE_STD_END = 0.05
COLLECT_ORE_POLICY_TEMPERATURE_START = 2.0
COLLECT_ORE_POLICY_TEMPERATURE_END = 0.5
COLLECT_ORE_POLICY_TEMPERATURE_ANNEAL_FRACTION = 1.0
COLLECT_ORE_GREEDY_FINETUNE_FRACTION = 0.25
COLLECT_ORE_ROLLOUT_REFRESH_EVERY = 12
COLLECT_ORE_NUM_ROLLOUTS_START = 8
COLLECT_ORE_NUM_ROLLOUTS_ANNEAL_FRACTION = 1.0

# 15-D feature indices (11-D target layout + 4 stone blockers).
COLLECT_ORE_FEAT_HAS_TARGET = 2
COLLECT_ORE_FEAT_BLOCK_DX = 11
COLLECT_ORE_FEAT_BLOCK_DY = 12
COLLECT_ORE_FEAT_ADJ_STONE = 13
COLLECT_ORE_FEAT_FACE_STONE = 14
