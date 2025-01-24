from enum import Enum

class WandbMetrics(Enum):
    AVG_SCORE = 'avg_score'
    EPOCH_STEP = 'epoch_step'
    EPSILON = 'epsilon'
    LOSS = 'loss'
    SCORE = 'score'
    SCORE_EPOCH = 'score_epoch'
    STEP_COUNTER = 'step_counter'

class WandbConfigKeys(Enum):
    DISCOUNT_FACTOR = 'discount_factor'
    LEARNING_RATE = 'learning_rate'
    EPSILON = 'epsilon'
    EPSILON_DECAY = 'epsilon_decay'
    EPSILON_MIN = 'epsilon_min'
    BATCH_SIZE = 'batch_size'
    TRAIN_START = 'train_start'
    TARGET_UPDATE_PERIOD = 'target_update_period'
    HIDDEN_SIZE = 'hidden_size'