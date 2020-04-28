# CONSTANT VARIABLES
########################################################
SIZE = 350
LOW = 0
ACTION_SPACE = 4
OBSERVATION_SPACE = (2,)
TITLE = "Find The Center"

HM_EPISODES = 600 # how many episodes
MAX_MOVE = 150
MOVE_PENALTY = 1
PENALTY = 5
REWARD = 500

#how higher is epsilon, agent makes more random action
EPSILON = 0.5
EPS_DECAY = 0.9998

LEARNING_RATE = 1e-3
DISCOUNT = 0.95
BATCH_SIZE = 32

VELOCITY = 10

RED_ARC_DIAMETER = 20
BLACK_ARC_DIAMETER = 65

MAX_POS = SIZE - RED_ARC_DIAMETER
########################################################