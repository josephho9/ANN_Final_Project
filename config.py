# config.py
# just putting all the settings in one place so we don't have to
# hardcode stuff everywhere

# paths to the pre-extracted keypoint arrays
CORRECT_NPY   = "/Users/josephho/Downloads/dataset/labels/correct.npy"
INCORRECT_NPY = "/Users/josephho/Downloads/dataset/labels/incorrect.npy"

# misc data dirs
DATA_DIR        = "data"
KEYPOINTS_DIR   = "data/keypoints"
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR     = "results"

# mediapipe gives us 33 landmarks, each with x and y
# so each frame is 33*2 = 66 numbers
NUM_KEYPOINTS = 33
KEYPOINT_DIM  = 2
INPUT_DIM     = 66   # 33 * 2

# we standardize everything to 150 frames (5 sec @ 30fps)
SEQUENCE_LEN = 150
TARGET_FPS   = 30
FRAME_SIZE   = (640, 480)

# training hyperparams
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42

# model architecture sizes
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_HEADS   = 4     # for transformer
FF_DIM      = 256   # transformer feedforward
NUM_CLASSES = 2     # good form / poor form

# joint angle thresholds for the rule-based baseline
# elbow should be around 90 degrees at the bottom of a pushup
# back should be basically straight (close to 180)
ELBOW_ANGLE_MIN    = 70
ELBOW_ANGLE_MAX    = 110
BACK_ALIGNMENT_MIN = 160
BACK_ALIGNMENT_MAX = 180
