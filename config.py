"""
Global configuration for Basketball Jumpshot Tracker.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_VIDEO_DIR = f"{DATA_DIR}/raw_videos"
PROCESSED_DIR = f"{DATA_DIR}/processed"
KEYPOINTS_DIR = f"{DATA_DIR}/keypoints"
LABELS_FILE = f"{DATA_DIR}/labels.csv"

# ── Pose estimation ───────────────────────────────────────────────────────────
NUM_KEYPOINTS = 17          # MediaPipe BlazePose upper-body + lower-body landmarks used
KEYPOINT_DIM = 3            # (x, y, visibility)
INPUT_DIM = NUM_KEYPOINTS * KEYPOINT_DIM   # flattened per-frame feature vector

# MediaPipe landmark indices used (subset of 33 BlazePose landmarks → 17 relevant)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_INDICES = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
    17,  # left pinky
    18,  # right pinky
    19,  # left index
    20,  # right index
]

# ── Preprocessing ─────────────────────────────────────────────────────────────
FRAME_SIZE = (640, 480)     # resize target (W, H)
TARGET_FPS = 30
SEQUENCE_LEN = 60           # frames per shot clip (2 s @ 30 fps)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ── Model ─────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_HEADS = 4               # Transformer attention heads
FF_DIM = 256                # Transformer feed-forward dim
NUM_CLASSES = 2             # Good Form / Poor Form

# ── Evaluation ────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"

# ── Rule-based baseline thresholds ────────────────────────────────────────────
# Joint angle thresholds for "good form" classification
ELBOW_ANGLE_MIN = 80        # degrees — shooting elbow at set point
ELBOW_ANGLE_MAX = 110
KNEE_BEND_MIN = 100         # degrees — knee at jump apex
KNEE_BEND_MAX = 160
WRIST_ANGLE_MIN = 150       # degrees — wrist at follow-through (near straight)
