"""
Global configuration for Pushup Form Tracker.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_VIDEO_DIR = f"{DATA_DIR}/raw_videos"
PROCESSED_DIR = f"{DATA_DIR}/processed"
KEYPOINTS_DIR = f"{DATA_DIR}/keypoints"

# Pre-extracted pushup keypoint arrays (50 clips each, shape: N x 150 x 66)
CORRECT_NPY   = "/Users/josephho/Downloads/dataset/labels/correct.npy"
INCORRECT_NPY = "/Users/josephho/Downloads/dataset/labels/incorrect.npy"

# ── Pose estimation ───────────────────────────────────────────────────────────
NUM_KEYPOINTS = 33          # all MediaPipe BlazePose landmarks
KEYPOINT_DIM = 2            # (x, y) — no visibility channel
INPUT_DIM = NUM_KEYPOINTS * KEYPOINT_DIM   # 66 — flattened per-frame feature vector

# ── Preprocessing ─────────────────────────────────────────────────────────────
FRAME_SIZE = (640, 480)     # resize target (W, H)
TARGET_FPS = 30
SEQUENCE_LEN = 150          # frames per pushup rep clip (5 s @ 30 fps)

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
# Joint angle thresholds for "good form" pushup classification
ELBOW_ANGLE_MIN = 70        # degrees — elbow at bottom of pushup (~90°)
ELBOW_ANGLE_MAX = 110
BACK_ALIGNMENT_MIN = 160    # degrees — shoulder-hip-ankle should be near straight
BACK_ALIGNMENT_MAX = 180
