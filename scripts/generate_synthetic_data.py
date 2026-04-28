"""
Synthetic keypoint data generator — lets you test the full pipeline
immediately without any real video clips.

Generates physically-plausible jump shot keypoint sequences:
  - Good Form: smooth arc, elbow angle 80–110° at release, knee bend 100–160°
  - Poor Form: erratic angles, out-of-range joints

Usage:
    python scripts/generate_synthetic_data.py --n 200
    python scripts/generate_synthetic_data.py --n 500 --seed 123
"""

import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

KP_DIR    = Path(config.KEYPOINTS_DIR)
LABEL_CSV = Path(config.LABELS_FILE)

T  = config.SEQUENCE_LEN       # 60 frames
N  = config.NUM_KEYPOINTS      # 17
D  = config.KEYPOINT_DIM       # 3  (x, y, vis)

# ── Rough normalized skeleton at "rest" position ──────────────────────────────
# (x, y) relative to hip midpoint, torso-height normalized
# Indices follow config.LANDMARK_INDICES order
BASE_POSE = np.array([
    [ 0.00, -1.80, 1.0],   # 0  nose
    [-0.25, -1.40, 1.0],   # 1  left shoulder
    [ 0.25, -1.40, 1.0],   # 2  right shoulder
    [-0.45, -0.95, 1.0],   # 3  left elbow
    [ 0.45, -0.95, 1.0],   # 4  right elbow
    [-0.35, -0.50, 1.0],   # 5  left wrist
    [ 0.35, -0.50, 1.0],   # 6  right wrist
    [-0.15,  0.00, 1.0],   # 7  left hip
    [ 0.15,  0.00, 1.0],   # 8  right hip
    [-0.20,  0.60, 1.0],   # 9  left knee
    [ 0.20,  0.60, 1.0],   # 10 right knee
    [-0.22,  1.20, 1.0],   # 11 left ankle
    [ 0.22,  1.20, 1.0],   # 12 right ankle
    [-0.50, -0.45, 0.8],   # 13 left pinky
    [ 0.50, -0.45, 0.8],   # 14 right pinky
    [-0.40, -0.48, 0.9],   # 15 left index
    [ 0.40, -0.48, 0.9],   # 16 right index
], dtype=np.float32)


def _shot_arc(t_norm: float, label: int) -> np.ndarray:
    """
    Generate one frame's keypoints at normalized time t_norm ∈ [0, 1].
    label=1 → good form, label=0 → poor form.
    """
    pose = BASE_POSE.copy()

    # ── Jump: body rises then falls ───────────────────────────────────────────
    jump_height = -0.3 * np.sin(np.pi * t_norm)   # negative = upward in image coords
    pose[:, 1] += jump_height

    # ── Arm raising (shooting arm = right, indices 2,4,6,14,16) ──────────────
    raise_frac = np.clip(t_norm * 2, 0, 1)  # arm goes up in first half
    arm_raise = raise_frac * 0.9            # radians

    if label == 1:  # Good form: controlled arc
        elbow_angle_rad = np.radians(95 - 15 * np.sin(np.pi * t_norm))
        wrist_snap = raise_frac
    else:           # Poor form: flaring elbow, no wrist snap
        elbow_angle_rad = np.radians(140 - 20 * t_norm)  # too wide
        wrist_snap = 0.0

    # Move right elbow and wrist
    pose[4, 0] = pose[2, 0] + 0.35 * np.cos(arm_raise)
    pose[4, 1] = pose[2, 1] - 0.45 * np.sin(arm_raise)
    pose[6, 0] = pose[4, 0] + 0.30 * np.cos(arm_raise + elbow_angle_rad - np.pi / 2)
    pose[6, 1] = pose[4, 1] - 0.30 * np.sin(arm_raise + elbow_angle_rad - np.pi / 2)

    # ── Knee bend (loading phase = first 25 %) ─────────────────────────────
    if t_norm < 0.25:
        knee_bend = t_norm / 0.25
        knee_offset = 0.15 * knee_bend
        if label == 0:
            knee_offset *= 0.4  # insufficient knee bend
        pose[9, 1]  += knee_offset
        pose[10, 1] += knee_offset
        pose[11, 1] += knee_offset * 0.5
        pose[12, 1] += knee_offset * 0.5

    # ── Add noise ──────────────────────────────────────────────────────────
    noise_scale = 0.02 if label == 1 else 0.06
    pose[:, :2] += np.random.normal(0, noise_scale, (N, 2)).astype(np.float32)

    return pose


def generate_clip(label: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single (T, 17, 3) synthetic clip."""
    seq = []
    # Random phase offset and speed jitter
    phase = rng.uniform(0, 0.1)
    speed = rng.uniform(0.85, 1.15)
    for t in range(T):
        t_norm = np.clip((t / T) * speed + phase, 0, 1)
        frame = _shot_arc(t_norm, label)
        seq.append(frame)
    return np.stack(seq, axis=0).astype(np.float32)   # (T, 17, 3)


def generate_dataset(n_clips: int = 200, seed: int = config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    KP_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ~60 % good, ~40 % poor (matches proposal)
    n_good = int(n_clips * 0.6)
    n_poor = n_clips - n_good

    labels = []
    total = 0

    print(f"[synthetic] Generating {n_good} good-form clips …")
    for i in range(n_good):
        clip_id = f"syn_good_{i+1:04d}"
        clip = generate_clip(label=1, rng=rng)
        np.save(KP_DIR / f"{clip_id}.npy", clip)
        labels.append(f"{clip_id},1")
        total += 1

    print(f"[synthetic] Generating {n_poor} poor-form clips …")
    for i in range(n_poor):
        clip_id = f"syn_poor_{i+1:04d}"
        clip = generate_clip(label=0, rng=rng)
        np.save(KP_DIR / f"{clip_id}.npy", clip)
        labels.append(f"{clip_id},0")
        total += 1

    with open(LABEL_CSV, "w") as f:
        f.write("clip_id,label\n")
        f.write("\n".join(labels) + "\n")

    print(f"\n[synthetic] Done. {total} clips → {KP_DIR}/")
    print(f"            Labels → {LABEL_CSV}")
    print("\nYou can now run the full pipeline:")
    print("  python -m src.train --model lstm")
    print("  python -m src.evaluate --model lstm")
    print("  python compare_models.py --quick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Total clips to generate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    generate_dataset(n_clips=args.n, seed=args.seed)
