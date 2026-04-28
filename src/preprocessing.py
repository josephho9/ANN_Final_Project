"""
Stage 02 – Frame Extraction & Preprocessing

Converts raw video clips into fixed-length keypoint sequences ready for
the temporal models. Also handles data augmentation.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.pose_estimation import extract_keypoints_from_video


# ── Video → Keypoint sequence ─────────────────────────────────────────────────

def process_video_to_keypoints(video_path: str, output_dir: str) -> str:
    """Extract keypoints from a video and save to output_dir/<stem>.npy."""
    stem = Path(video_path).stem
    out_path = str(Path(output_dir) / f"{stem}.npy")
    extract_keypoints_from_video(video_path, output_path=out_path)
    return out_path


def pad_or_truncate(sequence: np.ndarray, target_len: int = config.SEQUENCE_LEN) -> np.ndarray:
    """
    Ensure a keypoint sequence has exactly `target_len` frames.
    Short clips are zero-padded at the end; long clips are center-cropped.
    """
    T = len(sequence)
    if T == target_len:
        return sequence
    if T < target_len:
        pad = np.zeros((target_len - T, *sequence.shape[1:]), dtype=sequence.dtype)
        return np.concatenate([sequence, pad], axis=0)
    # Center crop
    start = (T - target_len) // 2
    return sequence[start: start + target_len]


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize x,y coordinates relative to the hip midpoint so the
    representation is camera-position invariant.

    sequence: (T, 17, 3) — (x, y, visibility)
    Returns:  (T, 17, 3) normalized
    """
    seq = sequence.copy()
    # Hip midpoint: landmark indices 7 (left hip) and 8 (right hip) in our 17-kp subset
    LEFT_HIP_IDX = 7
    RIGHT_HIP_IDX = 8

    hip_mid = (seq[:, LEFT_HIP_IDX, :2] + seq[:, RIGHT_HIP_IDX, :2]) / 2  # (T, 2)
    # Translate so hip midpoint is at origin
    seq[:, :, :2] -= hip_mid[:, np.newaxis, :]

    # Scale by torso height (shoulder mid → hip mid distance)
    LEFT_SHOULDER_IDX = 1
    RIGHT_SHOULDER_IDX = 2
    shoulder_mid = (seq[:, LEFT_SHOULDER_IDX, :2] + seq[:, RIGHT_SHOULDER_IDX, :2]) / 2
    torso_height = np.linalg.norm(shoulder_mid, axis=-1, keepdims=True) + 1e-6  # (T, 1)
    seq[:, :, :2] /= torso_height[:, np.newaxis, :]

    return seq


def flatten_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Flatten (T, 17, 3) → (T, 51) for use as LSTM/Transformer input.
    """
    return sequence.reshape(sequence.shape[0], -1)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    labels_csv: str = config.LABELS_FILE,
    keypoints_dir: str = config.KEYPOINTS_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all keypoint sequences and their labels.

    CSV format:
        clip_id,label      (label: 1=good form, 0=poor form)

    Returns:
        X: (N, SEQUENCE_LEN, INPUT_DIM)
        y: (N,)
    """
    df = pd.read_csv(labels_csv)
    X_list, y_list = [], []

    for _, row in df.iterrows():
        npy_path = Path(keypoints_dir) / f"{row['clip_id']}.npy"
        if not npy_path.exists():
            print(f"[WARN] Missing keypoints for {row['clip_id']}, skipping.")
            continue
        seq = np.load(str(npy_path))          # (T, 17, 3)
        seq = pad_or_truncate(seq)             # (SEQUENCE_LEN, 17, 3)
        seq = normalize_keypoints(seq)         # center + scale
        seq = flatten_sequence(seq)            # (SEQUENCE_LEN, 51)
        X_list.append(seq)
        y_list.append(int(row["label"]))

    X = np.stack(X_list, axis=0)              # (N, T, 51)
    y = np.array(y_list, dtype=np.int64)      # (N,)
    return X, y


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a keypoint sequence (T, 51).
    Augmentations: horizontal flip, speed jitter (time warp), Gaussian noise.
    """
    seq = sequence.reshape(sequence.shape[0], config.NUM_KEYPOINTS, config.KEYPOINT_DIM)

    # Horizontal flip: negate x-coordinates
    if np.random.rand() < 0.5:
        seq = seq.copy()
        seq[:, :, 0] *= -1

    # Speed jitter: randomly stretch/compress time by ±20 %
    if np.random.rand() < 0.5:
        T = seq.shape[0]
        factor = np.random.uniform(0.8, 1.2)
        new_T = max(1, int(T * factor))
        indices = np.linspace(0, T - 1, new_T)
        seq = np.array([
            seq[int(i)] * (1 - i % 1) + seq[min(int(i) + 1, T - 1)] * (i % 1)
            for i in indices
        ], dtype=np.float32)
        seq = pad_or_truncate(seq.reshape(new_T, config.NUM_KEYPOINTS, config.KEYPOINT_DIM))

    # Gaussian noise
    if np.random.rand() < 0.5:
        seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)

    return seq.reshape(seq.shape[0], -1)


def compute_joint_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Compute the angle at joint B given three 2D points A, B, C.
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
