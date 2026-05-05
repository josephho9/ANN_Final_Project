"""
Stage 02 – Frame Extraction & Preprocessing

Converts raw video clips into fixed-length keypoint sequences ready for
the temporal models. Also handles data augmentation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config



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

    sequence: (T, 33, 2) — (x, y) for all MediaPipe landmarks
    Returns:  (T, 33, 2) normalized
    """
    seq = sequence.copy()
    # Full MediaPipe landmark indices: left hip=23, right hip=24
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    hip_mid = (seq[:, LEFT_HIP_IDX, :] + seq[:, RIGHT_HIP_IDX, :]) / 2  # (T, 2)
    seq -= hip_mid[:, np.newaxis, :]

    # Scale by torso height (shoulder mid → hip mid distance)
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    shoulder_mid = (seq[:, LEFT_SHOULDER_IDX, :] + seq[:, RIGHT_SHOULDER_IDX, :]) / 2
    torso_height = np.linalg.norm(shoulder_mid, axis=-1, keepdims=True) + 1e-6  # (T, 1)
    seq /= torso_height[:, np.newaxis, :]

    return seq


def flatten_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Flatten (T, 33, 2) → (T, 66) for use as LSTM/Transformer input.
    """
    return sequence.reshape(sequence.shape[0], -1)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    correct_npy: str = config.CORRECT_NPY,
    incorrect_npy: str = config.INCORRECT_NPY,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted pushup keypoint sequences.

    Arrays shape: (N, 150, 66) — N clips × 150 frames × 33 landmarks × (x,y).
    Label: 1 = good form (correct), 0 = poor form (incorrect).

    Returns:
        X: (N, SEQUENCE_LEN, INPUT_DIM)
        y: (N,)
    """
    correct   = np.load(correct_npy).astype(np.float32)    # (50, 150, 66)
    incorrect = np.load(incorrect_npy).astype(np.float32)  # (50, 150, 66)

    X_list, y_list = [], []

    for raw in correct:
        seq = raw.reshape(config.SEQUENCE_LEN, config.NUM_KEYPOINTS, config.KEYPOINT_DIM)
        seq = normalize_keypoints(seq)
        seq = flatten_sequence(seq)
        X_list.append(seq)
        y_list.append(1)

    for raw in incorrect:
        seq = raw.reshape(config.SEQUENCE_LEN, config.NUM_KEYPOINTS, config.KEYPOINT_DIM)
        seq = normalize_keypoints(seq)
        seq = flatten_sequence(seq)
        X_list.append(seq)
        y_list.append(0)

    X = np.stack(X_list, axis=0)          # (N, 150, 66)
    y = np.array(y_list, dtype=np.int64)  # (N,)
    return X, y


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a keypoint sequence (T, 66).
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
        seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)

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
