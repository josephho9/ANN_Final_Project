"""
preprocessing.py

Loads the keypoint .npy files and gets them ready for the models.
Also handles normalization and augmentation.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def pad_or_truncate(sequence, target_len=config.SEQUENCE_LEN):
    """Make sure every clip is exactly target_len frames.
    Shorter clips get zero-padded at the end.
    Longer clips get center-cropped.
    """
    T = len(sequence)
    if T == target_len:
        return sequence
    if T < target_len:
        pad_shape = (target_len - T,) + sequence.shape[1:]
        pad = np.zeros(pad_shape, dtype=sequence.dtype)
        return np.concatenate([sequence, pad], axis=0)
    # center crop if too long
    start = (T - target_len) // 2
    return sequence[start : start + target_len]


def normalize_keypoints(sequence):
    """Normalize so position in frame doesn't matter.

    sequence shape: (T, 33, 2)

    Steps:
    1. Shift so hip midpoint is at origin
    2. Scale by torso height so body size doesn't matter
    """
    seq = sequence.copy()

    # hip midpoint (landmarks 23 and 24 are left/right hip)
    hip_mid = (seq[:, 23, :] + seq[:, 24, :]) / 2.0   # (T, 2)
    seq -= hip_mid[:, np.newaxis, :]

    # torso height = distance from shoulder mid to hip mid (after centering)
    shoulder_mid = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    torso_h = np.linalg.norm(shoulder_mid, axis=-1, keepdims=True) + 1e-6  # (T, 1)
    seq /= torso_h[:, np.newaxis, :]

    return seq


def flatten_sequence(sequence):
    """(T, 33, 2) -> (T, 66) so it works as model input."""
    return sequence.reshape(sequence.shape[0], -1)


def build_dataset(correct_npy=config.CORRECT_NPY, incorrect_npy=config.INCORRECT_NPY):
    """Load both keypoint files and return X, y arrays.

    Returns:
        X: (N, 150, 66)
        y: (N,)  — 1 = good form, 0 = poor form
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

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def compute_joint_angle(a, b, c):
    """Angle at joint B given 3 points A, B, C. Returns degrees."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


# this code was AI generated
def augment_sequence(sequence):
    """Random augmentations on a (T, 66) keypoint sequence.
    Applied only during training.
    Does: horizontal flip, speed jitter, gaussian noise.
    """
    seq = sequence.reshape(sequence.shape[0], config.NUM_KEYPOINTS, config.KEYPOINT_DIM)

    # flip horizontally with 50% chance
    if np.random.rand() < 0.5:
        seq = seq.copy()
        seq[:, :, 0] *= -1

    # speed jitter — stretch or compress the sequence in time by up to 20%
    # this code was AI generated
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

    # small gaussian noise
    if np.random.rand() < 0.5:
        seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)

    return seq.reshape(seq.shape[0], -1)
