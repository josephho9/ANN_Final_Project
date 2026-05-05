"""
Stage 06 – Rule-Based Baseline

Classifies pushup form using hard-coded joint angle thresholds only.
No ML — serves as the simplest possible reference point.

Usage:
    python baseline.py --input data/keypoints/clip_001.npy
    python baseline.py --evaluate          # runs on full test set
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple

import config
from src.preprocessing import compute_joint_angle, build_dataset
from src.dataset import make_dataloaders
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ── MediaPipe landmark indices (full 33-landmark set) ────────────────────────
RIGHT_SHOULDER = 12
RIGHT_ELBOW    = 14
RIGHT_WRIST    = 16
RIGHT_HIP      = 24
RIGHT_ANKLE    = 28


def classify_single_clip(sequence: np.ndarray) -> Tuple[int, dict]:
    """
    Rule-based classification for one pushup clip.

    Args:
        sequence: (T, 33, 2) — (x, y) per frame

    Returns:
        label: 1 (Good Form) or 0 (Poor Form)
        details: dict of computed angles vs thresholds
    """
    T = sequence.shape[0]
    # Bottom phase: frames where elbow is most bent (first half of rep)
    bottom = sequence[:T // 2, :, :]
    # Top phase: frames near full extension (second half)
    top = sequence[T // 2:, :, :]

    def mean_angle(frames, a, b, c):
        angles = [compute_joint_angle(f[a], f[b], f[c]) for f in frames]
        return float(np.mean(angles))

    elbow_angle     = mean_angle(bottom, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    back_alignment  = mean_angle(bottom, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)

    elbow_ok = config.ELBOW_ANGLE_MIN <= elbow_angle <= config.ELBOW_ANGLE_MAX
    back_ok  = config.BACK_ALIGNMENT_MIN <= back_alignment <= config.BACK_ALIGNMENT_MAX

    score = int(elbow_ok) + int(back_ok)
    label = 1 if score >= 1 else 0

    details = {
        "elbow_angle":    elbow_angle,
        "back_alignment": back_alignment,
        "elbow_ok":       elbow_ok,
        "back_ok":        back_ok,
        "score":          score,
    }
    return label, details


def evaluate_baseline():
    """Run the rule-based baseline on the test split and print metrics."""
    print("[baseline] Loading dataset …")
    X, y = build_dataset()         # X: (N, T, 66)
    _, _, test_loader, (X_test, y_test) = make_dataloaders(X, y)

    # Reshape sequences back to (T, 33, 2)
    N, T, _ = X_test.shape
    X_seq = X_test.reshape(N, T, config.NUM_KEYPOINTS, config.KEYPOINT_DIM)

    preds = []
    for i in range(N):
        label, _ = classify_single_clip(X_seq[i])
        preds.append(label)

    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="binary")

    print("\n" + "=" * 50)
    print("Rule-Based Baseline Results")
    print("=" * 50)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Poor Form", "Good Form"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based pushup form baseline")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a single .npy keypoint file (T, 33, 2) or (T, 66)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on full test set")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_baseline()
    elif args.input:
        seq = np.load(args.input)
        if seq.ndim == 2:
            seq = seq.reshape(seq.shape[0], config.NUM_KEYPOINTS, config.KEYPOINT_DIM)
        label, details = classify_single_clip(seq)
        print(f"\nPrediction: {'Good Form' if label == 1 else 'Poor Form'}")
        for k, v in details.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
