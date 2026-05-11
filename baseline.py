"""
baseline.py

Rule-based pushup classifier — no ML at all.
Just checks joint angles against thresholds.
Used as a sanity check / lower bound for the neural models.

Usage:
    python baseline.py --evaluate
    python baseline.py --input path/to/clip.npy
"""

import argparse
import numpy as np
from pathlib import Path

import config
from src.preprocessing import compute_joint_angle, build_dataset
from src.dataset import make_dataloaders
from sklearn.metrics import accuracy_score, f1_score, classification_report


# mediapipe landmark indices we care about
RIGHT_SHOULDER = 12
RIGHT_ELBOW    = 14
RIGHT_WRIST    = 16
RIGHT_HIP      = 24
RIGHT_ANKLE    = 28


def classify_clip(sequence):
    """Classify one clip using joint angle thresholds.

    sequence: (T, 33, 2)
    returns: (label, details_dict)
        label = 1 if good form, 0 if poor form
    """
    T = sequence.shape[0]
    # look at the bottom half of the rep (where the person is going down)
    bottom = sequence[: T // 2]

    # average elbow and back angle across those frames
    elbow_angles = [compute_joint_angle(f[RIGHT_SHOULDER], f[RIGHT_ELBOW], f[RIGHT_WRIST]) for f in bottom]
    back_angles  = [compute_joint_angle(f[RIGHT_SHOULDER], f[RIGHT_HIP],   f[RIGHT_ANKLE]) for f in bottom]

    mean_elbow = float(np.mean(elbow_angles))
    mean_back  = float(np.mean(back_angles))

    elbow_ok = config.ELBOW_ANGLE_MIN <= mean_elbow <= config.ELBOW_ANGLE_MAX
    back_ok  = config.BACK_ALIGNMENT_MIN <= mean_back <= config.BACK_ALIGNMENT_MAX

    # need at least one condition to pass — OR rule
    label = 1 if (elbow_ok or back_ok) else 0

    details = {
        "elbow_angle":    mean_elbow,
        "back_alignment": mean_back,
        "elbow_ok":       elbow_ok,
        "back_ok":        back_ok,
    }
    return label, details


def evaluate_baseline():
    print("[baseline] loading dataset...")
    X, y = build_dataset()
    _, _, test_loader, (X_test, y_test) = make_dataloaders(X, y)

    # reshape from (N, T, 66) back to (N, T, 33, 2) so we can compute angles
    N, T, _ = X_test.shape
    X_seq = X_test.reshape(N, T, config.NUM_KEYPOINTS, config.KEYPOINT_DIM)

    preds = []
    for i in range(N):
        label, _ = classify_clip(X_seq[i])
        preds.append(label)

    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="binary")

    print("\n" + "=" * 50)
    print("Rule-Based Baseline")
    print("=" * 50)
    print(f"  accuracy: {acc:.4f}")
    print(f"  f1:       {f1:.4f}")
    print()
    print(classification_report(y_test, preds, target_names=["Poor Form", "Good Form"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="run on full test set")
    parser.add_argument("--input",    default=None,        help="path to a single .npy clip")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_baseline()
    elif args.input:
        seq = np.load(args.input)
        if seq.ndim == 2:
            # flat (T, 66) -> (T, 33, 2)
            seq = seq.reshape(seq.shape[0], config.NUM_KEYPOINTS, config.KEYPOINT_DIM)
        label, details = classify_clip(seq)
        print(f"\nprediction: {'Good Form' if label == 1 else 'Poor Form'}")
        for k, v in details.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
