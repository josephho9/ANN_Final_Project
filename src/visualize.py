"""
Bonus Output – Per-joint angle visualization

For a given keypoint sequence, computes joint angles at each frame and
plots which joints deviate from optimal form thresholds.
Gives players actionable feedback beyond just a score.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.preprocessing import compute_joint_angle


# ── Joint definitions (indices into our 17-keypoint subset) ──────────────────
# Each entry: (joint_name, proximal_idx, vertex_idx, distal_idx)
JOINT_DEFS = [
    ("R_Elbow",   4,  3,  5),   # right shoulder → elbow → wrist  (shooting arm)
    ("L_Elbow",   1,  2,  4),   # left shoulder → elbow → wrist
    ("R_Knee",    9,  8, 10),   # right hip → knee → ankle  (wait — check indices)
    ("L_Knee",    6,  7,  9),
    ("R_Wrist",   3,  5, 15),   # elbow → wrist → index finger
    ("L_Wrist",   2,  4, 14),
]

# Optimal angle range per joint (degrees)
OPTIMAL_RANGES: Dict[str, tuple] = {
    "R_Elbow": (config.ELBOW_ANGLE_MIN, config.ELBOW_ANGLE_MAX),
    "L_Elbow": (config.ELBOW_ANGLE_MIN, config.ELBOW_ANGLE_MAX),
    "R_Knee":  (config.KNEE_BEND_MIN,   config.KNEE_BEND_MAX),
    "L_Knee":  (config.KNEE_BEND_MIN,   config.KNEE_BEND_MAX),
    "R_Wrist": (config.WRIST_ANGLE_MIN, 180),
    "L_Wrist": (config.WRIST_ANGLE_MIN, 180),
}


def compute_joint_angles_sequence(
    sequence: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame joint angles for all defined joints.

    Args:
        sequence: (T, 17, 3) keypoint sequence (x, y, visibility)

    Returns:
        Dict mapping joint_name → (T,) array of angles in degrees.
    """
    angles: Dict[str, List[float]] = {name: [] for name, *_ in JOINT_DEFS}

    for frame_kps in sequence:  # (17, 3)
        for name, a_idx, b_idx, c_idx in JOINT_DEFS:
            a = frame_kps[a_idx, :2]
            b = frame_kps[b_idx, :2]
            c = frame_kps[c_idx, :2]
            angle = compute_joint_angle(a, b, c)
            angles[name].append(angle)

    return {name: np.array(vals) for name, vals in angles.items()}


def plot_joint_angles(
    sequence: np.ndarray,
    predicted_label: int,
    confidence: float,
    save_path: Optional[str] = None,
):
    """
    Plot joint angle trajectories over the shot arc, shading regions that
    deviate from optimal form.

    Args:
        sequence: (T, 17, 3) keypoint sequence
        predicted_label: 0 = Poor Form, 1 = Good Form
        confidence: model's confidence in the prediction
        save_path: if given, save figure to this path
    """
    angles = compute_joint_angles_sequence(sequence)
    T = sequence.shape[0]
    frames = np.arange(T)

    label_str = "Good Form" if predicted_label == 1 else "Poor Form"
    label_color = "#2ecc71" if predicted_label == 1 else "#e74c3c"

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"Joint Angle Analysis  |  Prediction: {label_str} ({confidence:.1%} confidence)",
        fontsize=14, fontweight="bold", color=label_color,
    )
    fig.patch.set_facecolor("#1a1a1a")

    for ax, (name, _, _, _) in zip(axes.flat, JOINT_DEFS):
        joint_angles = angles[name]
        lo, hi = OPTIMAL_RANGES[name]

        ax.set_facecolor("#2a2a2a")
        ax.plot(frames, joint_angles, color="#f39c12", linewidth=2, label=name)
        ax.axhspan(lo, hi, alpha=0.25, color="#2ecc71", label="Optimal range")

        # Shade deviating regions
        below = joint_angles < lo
        above = joint_angles > hi
        if below.any():
            ax.fill_between(frames, joint_angles, lo, where=below,
                            alpha=0.4, color="#e74c3c", label="Below optimal")
        if above.any():
            ax.fill_between(frames, joint_angles, hi, where=above,
                            alpha=0.4, color="#3498db", label="Above optimal")

        ax.set_title(name, color="white", fontsize=11)
        ax.set_xlabel("Frame", color="gray")
        ax.set_ylabel("Angle (°)", color="gray")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        legend = ax.legend(fontsize=8, loc="upper right", framealpha=0.3)
        for text in legend.get_texts():
            text.set_color("white")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[visualize] Saved joint angle plot → {save_path}")
    else:
        plt.show()
    plt.close()


def generate_feedback(sequence: np.ndarray) -> List[str]:
    """
    Rule-based per-joint feedback strings for actionable coaching cues.
    Returns a list of feedback strings (empty list = no issues detected).
    """
    angles = compute_joint_angles_sequence(sequence)
    feedback = []

    # Use release frame = last 10 frames mean
    release_slice = slice(-10, None)

    r_elbow = np.mean(angles["R_Elbow"][release_slice])
    lo, hi = OPTIMAL_RANGES["R_Elbow"]
    if r_elbow < lo:
        feedback.append(f"Shooting elbow too tucked at release ({r_elbow:.0f}°). Target {lo}–{hi}°.")
    elif r_elbow > hi:
        feedback.append(f"Shooting elbow flaring out at release ({r_elbow:.0f}°). Target {lo}–{hi}°.")

    r_knee = np.mean(angles["R_Knee"][:15])  # first 15 frames = jump load
    lo, hi = OPTIMAL_RANGES["R_Knee"]
    if r_knee < lo:
        feedback.append(f"Knees too bent during jump ({r_knee:.0f}°). Target {lo}–{hi}°.")
    elif r_knee > hi:
        feedback.append(f"Not enough knee bend at jump apex ({r_knee:.0f}°). Target {lo}–{hi}°.")

    r_wrist = np.mean(angles["R_Wrist"][release_slice])
    lo, hi = OPTIMAL_RANGES["R_Wrist"]
    if r_wrist < lo:
        feedback.append(f"Wrist not snapping through at follow-through ({r_wrist:.0f}°). Target >{lo}°.")

    return feedback


def overlay_skeleton_on_video(
    video_path: str,
    keypoints_seq: np.ndarray,
    output_path: str,
    score: Optional[float] = None,
    label: Optional[int] = None,
):
    """
    Write a new video with the MediaPipe skeleton and form score overlaid.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Connections between our 17 landmark indices (pairs)
    CONNECTIONS = [
        (0, 1), (0, 2),          # nose → shoulders
        (1, 3), (2, 4),          # shoulder → elbow
        (3, 5), (4, 6),          # elbow → wrist
        (1, 7), (2, 8),          # shoulder → hip
        (7, 8),                  # hip bar
        (7, 9), (8, 10),         # hip → knee
        (9, 11), (10, 12),       # knee → ankle
    ]

    label_str = ("Good Form" if label == 1 else "Poor Form") if label is not None else ""
    color_map = {1: (46, 204, 113), 0: (231, 76, 60)}
    line_color = color_map.get(label, (242, 156, 18))

    for frame_idx in range(len(keypoints_seq)):
        ret, frame = cap.read()
        if not ret:
            break
        kps = keypoints_seq[frame_idx]  # (17, 3)

        # Draw skeleton
        for a_idx, b_idx in CONNECTIONS:
            ax, ay, av = kps[a_idx]
            bx, by, bv = kps[b_idx]
            if av > 0.4 and bv > 0.4:
                p1 = (int(ax * w), int(ay * h))
                p2 = (int(bx * w), int(by * h))
                cv2.line(frame, p1, p2, line_color, 2)

        for x, y, vis in kps:
            if vis > 0.4:
                cv2.circle(frame, (int(x * w), int(y * h)), 5, (255, 165, 0), -1)

        # Overlay score
        if score is not None:
            cv2.putText(frame, f"Score: {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, line_color, 2)
        if label_str:
            cv2.putText(frame, label_str, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, line_color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[visualize] Annotated video → {output_path}")
