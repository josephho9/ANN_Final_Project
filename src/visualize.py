"""
Bonus Output – Per-joint angle visualization

For a given keypoint sequence, computes joint angles at each frame and
plots which joints deviate from optimal pushup form thresholds.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.preprocessing import compute_joint_angle


# ── Joint definitions (MediaPipe full 33-landmark indices) ───────────────────
# Each entry: (joint_name, proximal_idx, vertex_idx, distal_idx)
JOINT_DEFS = [
    ("R_Elbow",        12, 14, 16),   # right shoulder → elbow → wrist
    ("L_Elbow",        11, 13, 15),   # left shoulder → elbow → wrist
    ("R_Back",         12, 24, 28),   # right shoulder → hip → ankle (back alignment)
    ("L_Back",         11, 23, 27),   # left shoulder → hip → ankle
    ("R_Hip_Angle",    14, 12, 24),   # elbow → shoulder → hip (torso tilt)
    ("L_Hip_Angle",    13, 11, 23),
]

# Optimal angle range per joint (degrees)
OPTIMAL_RANGES: Dict[str, tuple] = {
    "R_Elbow":    (config.ELBOW_ANGLE_MIN,      config.ELBOW_ANGLE_MAX),
    "L_Elbow":    (config.ELBOW_ANGLE_MIN,      config.ELBOW_ANGLE_MAX),
    "R_Back":     (config.BACK_ALIGNMENT_MIN,   config.BACK_ALIGNMENT_MAX),
    "L_Back":     (config.BACK_ALIGNMENT_MIN,   config.BACK_ALIGNMENT_MAX),
    "R_Hip_Angle": (150, 180),
    "L_Hip_Angle": (150, 180),
}


def compute_joint_angles_sequence(
    sequence: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame joint angles for all defined joints.

    Args:
        sequence: (T, 33, 2) keypoint sequence (x, y)

    Returns:
        Dict mapping joint_name → (T,) array of angles in degrees.
    """
    angles: Dict[str, List[float]] = {name: [] for name, *_ in JOINT_DEFS}

    for frame_kps in sequence:  # (33, 2)
        for name, a_idx, b_idx, c_idx in JOINT_DEFS:
            a = frame_kps[a_idx]
            b = frame_kps[b_idx]
            c = frame_kps[c_idx]
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
    Plot joint angle trajectories over the pushup rep, shading regions that
    deviate from optimal form.

    Args:
        sequence: (T, 33, 2) keypoint sequence
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
    Rule-based per-joint feedback strings for actionable pushup coaching cues.
    Returns a list of feedback strings (empty list = no issues detected).
    """
    angles = compute_joint_angles_sequence(sequence)
    feedback = []

    T = len(next(iter(angles.values())))
    bottom_slice = slice(0, T // 2)

    r_elbow = np.mean(angles["R_Elbow"][bottom_slice])
    lo, hi = OPTIMAL_RANGES["R_Elbow"]
    if r_elbow < lo:
        feedback.append(f"Elbows going too deep at bottom ({r_elbow:.0f}°). Target {lo}–{hi}°.")
    elif r_elbow > hi:
        feedback.append(f"Not going low enough — elbow angle too wide ({r_elbow:.0f}°). Target {lo}–{hi}°.")

    r_back = np.mean(angles["R_Back"][bottom_slice])
    lo, hi = OPTIMAL_RANGES["R_Back"]
    if r_back < lo:
        feedback.append(f"Hips sagging — keep your back straight ({r_back:.0f}°). Target {lo}–{hi}°.")

    l_back = np.mean(angles["L_Back"][bottom_slice])
    lo, hi = OPTIMAL_RANGES["L_Back"]
    if l_back < lo:
        feedback.append(f"Lower back rounding detected ({l_back:.0f}°). Engage your core.")

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
    keypoints_seq: (T, 33, 2)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Key MediaPipe pose connections for all 33 landmarks
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
        (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
        (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
    ]

    label_str = ("Good Form" if label == 1 else "Poor Form") if label is not None else ""
    color_map = {1: (46, 204, 113), 0: (231, 76, 60)}
    line_color = color_map.get(label, (242, 156, 18))

    for frame_idx in range(len(keypoints_seq)):
        ret, frame = cap.read()
        if not ret:
            break
        kps = keypoints_seq[frame_idx]  # (33, 2)

        for a_idx, b_idx in POSE_CONNECTIONS:
            ax, ay = kps[a_idx]
            bx, by = kps[b_idx]
            p1 = (int(ax * w), int(ay * h))
            p2 = (int(bx * w), int(by * h))
            cv2.line(frame, p1, p2, line_color, 2)

        for x, y in kps:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (255, 165, 0), -1)

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
