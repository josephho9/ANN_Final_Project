"""
visualize.py

Plots joint angle trajectories for a pushup clip and generates
plain-English coaching feedback based on the angles.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.preprocessing import compute_joint_angle


# joints we care about for pushup form analysis
# format: (name, point_A_idx, vertex_idx, point_C_idx)
JOINT_DEFS = [
    ("R_Elbow",     12, 14, 16),   # right shoulder -> elbow -> wrist
    ("L_Elbow",     11, 13, 15),   # left side
    ("R_Back",      12, 24, 28),   # shoulder -> hip -> ankle (back straightness)
    ("L_Back",      11, 23, 27),
    ("R_Hip_Angle", 14, 12, 24),   # elbow -> shoulder -> hip (torso tilt)
    ("L_Hip_Angle", 13, 11, 23),
]

# what we consider "good" for each joint
OPTIMAL_RANGES = {
    "R_Elbow":     (config.ELBOW_ANGLE_MIN,    config.ELBOW_ANGLE_MAX),
    "L_Elbow":     (config.ELBOW_ANGLE_MIN,    config.ELBOW_ANGLE_MAX),
    "R_Back":      (config.BACK_ALIGNMENT_MIN, config.BACK_ALIGNMENT_MAX),
    "L_Back":      (config.BACK_ALIGNMENT_MIN, config.BACK_ALIGNMENT_MAX),
    "R_Hip_Angle": (150, 180),
    "L_Hip_Angle": (150, 180),
}


def compute_joint_angles_sequence(sequence):
    """Compute angle at each joint for every frame.

    sequence: (T, 33, 2)
    returns dict: joint_name -> (T,) array of angles in degrees
    """
    angles = {name: [] for name, *_ in JOINT_DEFS}

    for frame_kps in sequence:
        for name, a_idx, b_idx, c_idx in JOINT_DEFS:
            angle = compute_joint_angle(frame_kps[a_idx], frame_kps[b_idx], frame_kps[c_idx])
            angles[name].append(angle)

    return {name: np.array(vals) for name, vals in angles.items()}


# this code was AI generated
def plot_joint_angles(sequence, predicted_label, confidence, save_path=None):
    """Plot angle trajectories for all joints with shading for out-of-range regions.

    sequence: (T, 33, 2)
    predicted_label: 0 or 1
    confidence: model confidence float
    """
    angles = compute_joint_angles_sequence(sequence)
    T = sequence.shape[0]
    frames = np.arange(T)

    label_str   = "Good Form" if predicted_label == 1 else "Poor Form"
    label_color = "#2ecc71"   if predicted_label == 1 else "#e74c3c"

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"Joint Angle Analysis  |  {label_str} ({confidence:.1%} confidence)",
        fontsize=14, fontweight="bold", color=label_color,
    )
    fig.patch.set_facecolor("#1a1a1a")

    for ax, (name, _, _, _) in zip(axes.flat, JOINT_DEFS):
        joint_angles = angles[name]
        lo, hi = OPTIMAL_RANGES[name]

        ax.set_facecolor("#2a2a2a")
        ax.plot(frames, joint_angles, color="#f39c12", linewidth=2, label=name)
        ax.axhspan(lo, hi, alpha=0.25, color="#2ecc71", label="optimal range")

        below = joint_angles < lo
        above = joint_angles > hi
        if below.any():
            ax.fill_between(frames, joint_angles, lo, where=below, alpha=0.4,
                            color="#e74c3c", label="below optimal")
        if above.any():
            ax.fill_between(frames, joint_angles, hi, where=above, alpha=0.4,
                            color="#3498db", label="above optimal")

        ax.set_title(name, color="white", fontsize=11)
        ax.set_xlabel("frame", color="gray")
        ax.set_ylabel("angle (deg)", color="gray")
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
        print(f"[visualize] saved joint angle plot -> {save_path}")
    else:
        plt.show()
    plt.close()


def generate_feedback(sequence):
    """Check joint angles and return a list of coaching cues.

    sequence: (T, 33, 2)
    returns: list of feedback strings (empty = no issues found)
    """
    angles = compute_joint_angles_sequence(sequence)
    feedback = []
    T = len(next(iter(angles.values())))
    bottom = slice(0, T // 2)  # look at the descent phase

    # elbow depth
    r_elbow = np.mean(angles["R_Elbow"][bottom])
    lo, hi = OPTIMAL_RANGES["R_Elbow"]
    if r_elbow < lo:
        feedback.append(f"Elbows going too deep at bottom ({r_elbow:.0f}°). Target {lo}–{hi}°.")
    elif r_elbow > hi:
        feedback.append(f"Not going low enough — elbow angle too wide ({r_elbow:.0f}°). Target {lo}–{hi}°.")

    # back alignment (the main one)
    r_back = np.mean(angles["R_Back"][bottom])
    lo, hi = OPTIMAL_RANGES["R_Back"]
    if r_back < lo:
        feedback.append(f"Hips sagging — keep your back straight ({r_back:.0f}°). Target {lo}–{hi}°.")

    l_back = np.mean(angles["L_Back"][bottom])
    lo, hi = OPTIMAL_RANGES["L_Back"]
    if l_back < lo:
        feedback.append(f"Lower back rounding detected ({l_back:.0f}°). Engage your core.")

    return feedback


def overlay_skeleton_on_video(video_path, keypoints_seq, output_path, score=None, label=None):
    """Write a new video with skeleton and form verdict overlaid.

    keypoints_seq: (T, 33, 2)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # this code was AI generated
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
        (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
        (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    label_str = ""
    line_color = (242, 156, 18)
    if label == 1:
        label_str  = "Good Form"
        line_color = (46, 204, 113)
    elif label == 0:
        label_str  = "Poor Form"
        line_color = (231, 76, 60)

    for frame_idx in range(len(keypoints_seq)):
        ret, frame = cap.read()
        if not ret:
            break
        kps = keypoints_seq[frame_idx]

        for a_idx, b_idx in POSE_CONNECTIONS:
            p1 = (int(kps[a_idx][0] * w), int(kps[a_idx][1] * h))
            p2 = (int(kps[b_idx][0] * w), int(kps[b_idx][1] * h))
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
    print(f"[visualize] annotated video -> {output_path}")
