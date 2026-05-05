"""
Dataset visualization — generates four plots saved to results/:

  1. skeleton_comparison.png   — side-by-side skeleton snapshots (correct vs incorrect)
  2. joint_angles.png          — elbow & back angle trajectories over the rep
  3. keypoint_heatmap.png      — per-landmark motion variance (which joints move most)
  4. dataset_overview.png      — mean ± std of joint angles across all 50 clips per class

Usage:
    python visualize_dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

CORRECT_NPY   = "/Users/josephho/Downloads/dataset/labels/correct.npy"
INCORRECT_NPY = "/Users/josephho/Downloads/dataset/labels/incorrect.npy"
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# MediaPipe BlazePose connections (33 landmarks)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

LANDMARK_NAMES = [
    "nose","l_eye_inner","l_eye","l_eye_outer",
    "r_eye_inner","r_eye","r_eye_outer",
    "l_ear","r_ear","l_mouth","r_mouth",
    "l_shoulder","r_shoulder","l_elbow","r_elbow",
    "l_wrist","r_wrist","l_pinky","r_pinky",
    "l_index","r_index","l_thumb","r_thumb",
    "l_hip","r_hip","l_knee","r_knee",
    "l_ankle","r_ankle","l_heel","r_heel",
    "l_foot","r_foot",
]


def compute_angle(a, b, c):
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def angles_for_clip(seq):
    """seq: (T, 33, 2) → elbow angles (T,) and back angles (T,)"""
    elbows = [compute_angle(seq[t, 12], seq[t, 14], seq[t, 16]) for t in range(len(seq))]
    backs  = [compute_angle(seq[t, 12], seq[t, 24], seq[t, 28]) for t in range(len(seq))]
    return np.array(elbows), np.array(backs)


def load():
    correct   = np.load(CORRECT_NPY).reshape(50, 150, 33, 2)
    incorrect = np.load(INCORRECT_NPY).reshape(50, 150, 33, 2)
    return correct, incorrect


def remap(seq):
    """Remap normalized coords to [0.05, 0.95] for display."""
    xy = seq[:, :2].copy()
    mn, mx = xy.min(), xy.max()
    return (xy - mn) / (mx - mn + 1e-6) * 0.9 + 0.05


# ── Plot 1: Skeleton comparison ───────────────────────────────────────────────

def plot_skeleton_comparison(correct, incorrect):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.patch.set_facecolor("#111")
    fig.suptitle("Skeleton Snapshots — Correct (top) vs Incorrect (bottom)",
                 color="white", fontsize=14, fontweight="bold")

    clip_indices = [0, 5, 10, 20, 35]
    frame_idx = 40  # mid-rep frame

    for col, ci in enumerate(clip_indices):
        for row, (clips, color, label) in enumerate([
            (correct,   "#2ecc71", "Correct"),
            (incorrect, "#e74c3c", "Incorrect"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("#1a1a1a")
            ax.set_xlim(0, 1); ax.set_ylim(1, 0)
            ax.set_aspect("equal")
            ax.axis("off")

            kps = clips[ci, frame_idx]  # (33, 2)
            xy  = remap(kps)

            for a, b in CONNECTIONS:
                ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                        color=color, lw=1.5, alpha=0.7)
            ax.scatter(xy[:, 0], xy[:, 1], c=color, s=20, zorder=5)

            if row == 0:
                ax.set_title(f"Clip {ci+1}", color="gray", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, color=color, fontsize=10)

    plt.tight_layout()
    path = RESULTS_DIR / "skeleton_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Saved → {path}")


# ── Plot 2: Joint angle trajectories ─────────────────────────────────────────

def plot_joint_angles(correct, incorrect):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#111")
    fig.suptitle("Joint Angle Trajectories (3 sample clips each class)",
                 color="white", fontsize=14, fontweight="bold")

    for ax, clips, color, label in [
        (axes[0], correct,   "#2ecc71", "Correct"),
        (axes[1], incorrect, "#e74c3c", "Incorrect"),
    ]:
        ax.set_facecolor("#1a1a1a")
        frames = np.arange(150)

        for i in [0, 10, 25]:
            e, b = angles_for_clip(clips[i])
            ax.plot(frames, e, color=color,      lw=1.5, alpha=0.8,
                    label="Elbow" if i == 0 else "_")
            ax.plot(frames, b, color="#f39c12", lw=1.5, alpha=0.8, linestyle="--",
                    label="Back align." if i == 0 else "_")

        ax.axhspan(70, 110, alpha=0.15, color=color,    label="Elbow optimal")
        ax.axhspan(160, 180, alpha=0.15, color="#f39c12", label="Back optimal")

        ax.set_title(label, color=color, fontsize=12)
        ax.set_xlabel("Frame", color="gray")
        ax.set_ylabel("Angle (°)", color="gray")
        ax.tick_params(colors="gray")
        ax.set_ylim(0, 200)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        leg = ax.legend(fontsize=8, framealpha=0.3)
        for t in leg.get_texts(): t.set_color("white")

    plt.tight_layout()
    path = RESULTS_DIR / "joint_angles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Saved → {path}")


# ── Plot 3: Landmark motion heatmap ──────────────────────────────────────────

def plot_keypoint_heatmap(correct, incorrect):
    # Variance of each landmark's position across all frames + clips
    var_correct   = correct.var(axis=(0, 1))    # (33, 2)
    var_incorrect = incorrect.var(axis=(0, 1))

    total_correct   = var_correct.sum(axis=1)    # (33,)
    total_incorrect = var_incorrect.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    x = np.arange(33)
    w = 0.4
    ax.bar(x - w/2, total_correct,   w, color="#2ecc71", alpha=0.85, label="Correct")
    ax.bar(x + w/2, total_incorrect, w, color="#e74c3c", alpha=0.85, label="Incorrect")

    ax.set_xticks(x)
    ax.set_xticklabels(LANDMARK_NAMES, rotation=60, ha="right", fontsize=7, color="gray")
    ax.set_ylabel("Position Variance", color="gray")
    ax.set_title("Landmark Motion Variance — Which joints differ between classes?",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    leg = ax.legend(framealpha=0.3)
    for t in leg.get_texts(): t.set_color("white")

    plt.tight_layout()
    path = RESULTS_DIR / "keypoint_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Saved → {path}")


# ── Plot 4: Mean ± std angle across all clips ─────────────────────────────────

def plot_dataset_overview(correct, incorrect):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#111")
    fig.suptitle("Mean ± Std Joint Angles Across All 50 Clips per Class",
                 color="white", fontsize=14, fontweight="bold")

    frames = np.arange(150)

    for clips, color, label, col in [
        (correct,   "#2ecc71", "Correct",   0),
        (incorrect, "#e74c3c", "Incorrect", 1),
    ]:
        all_elbows = np.array([angles_for_clip(clips[i])[0] for i in range(50)])
        all_backs  = np.array([angles_for_clip(clips[i])[1] for i in range(50)])

        for ax, data, joint_label, opt_lo, opt_hi in [
            (axes[0, col], all_elbows, "Elbow Angle", 70,  110),
            (axes[1, col], all_backs,  "Back Alignment", 160, 180),
        ]:
            ax.set_facecolor("#1a1a1a")
            mean = data.mean(axis=0)
            std  = data.std(axis=0)

            ax.plot(frames, mean, color=color, lw=2)
            ax.fill_between(frames, mean - std, mean + std,
                            color=color, alpha=0.25, label="±1 std")
            ax.axhspan(opt_lo, opt_hi, alpha=0.15, color="white", label="Optimal range")

            ax.set_title(f"{label} — {joint_label}", color=color, fontsize=11)
            ax.set_xlabel("Frame", color="gray")
            ax.set_ylabel("Angle (°)", color="gray")
            ax.tick_params(colors="gray")
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
            leg = ax.legend(fontsize=8, framealpha=0.3)
            for t in leg.get_texts(): t.set_color("white")

    plt.tight_layout()
    path = RESULTS_DIR / "dataset_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[viz] Loading dataset …")
    correct, incorrect = load()
    print(f"[viz] Correct: {correct.shape}  Incorrect: {incorrect.shape}")

    plot_skeleton_comparison(correct, incorrect)
    plot_joint_angles(correct, incorrect)
    plot_keypoint_heatmap(correct, incorrect)
    plot_dataset_overview(correct, incorrect)

    print(f"\n[viz] All plots saved to {RESULTS_DIR}/")
