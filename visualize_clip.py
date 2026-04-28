"""
Visualize pose estimation on a jump shot video or .npy keypoint file.

Usage:
    python visualize_clip.py --video my_shot.mp4
    python visualize_clip.py --npy data/keypoints/syn_good_0001.npy
    python visualize_clip.py --npy data/keypoints/syn_good_0001.npy --classify --model lstm
    python visualize_clip.py --video my_shot.mp4 --slow 0.25
    python visualize_clip.py --video my_shot.mp4 --save
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
import config
from src.pose_estimation import PoseEstimator, extract_keypoints_from_video
from src.preprocessing import normalize_keypoints, flatten_sequence, pad_or_truncate

# Connections between our 17-keypoint subset
CONNECTIONS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
    (1, 7), (2, 8),
    (7, 8),
    (7, 9), (8, 10),
    (9, 11), (10, 12),
]


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def draw_keypoints(frame, keypoints):
    """Draw our 17-keypoint skeleton on a blank or real frame."""
    h, w = frame.shape[:2]
    pts = {}
    for i, (x, y, vis) in enumerate(keypoints):
        if vis > 0.3:
            pts[i] = (int(x * w), int(y * h))
            cv2.circle(frame, pts[i], 6, (0, 165, 255), -1)
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (255, 165, 0), 2)


def draw_joint_angles(frame, keypoints):
    h, w = frame.shape[:2]

    def pt(idx):
        x, y, vis = keypoints[idx]
        return (x, y) if vis > 0.3 else None

    for name, a, b, c, lo, hi in [
        ("Elbow", 2, 4, 6,  config.ELBOW_ANGLE_MIN, config.ELBOW_ANGLE_MAX),
        ("Knee",  8, 10, 12, config.KNEE_BEND_MIN,   config.KNEE_BEND_MAX),
    ]:
        if all(pt(i) is not None for i in [a, b, c]):
            angle = compute_angle(pt(a), pt(b), pt(c))
            px = int(keypoints[b][0] * w) + 12
            py = int(keypoints[b][1] * h)
            ok = lo <= angle <= hi
            color = (46, 204, 113) if ok else (60, 76, 231)
            cv2.putText(frame, f"{name} {angle:.0f}", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def load_classifier(model_name, device=None):
    import torch
    from src.models import build_model
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = Path(config.CHECKPOINTS_DIR) / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        print(f"[visualize] No checkpoint at {ckpt_path}")
        return None, device
    model = build_model(model_name).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[visualize] Loaded {model_name} on {device}.")
    return model, device


def classify_sequence(seq_raw, classifier, device):
    """seq_raw: (T, 17, 3) → (label, confidence)"""
    import torch
    seq = pad_or_truncate(seq_raw)
    seq = normalize_keypoints(seq)
    seq = flatten_sequence(seq)
    x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(x), dim=-1).squeeze().cpu().numpy()
    label = int(probs.argmax())
    return label, float(probs[label])


# ── Mode 1: .npy file (synthetic or pre-extracted keypoints) ─────────────────

def run_on_npy(npy_path: str, use_classifier: bool, model_name: str, slow: float):
    seq = np.load(npy_path, allow_pickle=True)   # (T, 17, 3)
    T = len(seq)
    W, H = 800, 600
    delay = max(1, int((1000 / config.TARGET_FPS) / slow))

    # Denormalize: the synthetic data uses torso-relative coords, so
    # remap x/y from [-~2, ~2] range into [0.1, 0.9] for display
    xy = seq[:, :, :2]
    xy_min, xy_max = xy.min(), xy.max()
    seq_display = seq.copy()
    seq_display[:, :, :2] = (xy - xy_min) / (xy_max - xy_min + 1e-6) * 0.8 + 0.1
    seq_display[:, :, 2] = 1.0          # treat all as visible

    # Classifier
    label_str, label_color, pred_conf = None, None, None
    if use_classifier:
        classifier, device = load_classifier(model_name)
        if classifier:
            label, conf = classify_sequence(seq, classifier, device)
            pred_conf  = conf
            label_str  = "Good Form" if label == 1 else "Poor Form"
            label_color = (46, 204, 113) if label == 1 else (60, 76, 231)
            print(f"[visualize] {label_str} ({conf:.1%})")

    print(f"[visualize] Playing {T} frames — Q to quit, SPACE to pause.")
    paused = False
    frame_idx = 0

    while True:
        if not paused:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)      # dark background

            kps = seq_display[frame_idx]
            draw_keypoints(frame, kps)
            draw_joint_angles(frame, kps)

            # Frame counter
            cv2.putText(frame, f"Frame {frame_idx+1}/{T}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.putText(frame, Path(npy_path).stem, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            if label_str:
                cv2.putText(frame, f"{label_str}  {pred_conf:.0%}",
                            (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, label_color, 2)

            frame_idx = (frame_idx + 1) % T   # loop

        cv2.imshow("Jump Shot — Keypoint Viewer", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cv2.destroyAllWindows()


# ── Mode 2: video file ────────────────────────────────────────────────────────

def run_on_video(video_path: str, use_classifier: bool, model_name: str, save: bool, slow: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay  = max(1, int((1000 / fps) / slow))

    writer = None
    if save:
        out_path = str(Path(video_path).with_suffix("")) + "_pose.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"[visualize] Saving to {out_path}")

    label_str, label_color, pred_conf = None, None, None
    if use_classifier:
        classifier, device = load_classifier(model_name)
        if classifier:
            print("[visualize] Pre-extracting keypoints …")
            full_seq = extract_keypoints_from_video(video_path)
            label, conf = classify_sequence(full_seq, classifier, device)
            pred_conf   = conf
            label_str   = "Good Form" if label == 1 else "Poor Form"
            label_color = (46, 204, 113) if label == 1 else (60, 76, 231)
            print(f"[visualize] {label_str} ({conf:.1%})")

    print("[visualize] Playing — Q to quit, SPACE to pause.")
    paused = False
    frame = None

    with PoseEstimator() as estimator:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

                keypoints, landmarks_33 = estimator.extract_keypoints_with_world(frame)
                if keypoints is not None and landmarks_33 is not None:
                    frame = estimator.draw_full_skeleton(frame, landmarks_33)
                    draw_joint_angles(frame, keypoints)
                else:
                    cv2.putText(frame, "No pose detected", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if label_str:
                    cv2.putText(frame, f"{label_str}  {pred_conf:.0%}",
                                (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, label_color, 2)
                if writer:
                    writer.write(frame)

            if frame is not None:
                cv2.imshow("Jump Shot — Pose Estimation", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to video file (.mp4, .mov, …)")
    group.add_argument("--npy",   help="Path to keypoint file (.npy)")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--model",    default="lstm")
    parser.add_argument("--save",     action="store_true", help="Save annotated video (video mode only)")
    parser.add_argument("--slow",     type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 0.25 = quarter speed)")
    args = parser.parse_args()

    if args.npy:
        run_on_npy(args.npy, args.classify, args.model, args.slow)
    else:
        run_on_video(args.video, args.classify, args.model, args.save, args.slow)
