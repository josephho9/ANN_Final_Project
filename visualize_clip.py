"""
Visualize pose estimation on a pushup video or .npy keypoint file.

Usage:
    python visualize_clip.py --video my_pushup.mp4
    python visualize_clip.py --npy data/keypoints/clip_001.npy
    python visualize_clip.py --npy data/keypoints/clip_001.npy --classify --model lstm
    python visualize_clip.py --video my_pushup.mp4 --slow 0.25
    python visualize_clip.py --video my_pushup.mp4 --save
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

# All MediaPipe BlazePose connections (33 landmarks)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def draw_keypoints(frame, keypoints):
    """Draw full 33-landmark skeleton. keypoints: (33, 2)"""
    h, w = frame.shape[:2]
    pts = {}
    for i, (x, y) in enumerate(keypoints):
        pts[i] = (int(x * w), int(y * h))
        cv2.circle(frame, pts[i], 5, (0, 165, 255), -1)
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (255, 165, 0), 2)


def draw_joint_angles(frame, keypoints):
    """Overlay elbow and back alignment angles. keypoints: (33, 2)"""
    h, w = frame.shape[:2]

    def pt(idx):
        return keypoints[idx]  # (x, y)

    # Right elbow: shoulder(12) → elbow(14) → wrist(16)
    elbow_angle = compute_angle(pt(12), pt(14), pt(16))
    px = int(pt(14)[0] * w) + 12
    py = int(pt(14)[1] * h)
    ok = config.ELBOW_ANGLE_MIN <= elbow_angle <= config.ELBOW_ANGLE_MAX
    cv2.putText(frame, f"Elbow {elbow_angle:.0f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (46, 204, 113) if ok else (60, 76, 231), 2)

    # Back alignment: shoulder(12) → hip(24) → ankle(28)
    back_angle = compute_angle(pt(12), pt(24), pt(28))
    px2 = int(pt(24)[0] * w) + 12
    py2 = int(pt(24)[1] * h)
    ok2 = config.BACK_ALIGNMENT_MIN <= back_angle <= config.BACK_ALIGNMENT_MAX
    cv2.putText(frame, f"Back {back_angle:.0f}", (px2, py2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (46, 204, 113) if ok2 else (60, 76, 231), 2)


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
    """seq_raw: (T, 33, 2) → (label, confidence)"""
    import torch
    seq = pad_or_truncate(seq_raw)
    seq = normalize_keypoints(seq)
    seq = flatten_sequence(seq)
    x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(x), dim=-1).squeeze().cpu().numpy()
    label = int(probs.argmax())
    return label, float(probs[label])


# ── Mode 1: .npy file ─────────────────────────────────────────────────────────

def run_on_npy(npy_path: str, use_classifier: bool, model_name: str, slow: float):
    raw = np.load(npy_path, allow_pickle=True)

    # Support both (T, 66) flat and (T, 33, 2) shaped
    if raw.ndim == 2 and raw.shape[1] == config.INPUT_DIM:
        seq = raw.reshape(raw.shape[0], config.NUM_KEYPOINTS, config.KEYPOINT_DIM)
    else:
        seq = raw  # already (T, 33, 2)

    T = len(seq)
    W, H = 800, 600
    delay = max(1, int((1000 / config.TARGET_FPS) / slow))

    # Remap normalized coords into [0.1, 0.9] for display
    xy = seq[:, :, :2]
    xy_min, xy_max = xy.min(), xy.max()
    seq_display = seq.copy()
    seq_display[:, :, :2] = (xy - xy_min) / (xy_max - xy_min + 1e-6) * 0.8 + 0.1

    label_str, label_color, pred_conf = None, None, None
    if use_classifier:
        classifier, device = load_classifier(model_name)
        if classifier:
            label, conf = classify_sequence(seq, classifier, device)
            pred_conf   = conf
            label_str   = "Good Form" if label == 1 else "Poor Form"
            label_color = (46, 204, 113) if label == 1 else (60, 76, 231)
            print(f"[visualize] {label_str} ({conf:.1%})")

    print(f"[visualize] Playing {T} frames — Q to quit, SPACE to pause.")
    paused = False
    frame_idx = 0

    while True:
        if not paused:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)

            kps = seq_display[frame_idx]
            draw_keypoints(frame, kps)
            draw_joint_angles(frame, kps)

            cv2.putText(frame, f"Frame {frame_idx+1}/{T}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.putText(frame, Path(npy_path).stem, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            if label_str:
                cv2.putText(frame, f"{label_str}  {pred_conf:.0%}",
                            (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, label_color, 2)

            frame_idx = (frame_idx + 1) % T

        cv2.imshow("Pushup — Keypoint Viewer", frame)
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
                cv2.imshow("Pushup — Pose Estimation", frame)
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
    group.add_argument("--npy",   help="Path to keypoint .npy file")
    parser.add_argument("--classify", action="store_true", default=True)
    parser.add_argument("--model",    default="lstm")
    parser.add_argument("--save",     action="store_true", help="Save annotated video (video mode only)")
    parser.add_argument("--slow",     type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 0.25 = quarter speed)")
    args = parser.parse_args()

    if args.npy:
        run_on_npy(args.npy, args.classify, args.model, args.slow)
    else:
        run_on_video(args.video, args.classify, args.model, args.save, args.slow)
