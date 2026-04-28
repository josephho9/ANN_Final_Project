"""
Real-time pose estimation via webcam using MediaPipe Tasks API (v0.10+).

Shows skeleton overlaid on live camera feed.
Press Q to quit.

Usage:
    python webcam_demo.py
    python webcam_demo.py --camera 1        # external camera
    python webcam_demo.py --classify --model lstm   # live form classification
"""

import argparse
import time
import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import config
from src.pose_estimation import PoseEstimator


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def draw_joint_angles(frame, keypoints):
    h, w = frame.shape[:2]

    def pt(idx):
        x, y, vis = keypoints[idx]
        return (x, y) if vis > 0.4 else None

    # Right elbow: shoulder(2) → elbow(4) → wrist(6)
    if all(pt(i) is not None for i in [2, 4, 6]):
        angle = compute_angle(pt(2), pt(4), pt(6))
        ex = int(keypoints[4][0] * w) + 10
        ey = int(keypoints[4][1] * h)
        ok = config.ELBOW_ANGLE_MIN <= angle <= config.ELBOW_ANGLE_MAX
        cv2.putText(frame, f"Elbow {angle:.0f}°", (ex, ey),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (46, 204, 113) if ok else (60, 76, 231), 2)

    # Right knee: hip(8) → knee(10) → ankle(12)
    if all(pt(i) is not None for i in [8, 10, 12]):
        angle = compute_angle(pt(8), pt(10), pt(12))
        kx = int(keypoints[10][0] * w) + 10
        ky = int(keypoints[10][1] * h)
        ok = config.KNEE_BEND_MIN <= angle <= config.KNEE_BEND_MAX
        cv2.putText(frame, f"Knee {angle:.0f}°", (kx, ky),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (46, 204, 113) if ok else (60, 76, 231), 2)


def run_webcam(camera_idx: int = 0, use_classifier: bool = False, model_name: str = "lstm"):
    # ── Optional classifier ───────────────────────────────────────────────────
    classifier = None
    device = "cpu"
    if use_classifier:
        import torch
        from src.models import build_model
        ckpt_path = Path(config.CHECKPOINTS_DIR) / f"{model_name}_best.pt"
        if ckpt_path.exists():
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            classifier = build_model(model_name).to(device)
            ckpt = torch.load(str(ckpt_path), map_location=device)
            classifier.load_state_dict(ckpt["model_state"])
            classifier.eval()
            print(f"[webcam] Loaded {model_name} on {device}.")
        else:
            print(f"[webcam] No checkpoint found at {ckpt_path}, running pose-only.")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buffer = []
    pred_label = None
    pred_conf  = None
    fps_t = time.time()
    fps_count = 0
    fps = 0.0

    print("[webcam] Press Q to quit.")

    with PoseEstimator() as estimator:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            keypoints, landmarks_33 = estimator.extract_keypoints_with_world(frame)

            if keypoints is not None and landmarks_33 is not None:
                # Draw full 33-landmark skeleton
                frame = estimator.draw_full_skeleton(frame, landmarks_33)
                draw_joint_angles(frame, keypoints)

                # Classifier buffer
                if classifier is not None:
                    import torch
                    from src.preprocessing import normalize_keypoints, flatten_sequence, pad_or_truncate
                    buffer.append(keypoints)
                    if len(buffer) > config.SEQUENCE_LEN:
                        buffer.pop(0)
                    if len(buffer) == config.SEQUENCE_LEN:
                        seq = np.stack(buffer)
                        seq = normalize_keypoints(seq)
                        seq = flatten_sequence(seq)
                        x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            probs = torch.softmax(classifier(x), dim=-1).squeeze().cpu().numpy()
                        pred_label = int(probs.argmax())
                        pred_conf  = float(probs[pred_label])
            else:
                cv2.putText(frame, "No pose detected", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_t = time.time()
                fps_count = 0

            cv2.putText(frame, f"FPS {fps:.0f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            if pred_label is not None:
                label_str = "Good Form" if pred_label == 1 else "Poor Form"
                color = (46, 204, 113) if pred_label == 1 else (60, 76, 231)
                cv2.putText(frame, f"{label_str} {pred_conf:.0%}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow("Basketball Jumpshot Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--model",    type=str, default="lstm")
    parser.add_argument("--classify", action="store_true")
    args = parser.parse_args()
    run_webcam(args.camera, args.classify, args.model)
