"""
Real-time pushup form analysis via webcam using MediaPipe Tasks API (v0.10+).

Shows skeleton overlaid on live camera feed with elbow and back alignment angles.
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
    """keypoints: (33, 2) — full MediaPipe landmarks, x/y only."""
    h, w = frame.shape[:2]

    def pt(idx):
        return keypoints[idx]  # (x, y)

    # Right elbow: shoulder(12) → elbow(14) → wrist(16)
    elbow = compute_angle(pt(12), pt(14), pt(16))
    ex = int(pt(14)[0] * w) + 10
    ey = int(pt(14)[1] * h)
    ok = config.ELBOW_ANGLE_MIN <= elbow <= config.ELBOW_ANGLE_MAX
    cv2.putText(frame, f"Elbow {elbow:.0f}", (ex, ey),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (46, 204, 113) if ok else (60, 76, 231), 2)

    # Back alignment: shoulder(12) → hip(24) → ankle(28)
    back = compute_angle(pt(12), pt(24), pt(28))
    bx = int(pt(24)[0] * w) + 10
    by = int(pt(24)[1] * h)
    ok2 = config.BACK_ALIGNMENT_MIN <= back <= config.BACK_ALIGNMENT_MAX
    cv2.putText(frame, f"Back {back:.0f}", (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (46, 204, 113) if ok2 else (60, 76, 231), 2)


def run_webcam(camera_idx: int = 0, use_classifier: bool = False, model_name: str = "lstm"):
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
            print(f"[webcam] No checkpoint at {ckpt_path}, running pose-only.")

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
                frame = estimator.draw_full_skeleton(frame, landmarks_33)
                draw_joint_angles(frame, keypoints)

                if classifier is not None:
                    import torch
                    from src.preprocessing import normalize_keypoints, flatten_sequence, pad_or_truncate
                    buffer.append(keypoints)
                    if len(buffer) > config.SEQUENCE_LEN:
                        buffer.pop(0)
                    if len(buffer) == config.SEQUENCE_LEN:
                        seq = np.stack(buffer)              # (T, 33, 2)
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

            cv2.imshow("Pushup Form Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time pushup form analysis via webcam")
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--model",    type=str, default="lstm")
    parser.add_argument("--classify", action="store_true")
    args = parser.parse_args()
    run_webcam(args.camera, args.classify, args.model)
