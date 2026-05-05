"""
Inference script — run a trained model on a new pushup video clip.

Usage:
    python infer.py --video my_pushup.mp4 --model lstm
    python infer.py --video my_pushup.mp4 --model transformer --plot
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import config
from src.models import build_model
from src.pose_estimation import extract_keypoints_from_video
from src.preprocessing import pad_or_truncate, normalize_keypoints, flatten_sequence
from src.visualize import plot_joint_angles, generate_feedback, overlay_skeleton_on_video


def infer(
    video_path: str,
    model_name: str = "lstm",
    checkpoint_path: str = None,
    plot: bool = False,
    annotate_video: bool = False,
    device: str = None,
):
    device = device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    if checkpoint_path is None:
        checkpoint_path = str(Path(config.CHECKPOINTS_DIR) / f"{model_name}_best.pt")

    # ── 1. Extract keypoints ──────────────────────────────────────────────────
    print(f"[infer] Extracting keypoints from {video_path} …")
    raw_seq = extract_keypoints_from_video(video_path)   # (T, 33, 2)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    seq_norm = pad_or_truncate(raw_seq)                   # (T_target, 33, 2)
    seq_norm = normalize_keypoints(seq_norm)
    seq_flat = flatten_sequence(seq_norm)                 # (T_target, 66)

    # ── 3. Model inference ────────────────────────────────────────────────────
    model = build_model(model_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.tensor(seq_flat[np.newaxis], dtype=torch.float32).to(device)  # (1, T, 66)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    label = int(probs.argmax())
    confidence = float(probs[label])

    print("\n" + "=" * 50)
    print(f"  Prediction : {'Good Form' if label == 1 else 'Poor Form'}")
    print(f"  Confidence : {confidence:.1%}  (Good={probs[1]:.1%}  Poor={probs[0]:.1%})")

    # ── 4. Actionable feedback ────────────────────────────────────────────────
    feedback = generate_feedback(raw_seq)
    if feedback:
        print("\n  Coaching feedback:")
        for fb in feedback:
            print(f"    • {fb}")
    else:
        print("\n  Coaching feedback: All key joint angles within optimal range.")

    # ── 5. Optional visualizations ────────────────────────────────────────────
    if plot:
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        plot_joint_angles(
            raw_seq, label, confidence,
            save_path=str(results_dir / f"{stem}_joint_angles.png"),
        )

    if annotate_video:
        out_path = str(Path(video_path).with_suffix("") ) + "_annotated.mp4"
        overlay_skeleton_on_video(
            video_path, raw_seq, out_path,
            score=confidence, label=label,
        )

    return label, confidence, feedback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pushup form analysis on a video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer", "cnn_lstm", "mlp"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot", action="store_true", help="Save joint angle plot")
    parser.add_argument("--annotate", action="store_true", help="Write annotated output video")
    args = parser.parse_args()

    infer(
        video_path=args.video,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        plot=args.plot,
        annotate_video=args.annotate,
    )
