"""
infer.py

Run a trained model on a single pushup video and print the result.

Usage:
    python infer.py --video my_pushup.mp4 --model cnn_lstm
    python infer.py --video my_pushup.mp4 --model lstm --plot
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


def infer(video_path, model_name="cnn_lstm", checkpoint_path=None, plot=False,
          annotate_video=False, device=None):

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if checkpoint_path is None:
        checkpoint_path = str(Path(config.CHECKPOINTS_DIR) / f"{model_name}_best.pt")

    # step 1: extract keypoints from the video
    print(f"[infer] extracting keypoints from {video_path}...")
    raw_seq = extract_keypoints_from_video(video_path)   # (T, 33, 2)

    # step 2: preprocess (same as training)
    seq = pad_or_truncate(raw_seq)        # (150, 33, 2)
    seq = normalize_keypoints(seq)
    seq = flatten_sequence(seq)           # (150, 66)

    # step 3: run through the model
    model = build_model(model_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)  # (1, 150, 66)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    label      = int(probs.argmax())
    confidence = float(probs[label])

    print("\n" + "=" * 50)
    print(f"  Prediction : {'Good Form' if label == 1 else 'Poor Form'}")
    print(f"  Confidence : {confidence:.1%}  (Good={probs[1]:.1%}  Poor={probs[0]:.1%})")

    # step 4: coaching feedback from rule-based angles
    feedback = generate_feedback(raw_seq)
    if feedback:
        print("\n  Coaching feedback:")
        for fb in feedback:
            print(f"    - {fb}")
    else:
        print("\n  Coaching feedback: all key angles look good.")

    # optional: save plots / annotated video
    if plot:
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        plot_joint_angles(raw_seq, label, confidence,
                          save_path=str(results_dir / f"{stem}_joint_angles.png"))

    if annotate_video:
        out_path = str(Path(video_path).with_suffix("")) + "_annotated.mp4"
        overlay_skeleton_on_video(video_path, raw_seq, out_path, score=confidence, label=label)

    return label, confidence, feedback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run pushup form analysis on a video")
    parser.add_argument("--video",      required=True)
    parser.add_argument("--model",      default="cnn_lstm", choices=["lstm", "transformer", "cnn_lstm", "mlp"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot",       action="store_true", help="save joint angle plot")
    parser.add_argument("--annotate",   action="store_true", help="write annotated output video")
    args = parser.parse_args()

    infer(
        video_path=args.video,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        plot=args.plot,
        annotate_video=args.annotate,
    )
