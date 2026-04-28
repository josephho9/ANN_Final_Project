"""
Batch-extract MediaPipe keypoints from all downloaded video clips.

Usage:
    python scripts/extract_all_keypoints.py
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.pose_estimation import extract_keypoints_from_video

RAW_DIR = Path(config.RAW_VIDEO_DIR)
KP_DIR  = Path(config.KEYPOINTS_DIR)
KP_DIR.mkdir(parents=True, exist_ok=True)

video_files = list(RAW_DIR.rglob("*.mp4")) + list(RAW_DIR.rglob("*.mov"))
print(f"[extract] Found {len(video_files)} videos.")

for i, vpath in enumerate(video_files, 1):
    clip_id = vpath.stem
    out = KP_DIR / f"{clip_id}.npy"
    if out.exists():
        print(f"  [{i}/{len(video_files)}] Skip (exists): {clip_id}")
        continue
    print(f"  [{i}/{len(video_files)}] Processing: {clip_id}")
    try:
        extract_keypoints_from_video(str(vpath), output_path=str(out))
    except Exception as e:
        print(f"  [WARN] {clip_id}: {e}")

print(f"\n[extract] Done. Keypoints saved to {KP_DIR}/")
print("Next step: python -m src.train --model lstm")
