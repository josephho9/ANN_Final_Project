"""
Download basketball jump shot clips for the dataset.

Sources:
  1. YouTube search via yt-dlp (free, open footage)
  2. Penn Action Dataset (academic, requires manual download — see instructions below)

Requirements:
    pip install yt-dlp
    brew install ffmpeg   # for trimming

Usage:
    python scripts/download_videos.py --source youtube --n 50
    python scripts/download_videos.py --source youtube --query "NBA jump shot slow motion" --n 20
"""

import argparse
import subprocess
import json
import shutil
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

RAW_DIR = Path(config.RAW_VIDEO_DIR)

# ── YouTube search queries for good/varied jumpshot footage ──────────────────
GOOD_FORM_QUERIES = [
    "Stephen Curry jump shot slow motion breakdown",
    "Klay Thompson shooting form analysis",
    "Ray Allen jump shot mechanics",
    "Kevin Durant jump shot form slow motion",
    "Dirk Nowitzki one leg fadeaway slow motion",
    "Damian Lillard pull up jumper slow motion",
    "Jayson Tatum jump shot mechanics breakdown",
    "Devin Booker shooting form slow motion",
    "proper basketball shooting form tutorial beginner",
    "basketball shot mechanics elbow alignment tutorial",
    "how to shoot a basketball perfectly slow motion",
    "NBA player shooting form comparison slow motion",
    "perfect jump shot release point slow motion",
    "basketball shooting fundamentals form tutorial",
    "Steph Curry shooting form breakdown 2024",
    "NBA shooting form side angle slow motion",
    "college basketball jump shot slow motion",
    "basketball shooting mechanics wrist follow through",
    "how to fix jump shot form tutorial",
    "NBA 3 point shot slow motion release",
    "Trae Young jump shot slow motion",
    "Luka Doncic step back jumper slow motion",
    "basketball free throw form slow motion tutorial",
    "Paul George jump shot mechanics breakdown",
    "Zach LaVine jump shot slow motion",
]

POOR_FORM_QUERIES = [
    "basketball shooting form mistakes corrections coach",
    "bad jump shot form how to fix",
    "common basketball shooting errors beginner",
    "elbow flare basketball shot correction",
    "basketball shot too flat trajectory fix",
    "off balance jump shot mistakes basketball",
    "wrong shooting form basketball tutorial fix",
    "basketball shooting problems guide coach",
    "how not to shoot a basketball form mistakes",
    "beginner basketball shooting form errors",
    "basketball shot mechanics problems fix guide",
    "why is my jump shot off coaching tips",
    "basketball shooting form check mistakes",
    "sloppy jump shot mechanics fix basketball",
    "youth basketball bad shooting form correction",
    "basketball shooting inconsistency form errors",
    "jump shot arc too low fix basketball",
    "shooting off the wrong foot basketball mistake",
    "basketball shot power issues form fix",
    "grip and release problems basketball shot",
]


def check_ytdlp():
    if not shutil.which("yt-dlp"):
        print("[ERROR] yt-dlp not found. Install with: pip install yt-dlp")
        return False
    return True


def download_youtube_clip(query: str, output_dir: Path, clip_id: str, max_duration: int = 120):
    """
    Search YouTube and download the top result matching `query`.
    Clips longer than max_duration seconds are trimmed to first 30s.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / f"{clip_id}.%(ext)s")

    import random
    pick = random.randint(1, 3)   # pick 1st, 2nd, or 3rd result randomly
    cmd = [
        "yt-dlp",
        f"ytsearch{pick}:{query}",
        "--playlist-items", str(pick),  # only download the picked result
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--no-playlist",
        "--match-filter", f"duration < {max_duration}",
        "--quiet",
        "--no-warnings",
    ]

    print(f"  Searching: {query[:60]}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] Failed: {result.stderr.strip()[:120]}")
        return None

    # Find the downloaded file
    matches = list(output_dir.glob(f"{clip_id}.*"))
    return str(matches[0]) if matches else None


def trim_clip(input_path: str, output_path: str, duration: int = 10):
    """Trim to first `duration` seconds using ffmpeg."""
    if not shutil.which("ffmpeg"):
        return input_path
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        output_path,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    Path(input_path).unlink(missing_ok=True)
    return output_path


def download_dataset(
    n_good: int = 25,
    n_poor: int = 25,
    extra_query: str = None,
    trim_to: int = 15,
):
    if not check_ytdlp():
        return

    labels = []

    # Find the highest existing clip number to continue from
    existing_good = list((RAW_DIR / "good").glob("good_*.mp4")) if (RAW_DIR / "good").exists() else []
    existing_poor = list((RAW_DIR / "poor").glob("poor_*.mp4")) if (RAW_DIR / "poor").exists() else []
    good_start = max((int(f.stem.split("_")[1]) for f in existing_good), default=0)
    poor_start = max((int(f.stem.split("_")[1]) for f in existing_poor), default=0)

    print(f"\n[download] Downloading {n_good} good-form clips (continuing from good_{good_start:03d}) …")
    for i in range(n_good):
        query = GOOD_FORM_QUERIES[i % len(GOOD_FORM_QUERIES)]
        if extra_query:
            query = extra_query + " " + query
        clip_id = f"good_{good_start + i + 1:03d}"
        raw_dir = RAW_DIR / "good"
        path = download_youtube_clip(query, raw_dir, clip_id)
        if path:
            trimmed = str(raw_dir / f"{clip_id}.mp4")
            trim_clip(path, trimmed, duration=trim_to)
            labels.append({"clip_id": clip_id, "label": 1})
            print(f"  ✓ {clip_id}")

    print(f"\n[download] Downloading {n_poor} poor-form clips (continuing from poor_{poor_start:03d}) …")
    for i in range(n_poor):
        query = POOR_FORM_QUERIES[i % len(POOR_FORM_QUERIES)]
        clip_id = f"poor_{poor_start + i + 1:03d}"
        raw_dir = RAW_DIR / "poor"
        path = download_youtube_clip(query, raw_dir, clip_id)
        if path:
            trimmed = str(raw_dir / f"{clip_id}.mp4")
            trim_clip(path, trimmed, duration=trim_to)
            labels.append({"clip_id": clip_id, "label": 0})
            print(f"  ✓ {clip_id}")

    # Write labels CSV
    labels_path = Path(config.LABELS_FILE)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        f.write("clip_id,label\n")
        for row in labels:
            f.write(f"{row['clip_id']},{row['label']}\n")

    print(f"\n[download] Done. {len(labels)} clips. Labels → {labels_path}")
    print("\nNext step: python scripts/extract_all_keypoints.py")


# ── Penn Action Dataset instructions ─────────────────────────────────────────
PENN_ACTION_INSTRUCTIONS = """
Penn Action Dataset (academic, recommended for labeled clips)
─────────────────────────────────────────────────────────────
1. Request access at: http://dreamdragon.github.io/PennAction/
2. Download PennAction.tar.gz (~2 GB)
3. Extract to data/penn_action/
4. Run: python scripts/prepare_penn_action.py
   (filters to 'baseball_pitch' and 'jumping_jacks' as proxy for arm/leg motion;
    use your own basketball clips for the actual jumpshot class)
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["youtube", "penn"], default="youtube")
    parser.add_argument("--n", type=int, default=50, help="Total clips (split 50/50 good/poor)")
    parser.add_argument("--query", type=str, default=None, help="Extra search term prefix")
    parser.add_argument("--trim", type=int, default=15, help="Trim clips to N seconds")
    args = parser.parse_args()

    if args.source == "penn":
        print(PENN_ACTION_INSTRUCTIONS)
    else:
        n_each = args.n // 2
        download_dataset(n_good=n_each, n_poor=n_each, extra_query=args.query, trim_to=args.trim)
