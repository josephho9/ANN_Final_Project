"""
Stage 03 – Pose Estimation
Extracts 17 body keypoints per frame using MediaPipe PoseLandmarker (Tasks API, v0.10+).

Model file: models/pose_landmarker_lite.task
Download:   python -c "import urllib.request; urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                'models/pose_landmarker_lite.task')"
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task")

# ── Drawing helpers (Tasks API does not include drawing_utils) ────────────────
CONNECTIONS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
    (1, 7), (2, 8),
    (7, 8),
    (7, 9), (8, 10),
    (9, 11), (10, 12),
]


def _build_landmarker(model_path: str = DEFAULT_MODEL):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker (Tasks API) for per-frame keypoint extraction."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.landmarker = _build_landmarker(model_path)

    def extract_keypoints(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run pose landmarker on a single BGR frame.

        Returns:
            np.ndarray (NUM_KEYPOINTS, 3) — (x, y, visibility), normalized [0,1].
            Returns None if no pose detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]   # first (only) person
        keypoints = np.array(
            [[lm[i].x, lm[i].y, lm[i].visibility] for i in config.LANDMARK_INDICES],
            dtype=np.float32,
        )
        return keypoints

    def extract_keypoints_with_world(self, frame_bgr: np.ndarray):
        """Also return raw landmark list for drawing (33 landmarks)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None, None

        lm = result.pose_landmarks[0]
        keypoints = np.array(
            [[lm[i].x, lm[i].y, lm[i].visibility] for i in config.LANDMARK_INDICES],
            dtype=np.float32,
        )
        return keypoints, lm   # (17,3), full 33-landmark list

    def draw_pose(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        h, w = frame.shape[:2]
        pts = {}
        for i, (x, y, vis) in enumerate(keypoints):
            if vis > 0.4:
                pts[i] = (int(x * w), int(y * h))
                cv2.circle(frame, pts[i], 6, (0, 165, 255), -1)
        for a, b in CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (255, 165, 0), 2)
        return frame

    def draw_full_skeleton(self, frame: np.ndarray, landmarks_33) -> np.ndarray:
        """Draw the full 33-landmark skeleton using the raw landmark list."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        # All MediaPipe pose connections (33 landmarks)
        POSE_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
            (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
            (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
            (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
            (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
        ]

        pts = {}
        for i, lm in enumerate(landmarks_33):
            if lm.visibility > 0.4:
                pts[i] = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pts[i], 4, (0, 165, 255), -1)

        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (255, 165, 0), 2)

        return frame

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def extract_keypoints_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    interpolate_missing: bool = True,
    model_path: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Extract keypoints for every frame in a video clip.

    Returns np.ndarray of shape (T, NUM_KEYPOINTS, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    keypoints_seq = []
    missing_frames = []

    with PoseEstimator(model_path) as estimator:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, config.FRAME_SIZE)
            kp = estimator.extract_keypoints(frame)
            if kp is None:
                missing_frames.append(frame_idx)
                keypoints_seq.append(None)
            else:
                keypoints_seq.append(kp)
            frame_idx += 1

    cap.release()

    if not keypoints_seq:
        raise ValueError(f"No frames extracted from {video_path}")

    if interpolate_missing and missing_frames:
        keypoints_seq = _interpolate_missing(keypoints_seq)

    placeholder = np.zeros((config.NUM_KEYPOINTS, config.KEYPOINT_DIM), dtype=np.float32)
    keypoints_seq = [kp if kp is not None else placeholder for kp in keypoints_seq]

    result = np.stack(keypoints_seq, axis=0)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)

    return result


def _interpolate_missing(seq: list) -> list:
    n = len(seq)
    for i in range(n):
        if seq[i] is None:
            prev_idx = next((j for j in range(i - 1, -1, -1) if seq[j] is not None), None)
            next_idx = next((j for j in range(i + 1, n) if seq[j] is not None), None)
            if prev_idx is not None and next_idx is not None:
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                seq[i] = (1 - alpha) * seq[prev_idx] + alpha * seq[next_idx]
            elif prev_idx is not None:
                seq[i] = seq[prev_idx]
            elif next_idx is not None:
                seq[i] = seq[next_idx]
    return seq
