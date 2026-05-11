"""
pose_estimation.py

Wraps MediaPipe PoseLandmarker to extract 33 body keypoints per frame.
Each keypoint is just (x, y) normalized to [0, 1].

Model file needs to be downloaded separately (see README).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task")

# all 33 mediapipe pose connections for drawing the skeleton
# this code was AI generated
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]


# this code was AI generated
def _build_landmarker(model_path=DEFAULT_MODEL):
    """Set up the MediaPipe PoseLandmarker with the Tasks API."""
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
    """Context manager for running MediaPipe on frames."""

    def __init__(self, model_path=DEFAULT_MODEL):
        self.landmarker = _build_landmarker(model_path)

    def extract_keypoints(self, frame_bgr):
        """Run on a single BGR frame.

        Returns (33, 2) array of (x, y) coordinates, or None if no person found.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]
        keypoints = np.array([[lm[i].x, lm[i].y] for i in range(33)], dtype=np.float32)
        return keypoints

    def extract_keypoints_with_world(self, frame_bgr):
        """Same as extract_keypoints but also returns the raw landmark list for drawing."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None, None

        lm = result.pose_landmarks[0]
        keypoints = np.array([[lm[i].x, lm[i].y] for i in range(33)], dtype=np.float32)
        return keypoints, lm

    def draw_full_skeleton(self, frame, landmarks_33):
        """Draw the full 33-point skeleton on a frame."""
        frame = frame.copy()
        h, w = frame.shape[:2]

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


def extract_keypoints_from_video(video_path, output_path=None, interpolate_missing=True,
                                  model_path=DEFAULT_MODEL):
    """Extract keypoints for every frame in a video.

    Returns np.ndarray of shape (T, 33, 2).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"can't open video: {video_path}")

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
        raise ValueError(f"no frames extracted from {video_path}")

    if interpolate_missing and missing_frames:
        keypoints_seq = _interpolate_missing(keypoints_seq)

    # replace any remaining None with zeros
    placeholder = np.zeros((config.NUM_KEYPOINTS, config.KEYPOINT_DIM), dtype=np.float32)
    keypoints_seq = [kp if kp is not None else placeholder for kp in keypoints_seq]

    result = np.stack(keypoints_seq, axis=0)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)

    return result


def _interpolate_missing(seq):
    """Fill in None frames by linear interpolation between neighbors."""
    # this code was AI generated
    n = len(seq)
    for i in range(n):
        if seq[i] is None:
            prev_idx = next((j for j in range(i - 1, -1, -1) if seq[j] is not None), None)
            next_idx = next((j for j in range(i + 1, n)      if seq[j] is not None), None)
            if prev_idx is not None and next_idx is not None:
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                seq[i] = (1 - alpha) * seq[prev_idx] + alpha * seq[next_idx]
            elif prev_idx is not None:
                seq[i] = seq[prev_idx]
            elif next_idx is not None:
                seq[i] = seq[next_idx]
    return seq
