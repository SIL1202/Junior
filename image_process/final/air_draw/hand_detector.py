"""Hand detection wrapper around MediaPipe's HandLandmarker task.

MediaPipe's legacy `solutions.hands` API (used in the original proposal write-up)
has been removed from current MediaPipe releases in favor of the Tasks API. We
use `HandLandmarker` here, configured for up to MAX_HANDS simultaneous hands —
this replaces the proposal's "fine-tune YOLOv8-hand for 6 hands" step with an
off-the-shelf detector that already supports >2 hands, avoiding the need to
collect/train on EgoHands + 11k Hands.
"""
from dataclasses import dataclass

import mediapipe as mp
import numpy as np

from . import config

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class Detection:
    bbox: np.ndarray       # (4,) float [x1, y1, x2, y2] in pixel coords
    landmarks: np.ndarray  # (21, 2) float pixel coords
    score: float
    handedness: str        # "Left" or "Right" (camera-mirrored)


class HandDetector:
    def __init__(self, max_hands: int = config.MAX_HANDS):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[Detection]:
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks], dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            pad = 0.15 * max(x2 - x1, y2 - y1, 1.0)
            bbox = np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad], dtype=np.float32)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h - 1)
            score = float(handedness[0].score) if handedness else 1.0
            label = handedness[0].category_name if handedness else "Unknown"
            detections.append(Detection(bbox=bbox, landmarks=pts, score=score, handedness=label))
        return detections

    def close(self):
        self._landmarker.close()
