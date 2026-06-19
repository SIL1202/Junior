"""Pinch (thumb-index) pen-down/up gesture, proposal section 2d.

Distance between landmarks 4 (thumb tip) and 8 (index fingertip), normalized
by a scale-invariant hand-size reference (wrist-to-middle-MCP distance, i.e.
landmarks 0 and 9), drives pen state. A hysteresis band (close below
PINCH_CLOSE_RATIO, open above PINCH_OPEN_RATIO) avoids flicker right at the
threshold instead of a single cutoff.
"""
import numpy as np

from . import config

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9


def pinch_ratio(landmarks: np.ndarray) -> float:
    scale = np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])
    if scale < 1e-3:
        return 1.0
    dist = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])
    return float(dist / scale)


def cursor_point(landmarks: np.ndarray) -> np.ndarray:
    return landmarks[INDEX_TIP]


class PinchDetector:
    def __init__(self):
        self._pen_down: dict[int, bool] = {}

    def update(self, user_id: int, landmarks: np.ndarray) -> bool:
        ratio = pinch_ratio(landmarks)
        down = self._pen_down.get(user_id, False)
        if down and ratio > config.PINCH_OPEN_RATIO:
            down = False
        elif not down and ratio < config.PINCH_CLOSE_RATIO:
            down = True
        self._pen_down[user_id] = down
        return down

    def reset(self, user_id: int):
        self._pen_down.pop(user_id, None)
