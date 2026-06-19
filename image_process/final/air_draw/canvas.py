"""Shared drawing canvas: stroke rendering + timestamped stroke log.

Strokes persist on a layer keyed by user_id and rendered in the owner's
color. The stroke log (user_id, point, t_ms) is what the evaluation/ablation
side of the proposal would consume for stroke-attribution accuracy.
"""
import time

import cv2
import numpy as np


class Canvas:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._last_point: dict[int, np.ndarray] = {}
        self.stroke_log: list[dict] = []

    def begin_stroke(self, user_id: int, point: np.ndarray, t_ms: float):
        self._last_point[user_id] = point.copy()
        self.stroke_log.append({"user_id": user_id, "event": "down",
                                 "x": float(point[0]), "y": float(point[1]), "t_ms": t_ms})

    def extend_stroke(self, user_id: int, point: np.ndarray, color: tuple, t_ms: float):
        prev = self._last_point.get(user_id)
        if prev is None:
            self.begin_stroke(user_id, point, t_ms)
            return
        if np.linalg.norm(point - prev) < 2:
            return
        cv2.line(self.layer, tuple(prev.astype(int)), tuple(point.astype(int)),
                 color, thickness=4, lineType=cv2.LINE_AA)
        self._last_point[user_id] = point.copy()
        self.stroke_log.append({"user_id": user_id, "event": "move",
                                 "x": float(point[0]), "y": float(point[1]), "t_ms": t_ms})

    def end_stroke(self, user_id: int, t_ms: float):
        if user_id in self._last_point:
            self.stroke_log.append({"user_id": user_id, "event": "up", "t_ms": t_ms,
                                     "x": float(self._last_point[user_id][0]),
                                     "y": float(self._last_point[user_id][1])})
            del self._last_point[user_id]

    def clear(self):
        self.layer[:] = 0
        self._last_point.clear()
        self.stroke_log.clear()

    def composite_onto(self, frame: np.ndarray) -> np.ndarray:
        mask = self.layer.any(axis=2)
        out = frame.copy()
        out[mask] = self.layer[mask]
        return out

    def now_ms(self) -> float:
        return time.time() * 1000.0
