import cv2
import numpy as np

WAITING_COLOR = (180, 180, 180)


def draw_hand_overlay(frame, bbox, color, label, pen_down):
    x1, y1, x2, y2 = bbox.astype(int)
    thickness = 3 if pen_down else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2, cv2.LINE_AA)


def draw_legend(frame, user_colors: dict, pen_state: dict):
    x0, y0 = 10, 10
    row_h = 26
    for i, (user_id, color) in enumerate(sorted(user_colors.items())):
        y = y0 + i * row_h
        cv2.rectangle(frame, (x0, y), (x0 + 18, y + 18), color, -1)
        status = "drawing" if pen_state.get(user_id) else "idle"
        cv2.putText(frame, f"User {user_id} ({status})", (x0 + 26, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def draw_waiting_hint(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), WAITING_COLOR, 1)
    cv2.putText(frame, "wave to join", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                WAITING_COLOR, 1, cv2.LINE_AA)
