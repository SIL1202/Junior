"""Real-time multi-user air-drawing demo (deliverable #1 of the proposal).

Run: python -m air_draw.main [--camera 0]

Controls:
  q       quit
  c       clear the shared canvas
  s       save a snapshot of canvas + stroke log to ./session_<timestamp>/

How it works end to end, frame by frame:
  webcam frame -> HandDetector (MediaPipe HandLandmarker, up to MAX_HANDS hands)
               -> HandTracker (Kalman+IoU two-stage assoc. + appearance re-ID)
               -> CalibrationManager (wave gesture binds track_id -> user_id)
               -> PinchDetector (thumb-index pinch -> pen up/down)
               -> Canvas (renders the owner-colored stroke, logs timestamps)
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from . import config, gui
from .calibration import CalibrationManager
from .canvas import Canvas
from .hand_detector import HandDetector
from .pinch import PinchDetector, cursor_point
from .tracker import HandTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    detector = HandDetector()
    tracker = HandTracker()
    calibration = CalibrationManager()
    pinch_detector = PinchDetector()
    canvas = None

    t_start = time.perf_counter()
    fps_smoothed = 0.0
    prev_loop_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural "air drawing" interaction
            h, w = frame.shape[:2]
            if canvas is None:
                canvas = Canvas(w, h)

            timestamp_ms = int((time.perf_counter() - t_start) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = detector.detect(frame_rgb, timestamp_ms)
            confirmed = tracker.update(detections, frame, timestamp_s=timestamp_ms / 1000.0)
            bound = calibration.update(confirmed)

            t_ms = canvas.now_ms()
            pen_state = {}
            for track in confirmed:
                user_id = bound.get(track.track_id)
                if user_id is None:
                    gui.draw_waiting_hint(frame, track.bbox)
                    continue

                color = calibration.user_color(user_id)
                cursor = cursor_point(track.landmarks)
                pen_down = pinch_detector.update(user_id, track.landmarks)
                pen_state[user_id] = pen_down

                if pen_down:
                    canvas.extend_stroke(user_id, cursor, color, t_ms)
                else:
                    canvas.end_stroke(user_id, t_ms)

                gui.draw_hand_overlay(frame, track.bbox, color,
                                       f"U{user_id} #{track.track_id}", pen_down)

            frame = canvas.composite_onto(frame)
            gui.draw_legend(frame, calibration.user_colors, pen_state)

            now = time.perf_counter()
            dt = now - prev_loop_t
            prev_loop_t = now
            if dt > 0:
                fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt) if fps_smoothed else 1.0 / dt
            gui.draw_fps(frame, fps_smoothed)

            cv2.imshow("Multi-User Air Drawing", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                canvas.clear()
            elif key == ord("s"):
                _save_session(canvas)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def _save_session(canvas: Canvas):
    out_dir = f"session_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "canvas.png"), canvas.layer)
    with open(os.path.join(out_dir, "stroke_log.json"), "w") as f:
        json.dump(canvas.stroke_log, f, indent=2)
    print(f"Saved session to {out_dir}/")


if __name__ == "__main__":
    main()
