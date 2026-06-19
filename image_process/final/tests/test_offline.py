"""Camera-free smoke tests for every air_draw module.

No webcam is available in the dev/CI environment these were written in, so
these tests exercise each stage of the pipeline with synthetic data and one
real photo (test_assets/sample_hands.jpg, a MediaPipe sample asset) instead.
Run with: python -m tests.test_offline
"""
import math
import os

import cv2
import numpy as np
from types import SimpleNamespace

from air_draw import config, reid
from air_draw.calibration import CalibrationManager
from air_draw.canvas import Canvas
from air_draw.hand_detector import Detection, HandDetector
from air_draw.one_euro_filter import OneEuroFilter
from air_draw.pinch import PinchDetector, cursor_point, pinch_ratio
from air_draw.tracker import HandTracker

ASSETS = os.path.join(os.path.dirname(__file__), "..", "test_assets")


def test_one_euro_filter_reduces_jitter():
    rng = np.random.default_rng(0)
    n, freq = 90, 30.0
    t = np.arange(n) / freq
    clean = 300 + 150 * t
    noisy = clean + rng.normal(0, 2.0, n)

    f = OneEuroFilter(shape=(), freq=freq, mincutoff=config.ONE_EURO_MINCUTOFF,
                       beta=config.ONE_EURO_BETA)
    out = np.array([f(noisy[i], timestamp_s=t[i]) for i in range(n)])

    raw_jitter = np.std(np.diff(noisy))
    filt_jitter = np.std(np.diff(out))
    lag_err = np.sqrt(np.mean((out[10:] - clean[10:]) ** 2))
    assert filt_jitter < raw_jitter * 0.5, "filter should cut frame-to-frame jitter by >2x"
    assert lag_err < 10, "filter should still track the underlying motion closely"
    print(f"  jitter std {raw_jitter:.2f}px -> {filt_jitter:.2f}px, tracking err {lag_err:.2f}px")


def test_reid_embedding_distinguishes_appearance():
    rng = np.random.default_rng(0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def patch(base_bgr, region, noise=12):
        y1, x1, y2, x2 = region
        block = np.clip(np.array(base_bgr) + rng.normal(0, noise, (y2 - y1, x2 - x1, 3)),
                         0, 255).astype(np.uint8)
        frame[y1:y2, x1:x2] = block

    patch((120, 150, 200), (10, 10, 90, 90))
    patch((120, 150, 200), (110, 10, 190, 90))
    patch((200, 60, 30), (10, 110, 90, 190))

    e_a1 = reid.extract_embedding(frame, np.array([10, 10, 90, 90]))
    e_a2 = reid.extract_embedding(frame, np.array([10, 110, 90, 190]))
    e_b = reid.extract_embedding(frame, np.array([110, 10, 190, 90]))

    sim_same = reid.cosine_similarity(e_a1, e_a2)
    sim_diff = reid.cosine_similarity(e_a1, e_b)
    assert sim_same > sim_diff
    print(f"  cosine(same hand) = {sim_same:.3f} > cosine(different hand) = {sim_diff:.3f}")


def _make_frame_with_hand(rng, cx, cy, size=80, color=(120, 150, 200)):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    half = size // 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    block = np.clip(np.array(color) + rng.normal(0, 10, (size, size, 3)), 0, 255).astype(np.uint8)
    frame[y1:y2, x1:x2] = block
    return frame, np.array([x1, y1, x2, y2], dtype=np.float32)


def _det_for(bbox, score=0.9):
    landmarks = np.tile((bbox[:2] + bbox[2:]) / 2, (21, 1)).astype(np.float32)
    return Detection(bbox=bbox.astype(np.float32), landmarks=landmarks, score=score,
                      handedness="Right")


def test_tracker_keeps_stable_id_through_motion_occlusion_and_reentry():
    rng = np.random.default_rng(1)
    tracker = HandTracker()

    cx = 200
    ids_seen = []
    for i in range(6):
        cx += 5
        frame, bbox = _make_frame_with_hand(rng, cx, 200)
        out = tracker.update([_det_for(bbox)], frame, timestamp_s=i / 30)
        ids_seen.append([t.track_id for t in out])
    confirmed_ids = {tid for frame_ids in ids_seen[2:] for tid in frame_ids}
    assert len(confirmed_ids) == 1
    stable_id = confirmed_ids.pop()

    for i in range(6, 9):
        cx += 5
        tracker.update([], np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=i / 30)
    cx += 5
    frame, bbox = _make_frame_with_hand(rng, cx, 200)
    out = tracker.update([_det_for(bbox)], frame, timestamp_s=9 / 30)
    assert any(t.track_id == stable_id for t in out), "id should survive short occlusion"

    for i in range(10, 50):
        tracker.update([], np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=i / 30)
    frame, bbox = _make_frame_with_hand(rng, 450, 350)
    out = tracker.update([_det_for(bbox)], frame, timestamp_s=50 / 30)
    assert any(t.track_id == stable_id for t in out), "id should be revived via re-ID after re-entry"
    print(f"  track {stable_id} survived motion, short occlusion, and long-occlusion re-entry")


def test_tracker_handles_fast_oscillation_without_drift():
    """Regression test: a fast wave (large per-frame displacement) used to make IoU
    drop to ~0 every frame, causing the track to be 'lost' and its box to drift away
    on stale Kalman velocity while a new tentative track spawned at the real hand
    position. Fixed by a center-distance fallback association stage gated on each
    track's last *observed* box rather than its velocity-extrapolated one."""
    rng = np.random.default_rng(5)
    tracker = HandTracker()
    cx0, amp = 320, 150
    ids_used = set()
    max_err = 0.0
    for i in range(60):
        cx = int(cx0 + amp * math.sin(i * 0.8))
        frame, bbox = _make_frame_with_hand(rng, cx, 240)
        out = tracker.update([_det_for(bbox)], frame, timestamp_s=i / 30)
        ids_used.update(t.track_id for t in out)
        for t in out:
            tracked_cx = (t.bbox[0] + t.bbox[2]) / 2
            max_err = max(max_err, abs(tracked_cx - cx))
    assert len(ids_used) == 1, f"expected one stable id through a fast wave, got {ids_used}"
    assert max_err < 20, f"tracked box drifted {max_err:.1f}px from the real hand position"
    print(f"  fast wave (60 frames, up to 120px/frame): 1 stable id, max tracking error {max_err:.1f}px")


def test_tracker_does_not_swap_ids_when_hands_cross():
    rng = np.random.default_rng(2)
    tracker = HandTracker()
    color_a, color_b = (200, 60, 30), (30, 180, 60)
    half, y = 40, 240
    x_a, x_b = 150, 450

    def frame_for(box_a, box_b):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for box, color in ((box_a, color_a), (box_b, color_b)):
            x1, y1, x2, y2 = box.astype(int)
            block = np.clip(np.array(color) + rng.normal(0, 8, (y2 - y1, x2 - x1, 3)),
                             0, 255).astype(np.uint8)
            frame[y1:y2, x1:x2] = block
        return frame

    id_a = id_b = None
    for i in range(40):
        x_a += 8
        x_b -= 8
        box_a = np.array([x_a - half, y - half, x_a + half, y + half], dtype=np.float32)
        box_b = np.array([x_b - half, y - half, x_b + half, y + half], dtype=np.float32)
        frame = frame_for(box_a, box_b)
        out = tracker.update([_det_for(box_a), _det_for(box_b)], frame, timestamp_s=i / 30)
        by_x = sorted(out, key=lambda t: (t.bbox[0] + t.bbox[2]) / 2)
        if i == 10 and len(by_x) == 2:
            id_a, id_b = by_x[0].track_id, by_x[1].track_id
        if i == 39 and len(by_x) == 2:
            assert by_x[1].track_id == id_a, "track starting on the left should end on the right"
            assert by_x[0].track_id == id_b, "track starting on the right should end on the left"
    print("  no ID swap detected across a head-on crossing")


def test_calibration_wave_detection():
    cm = CalibrationManager()

    def make_track(tid, cx, w=80, h=80, y=200):
        bbox = np.array([cx - w / 2, y - h / 2, cx + w / 2, y + h / 2], dtype=np.float32)
        return SimpleNamespace(track_id=tid, bbox=bbox)

    bound = {}
    for i in range(config.WAVE_WINDOW_FRAMES + 5):
        cx = 300 + 60 * math.sin(i * 0.9)
        bound = cm.update([make_track(1, cx)])
    assert 1 in bound, "waving track should be calibrated"

    bound2 = {}
    for i in range(config.WAVE_WINDOW_FRAMES + 5):
        cx = 100 + 5 * i
        bound2 = cm.update([make_track(2, cx)])
    assert 2 not in bound2, "straight-line motion should not trigger calibration"
    print("  wave gesture binds a user; straight-line drawing motion does not false-trigger")


def test_pinch_and_canvas_drawing():
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[0] = [300, 400]
    lm[9] = [300, 300]
    lm[4] = [250, 250]
    lm[8] = [350, 250]

    pd = PinchDetector()
    assert pd.update(1, lm) is False

    lm[4] = [295, 250]
    lm[8] = [305, 250]
    assert pd.update(1, lm) is True

    canvas = Canvas(640, 480)
    canvas.begin_stroke(1, cursor_point(lm), t_ms=0)
    lm[8] = [320, 260]
    canvas.extend_stroke(1, cursor_point(lm), color=(255, 0, 0), t_ms=33)
    canvas.end_stroke(1, t_ms=66)
    assert canvas.layer.any()
    assert canvas.stroke_log[0]["event"] == "down"
    assert canvas.stroke_log[-1]["event"] == "up"
    print(f"  pinch ratio open={pinch_ratio(lm):.2f}, {len(canvas.stroke_log)} stroke events logged")


def test_real_hand_detection_on_sample_photo():
    img_path = os.path.join(ASSETS, "sample_hands.jpg")
    img = cv2.imread(img_path)
    assert img is not None, f"missing test asset {img_path}"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = HandDetector()
    try:
        dets = detector.detect(rgb, 0)
    finally:
        detector.close()
    assert len(dets) == 2, f"expected 2 hands in sample photo, got {len(dets)}"
    for d in dets:
        assert d.score > 0.5
        assert d.landmarks.shape == (21, 2)
    print(f"  detected {len(dets)} hands in sample photo, scores={[round(d.score, 2) for d in dets]}")


def test_full_pipeline_runs_headless_without_exceptions():
    rng = np.random.default_rng(3)
    detector = HandDetector()
    tracker = HandTracker()
    calibration = CalibrationManager()
    pinch_detector = PinchDetector()
    canvas = Canvas(640, 480)
    try:
        for i in range(5):
            frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector.detect(rgb, i * 33)
            confirmed = tracker.update(dets, frame, timestamp_s=i * 0.033)
            bound = calibration.update(confirmed)
            for t in confirmed:
                user_id = bound.get(t.track_id)
                if user_id is None:
                    continue
                cursor = cursor_point(t.landmarks)
                if pinch_detector.update(user_id, t.landmarks):
                    canvas.extend_stroke(user_id, cursor, calibration.user_color(user_id), canvas.now_ms())
                else:
                    canvas.end_stroke(user_id, canvas.now_ms())
    finally:
        detector.close()
    print("  5-frame headless pipeline run completed without exceptions")


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"[RUN] {t.__name__}")
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\nAll {len(tests)} offline smoke tests passed.")


if __name__ == "__main__":
    run_all()
