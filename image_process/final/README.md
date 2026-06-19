# Multi-User Collaborative Air Drawing on a Single Webcam

Real-time system that lets 2–6 users draw on a shared canvas using one
webcam, with per-user hand tracking that survives occlusion, hand-crossing,
and frame exit/re-entry. Implements the system described in
`Proposal Document.pdf`, with the simplifications noted below.

## Setup

```bash
python3.12 -m venv venv      # MediaPipe does not yet support Python 3.13+
./venv/bin/pip install -r requirements.txt
```

The MediaPipe HandLandmarker model bundle is expected at
`air_draw/models/hand_landmarker.task`. If missing, download it:

```bash
curl -L -o air_draw/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run

```bash
./venv/bin/python -m air_draw.main --camera 0
```

**Calibration:** when you step into frame, your hand is outlined in gray
with a "wave to join" hint. Wave your hand left-right 3+ times to bind it to
a user color. This works at any time, so a user can join mid-session.

**Drawing:** pinch your thumb and index finger together to pen-down; release
to pen-up. The cursor is your index fingertip.

**Keys:** `q` quit · `c` clear canvas · `s` save canvas + stroke log to
`session_<timestamp>/`.

## Run the test suite

No webcam is available in the environment this was developed in, so
correctness was verified with synthetic-data and real-photo smoke tests
instead of a live camera session:

```bash
./venv/bin/python -m tests.test_offline
```

This covers: 1€ filter jitter reduction, Re-ID embedding separability,
tracker identity stability through motion/occlusion/re-entry, no ID-swap on
hand-crossing, calibration wave detection (incl. false-positive check against
straight-line drawing motion), pinch+canvas drawing, real hand detection on
a sample photo, and a full headless pipeline run. **You should still run the
live demo yourself with a real webcam** before treating it as done — these
tests can't see whether the wave gesture or pinch feel right in practice.

## Architecture

```
webcam frame
  -> HandDetector        MediaPipe HandLandmarker, up to MAX_HANDS hands/frame
  -> HandTracker         Kalman+IoU two-stage assoc. (ByteTrack-style) + appearance re-ID
  -> CalibrationManager  wave gesture binds track_id -> user_id/color
  -> PinchDetector       thumb-index pinch -> pen up/down (1€-filtered cursor)
  -> Canvas              owner-colored stroke rendering + timestamped stroke log
```

| Module | File |
|---|---|
| Hand detection | `air_draw/hand_detector.py` |
| 1€ filter | `air_draw/one_euro_filter.py` |
| Tracker | `air_draw/tracker.py` |
| Appearance Re-ID | `air_draw/reid.py` |
| Calibration / user binding | `air_draw/calibration.py` |
| Pinch gesture | `air_draw/pinch.py` |
| Canvas + stroke log | `air_draw/canvas.py` |
| GUI overlay | `air_draw/gui.py` |
| Main loop | `air_draw/main.py` |

## Deviations from the proposal (and why)

The proposal targets a research-grade pipeline (custom-trained detector,
learned Re-ID embedding, multi-session annotated evaluation, user study).
Building that requires a GPU training pipeline, real multi-subject capture
sessions, and CVAT annotation — none of which are reproducible inside this
coding session. This implementation keeps the same architecture but swaps in
practical, training-free components so the system actually runs:

| Proposal | This implementation | Why |
|---|---|---|
| Fine-tuned YOLOv8-hand (up to 6 hands) on EgoHands + 11k Hands | MediaPipe `HandLandmarker` Tasks API, `num_hands` configurable | Off-the-shelf detector already handles >2 hands; avoids dataset collection + GPU training |
| MobileNetV3 backbone, triplet-loss-trained embedding | HSV color histogram + HOG descriptor, cosine similarity | No self-collected 10-subject dataset or training pipeline available; classical descriptor fills the same "embedding -> cosine similarity" slot in the tracker |
| ByteTrack | Custom two-stage Kalman+IoU associator modeled on ByteTrack's high/low-confidence tiers | Reimplemented directly against OpenCV/SciPy rather than pulling in the ByteTrack package, to keep the dependency footprint small |

Practical effect: re-ID is noticeably weaker than a trained embedding would
be (see `tests/test_offline.py::test_reid_embedding_distinguishes_appearance`
— the similarity margin between same-hand and different-hand crops is real
but modest), so `REID_COSINE_THRESH` in `air_draw/config.py` is a
manually-tuned parameter you may need to adjust per lighting/clothing. This
is the main limitation to call out in the report's Discussion section.

## Known limitations

- No GPU/learned detector — relies on MediaPipe's hand detector's own
  limitations (fast motion blur, hands at extreme angles).
- Re-ID uses a hand-tuned classical descriptor, not a trained embedding —
  expect more identity loss on long occlusions or big lighting changes than
  the proposal's 0.80 IDF1 target.
- `REID_COSINE_THRESH`, `WAVE_*`, and `PINCH_*` thresholds in `config.py` are
  manually tuned and may need adjustment for your camera/lighting/skin tone.
- No camera access was available in the development environment, so the
  live wave-gesture/pinch/multi-hand-crossing UX has only been validated via
  the synthetic unit tests in `tests/test_offline.py`, not a live session —
  run it yourself with your webcam before relying on it.
