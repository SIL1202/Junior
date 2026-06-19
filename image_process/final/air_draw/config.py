import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
HAND_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MAX_HANDS = 6
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5

# -- Tracker --
IOU_HIGH_THRESH = 0.3       # stage-1 association (confident detections)
IOU_LOW_THRESH = 0.1        # stage-2 association (low-confidence / recovery)
CENTER_DIST_GATE_RATIO = 1.5  # stage-3 fallback: match by center distance / box diagonal
                               # (IoU goes to 0 for fast motion even when it's clearly the same hand)
MAX_AGE_ACTIVE = 8          # frames a confirmed track survives purely on Kalman prediction
                             # (kept short so a track that stops matching doesn't visibly
                             # drift away on stale velocity for too long before being dropped)
MAX_VELOCITY_PX_PER_FRAME = 200  # clamp on Kalman velocity state to bound prediction runaway
MAX_AGE_LOST = 150          # frames a lost track's appearance embedding stays eligible for re-ID (~5s @30fps)
MIN_HITS_TO_CONFIRM = 3
REID_COSINE_THRESH = 0.55   # min cosine similarity to re-link a lost track to a new detection

# -- Pinch / drawing --
PINCH_CLOSE_RATIO = 0.35    # thumb-index distance / hand-size below this => pen down
PINCH_OPEN_RATIO = 0.45     # above this => pen up (hysteresis band avoids flicker)
STROKE_MIN_POINT_DIST = 2   # px, drop near-duplicate points

# -- Calibration (wave gesture) --
WAVE_WINDOW_FRAMES = 24     # ~0.8s @30fps window to detect a wave
WAVE_MIN_DIRECTION_CHANGES = 3
WAVE_MIN_AMPLITUDE_RATIO = 0.6  # amplitude relative to hand bbox width

USER_COLORS = [
    (66, 135, 245),   # blue
    (52, 199, 89),    # green
    (255, 149, 0),    # orange
    (255, 59, 48),    # red
    (175, 82, 222),   # purple
    (255, 214, 10),   # yellow
]

# -- One-Euro filter defaults (pixel-coordinate scale; Casiez et al. 2012 defaults) --
ONE_EURO_FREQ = 30.0
ONE_EURO_MINCUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_DCUTOFF = 1.0
