"""User-binding state machine (proposal section 2c).

Any confirmed hand track that isn't yet bound to a user is watched for a
"calibration wave" — a few left-right oscillations of the hand. Once detected,
the track_id is bound to a user_id/color/pen profile. Because this check runs
continuously (not just at t=0), a user can join mid-session with the same
gesture, exactly as the proposal describes. Re-ID (tracker.py) is what keeps
that binding valid across occlusion/re-entry, since the bound track_id is
revived rather than replaced.
"""
from collections import deque
from dataclasses import dataclass, field

from . import config


@dataclass
class _WaveHistory:
    cx: deque = field(default_factory=lambda: deque(maxlen=config.WAVE_WINDOW_FRAMES))
    width: deque = field(default_factory=lambda: deque(maxlen=config.WAVE_WINDOW_FRAMES))


def _direction_changes(values: list[float], min_move: float) -> int:
    diffs = [b - a for a, b in zip(values, values[1:]) if abs(b - a) >= min_move]
    if len(diffs) < 2:
        return 0
    changes = 0
    sign = diffs[0] > 0
    for d in diffs[1:]:
        s = d > 0
        if s != sign:
            changes += 1
            sign = s
    return changes


class CalibrationManager:
    def __init__(self):
        self.track_to_user: dict[int, int] = {}
        self.user_colors: dict[int, tuple] = {}
        self._next_user_id = 0
        self._histories: dict[int, _WaveHistory] = {}

    def _assign_new_user(self, track_id: int) -> int:
        user_id = self._next_user_id
        self._next_user_id += 1
        self.user_colors[user_id] = config.USER_COLORS[user_id % len(config.USER_COLORS)]
        self.track_to_user[track_id] = user_id
        return user_id

    def update(self, tracks):
        """tracks: list of Track-like objects with .track_id, .bbox.
        Returns dict track_id -> user_id for tracks bound this frame or earlier."""
        live_ids = {t.track_id for t in tracks}
        for stale_id in list(self._histories):
            if stale_id not in live_ids:
                del self._histories[stale_id]

        for t in tracks:
            if t.track_id in self.track_to_user:
                continue  # already bound

            hist = self._histories.setdefault(t.track_id, _WaveHistory())
            cx = float((t.bbox[0] + t.bbox[2]) / 2)
            width = float(t.bbox[2] - t.bbox[0])
            hist.cx.append(cx)
            hist.width.append(width)

            if len(hist.cx) < config.WAVE_WINDOW_FRAMES:
                continue

            avg_width = sum(hist.width) / len(hist.width)
            amplitude = max(hist.cx) - min(hist.cx)
            min_move = 0.08 * avg_width
            changes = _direction_changes(list(hist.cx), min_move)

            if (changes >= config.WAVE_MIN_DIRECTION_CHANGES
                    and amplitude >= config.WAVE_MIN_AMPLITUDE_RATIO * avg_width):
                self._assign_new_user(t.track_id)
                del self._histories[t.track_id]

        return dict(self.track_to_user)

    def user_color(self, user_id: int):
        return self.user_colors.get(user_id, (255, 255, 255))
