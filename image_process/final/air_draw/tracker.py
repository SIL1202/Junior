"""Simplified ByteTrack-style multi-hand tracker with appearance re-ID.

Pipeline per frame:
  1. Kalman-predict all active tracks' boxes.
  2. Stage 1: Hungarian/IoU match active tracks <-> high-score detections (tight IoU gate).
  3. Stage 2: Hungarian/IoU match still-unmatched tracks <-> low-score detections (loose IoU gate).
     This two-tier scheme is ByteTrack's core idea: don't throw away low-confidence boxes,
     use them to rescue tracks that would otherwise be lost to occlusion.
  4. Tracks unmatched for too long move from "active" (Kalman-coasting) to "lost"
     (appearance-only) and are dropped entirely after MAX_AGE_LOST frames.
  5. Detections still unmatched are checked against the lost-track pool by cosine similarity
     of appearance embeddings ("re-ID"); a hit revives the old track_id instead of minting a
     new one — this is what lets a hand keep its identity (and therefore its bound user_id)
     across a long occlusion or a brief exit-and-re-entry.
  6. Anything still unmatched becomes a new tentative track.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config, reid
from .hand_detector import Detection
from .one_euro_filter import OneEuroFilter

HIGH_SCORE_THRESH = 0.7


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xx1 = max(box_a[0], box_b[0])
    yy1 = max(box_a[1], box_b[1])
    xx2 = min(box_a[2], box_b[2])
    yy2 = min(box_a[3], box_b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    cx, cy, w, h = z
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def _clamp_velocity(kf: cv2.KalmanFilter):
    """Bound the Kalman velocity state so a track that stops getting matched
    extrapolates a short distance rather than flying off-screen indefinitely."""
    v = config.MAX_VELOCITY_PX_PER_FRAME
    kf.statePre[4:8] = np.clip(kf.statePre[4:8], -v, v)
    kf.statePost[4:8] = np.clip(kf.statePost[4:8], -v, v)


def _center_dist_cost(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = (box_a[:2] + box_a[2:]) / 2
    center_b = (box_b[:2] + box_b[2:]) / 2
    diag_a = np.linalg.norm(box_a[2:] - box_a[:2])
    diag_b = np.linalg.norm(box_b[2:] - box_b[:2])
    diag = max(diag_a, diag_b, 1.0)
    return float(np.linalg.norm(center_a - center_b) / diag)


def _make_kalman(bbox: np.ndarray) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    for i in range(4):
        kf.transitionMatrix[i, i + 4] = 1.0
    kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1.0
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5.0
    kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
    z = _bbox_to_z(bbox)
    kf.statePost = np.concatenate([z, np.zeros(4, dtype=np.float32)]).astype(np.float32)
    return kf


@dataclass
class Track:
    track_id: int
    kf: cv2.KalmanFilter
    bbox: np.ndarray
    landmarks: np.ndarray
    embedding: np.ndarray
    handedness: str
    state: str = "tentative"   # tentative -> confirmed -> lost -> (removed)
    hits: int = 1
    time_since_update: int = 0
    landmark_filter: OneEuroFilter = field(default=None)
    user_id: int = None
    last_matched_bbox: np.ndarray = None  # bbox as of the last real detection (no extrapolation)

    def predict(self):
        self.kf.predict()
        _clamp_velocity(self.kf)
        self.bbox = _z_to_bbox(self.kf.statePre[:4].flatten())

    def update(self, det: Detection, embedding: np.ndarray, timestamp_s: float):
        z = _bbox_to_z(det.bbox)
        self.kf.correct(z)
        _clamp_velocity(self.kf)
        self.bbox = det.bbox
        self.last_matched_bbox = det.bbox
        if self.landmark_filter is None:
            self.landmark_filter = OneEuroFilter(
                shape=det.landmarks.shape, freq=config.ONE_EURO_FREQ,
                mincutoff=config.ONE_EURO_MINCUTOFF, beta=config.ONE_EURO_BETA,
                dcutoff=config.ONE_EURO_DCUTOFF)
        self.landmarks = self.landmark_filter(det.landmarks, timestamp_s=timestamp_s)
        self.embedding = 0.9 * self.embedding + 0.1 * embedding
        self.handedness = det.handedness
        self.hits += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.hits >= config.MIN_HITS_TO_CONFIRM:
            self.state = "confirmed"


class HandTracker:
    def __init__(self):
        self._next_id = 1
        self.active: list[Track] = []
        self.lost: list[Track] = []

    def _new_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: list[Detection], frame_bgr: np.ndarray, timestamp_s: float):
        embeddings = [reid.extract_embedding(frame_bgr, d.bbox) for d in detections]

        for t in self.active:
            t.predict()

        high_idx = [i for i, d in enumerate(detections) if d.score >= HIGH_SCORE_THRESH]
        low_idx = [i for i, d in enumerate(detections) if d.score < HIGH_SCORE_THRESH]

        unmatched_tracks = list(range(len(self.active)))
        matched_det = set()

        def _match(track_idx_pool, det_idx_pool, iou_thresh):
            if not track_idx_pool or not det_idx_pool:
                return [], track_idx_pool, det_idx_pool
            cost = np.ones((len(track_idx_pool), len(det_idx_pool)), dtype=np.float32)
            for r, ti in enumerate(track_idx_pool):
                for c, di in enumerate(det_idx_pool):
                    cost[r, c] = 1.0 - iou(self.active[ti].bbox, detections[di].bbox)
            rows, cols = linear_sum_assignment(cost)
            matches, leftover_tracks, leftover_dets = [], list(track_idx_pool), list(det_idx_pool)
            for r, c in zip(rows, cols):
                ti, di = track_idx_pool[r], det_idx_pool[c]
                if 1.0 - cost[r, c] >= iou_thresh:
                    matches.append((ti, di))
                    leftover_tracks.remove(ti)
                    leftover_dets.remove(di)
            return matches, leftover_tracks, leftover_dets

        # stage 1: active tracks vs high-score detections (tight IoU gate)
        matches1, unmatched_tracks, unmatched_high = _match(
            unmatched_tracks, high_idx, config.IOU_HIGH_THRESH)
        # stage 2: leftover tracks vs low-score detections (loose IoU gate, occlusion rescue)
        matches2, unmatched_tracks, unmatched_low = _match(
            unmatched_tracks, low_idx, config.IOU_LOW_THRESH)

        # stage 3: fallback center-distance gate, measured from each track's last *real*
        # detection (not the Kalman-extrapolated box). IoU drops to exactly 0 once boxes
        # stop overlapping at all, which happens every frame during fast motion (e.g. a
        # fast wave) even though the hand clearly hasn't gone anywhere else. Gating on the
        # last observed position rather than the velocity-extrapolated one matters because
        # a constant-velocity Kalman model overshoots badly right at a direction reversal —
        # exactly what a "wave" is — so the prediction itself is the less reliable reference.
        unmatched_dets_pool = unmatched_high + unmatched_low
        if unmatched_tracks and unmatched_dets_pool:
            cost = np.zeros((len(unmatched_tracks), len(unmatched_dets_pool)), dtype=np.float32)
            for r, ti in enumerate(unmatched_tracks):
                ref_bbox = self.active[ti].last_matched_bbox
                for c, di in enumerate(unmatched_dets_pool):
                    cost[r, c] = _center_dist_cost(ref_bbox, detections[di].bbox)
            rows, cols = linear_sum_assignment(cost)
            matches3 = []
            for r, c in zip(rows, cols):
                if cost[r, c] <= config.CENTER_DIST_GATE_RATIO:
                    ti, di = unmatched_tracks[r], unmatched_dets_pool[c]
                    matches3.append((ti, di))
            for ti, di in matches3:
                unmatched_tracks.remove(ti)
                unmatched_dets_pool.remove(di)
        else:
            matches3 = []

        for ti, di in matches1 + matches2 + matches3:
            self.active[ti].update(detections[di], embeddings[di], timestamp_s)
            matched_det.add(di)

        # age out unmatched active tracks; move long-stale ones to the lost pool
        still_active = []
        for i, t in enumerate(self.active):
            if i in unmatched_tracks:
                t.time_since_update += 1
                if t.time_since_update > config.MAX_AGE_ACTIVE:
                    t.state = "lost"
                    self.lost.append(t)
                    continue
            still_active.append(t)
        self.active = still_active

        # drop lost tracks that have been gone too long
        self.lost = [t for t in self.lost if t.time_since_update <= config.MAX_AGE_LOST]
        for t in self.lost:
            if t not in still_active:
                t.time_since_update += 1

        # re-ID: try to revive lost tracks using appearance similarity to unmatched detections
        unmatched_all = sorted(set(range(len(detections))) - matched_det)
        revived_ids = set()
        for di in list(unmatched_all):
            best_track, best_sim = None, 0.0
            for t in self.lost:
                sim = reid.cosine_similarity(t.embedding, embeddings[di])
                if sim > best_sim:
                    best_sim, best_track = sim, t
            if best_track is not None and best_sim >= config.REID_COSINE_THRESH:
                best_track.state = "confirmed"
                best_track.kf = _make_kalman(detections[di].bbox)
                best_track.update(detections[di], embeddings[di], timestamp_s)
                self.lost.remove(best_track)
                self.active.append(best_track)
                matched_det.add(di)
                revived_ids.add(di)

        # remaining detections spawn new tentative tracks
        for di in sorted(set(range(len(detections))) - matched_det):
            det = detections[di]
            track = Track(
                track_id=self._new_track_id(),
                kf=_make_kalman(det.bbox),
                bbox=det.bbox,
                landmarks=det.landmarks,
                embedding=embeddings[di],
                handedness=det.handedness,
                last_matched_bbox=det.bbox,
            )
            track.landmark_filter = OneEuroFilter(
                shape=det.landmarks.shape, freq=config.ONE_EURO_FREQ,
                mincutoff=config.ONE_EURO_MINCUTOFF, beta=config.ONE_EURO_BETA,
                dcutoff=config.ONE_EURO_DCUTOFF)
            self.active.append(track)

        return [t for t in self.active if t.state == "confirmed"]
