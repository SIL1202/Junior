"""Lightweight appearance embedding for hand re-identification.

The proposal calls for a MobileNetV3 backbone trained with triplet loss on a
self-collected ~10-subject dataset. Collecting that dataset and training a
deep embedding network is out of scope for this implementation (no real
multi-subject capture sessions, no GPU training pipeline available here), so
this module substitutes a classical, training-free appearance descriptor:
an HSV color histogram (skin tone / sleeve / glove color cues) concatenated
with a HOG descriptor (local hand shape/texture), L2-normalized. It plugs
into the same "embedding -> cosine similarity -> re-link lost track" slot
the proposal's appearance head would occupy.
"""
import cv2
import numpy as np

_HOG = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8),
                          _cellSize=(8, 8), _nbins=9)

EMBED_DIM = 32 * 32 + _HOG.getDescriptorSize()  # color hist bins + HOG dims


def extract_embedding(frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Compute an appearance descriptor for the hand crop defined by bbox."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    crop = frame_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hog_desc = _HOG.compute(gray).flatten()

    embedding = np.concatenate([hist, hog_desc]).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding /= norm
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)
