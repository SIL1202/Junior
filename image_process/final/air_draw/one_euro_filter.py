"""1€ Filter (Casiez et al., CHI 2012) for smoothing noisy landmark positions.

Operates elementwise on numpy arrays so a single filter instance can smooth
all 21 landmarks (x, y) of a hand at once.
"""
import math

import numpy as np


class LowPassFilter:
    def __init__(self, shape):
        self._y = np.zeros(shape, dtype=np.float64)
        self._initialized = False

    def filter(self, x: np.ndarray, alpha: np.ndarray):
        if not self._initialized:
            self._y = x.copy()
            self._initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def value(self):
        return self._y


class OneEuroFilter:
    def __init__(self, shape, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_filter = LowPassFilter(shape)
        self._dx_filter = LowPassFilter(shape)
        self._last_time = None

    @staticmethod
    def _alpha(freq, cutoff):
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self._x_filter._initialized = False
        self._dx_filter._initialized = False
        self._last_time = None

    def __call__(self, x: np.ndarray, timestamp_s: float = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)

        if timestamp_s is not None and self._last_time is not None:
            dt = timestamp_s - self._last_time
            if dt > 0:
                self.freq = 1.0 / dt
        if timestamp_s is not None:
            self._last_time = timestamp_s

        prev_x = self._x_filter.value if self._x_filter._initialized else x
        dx = (x - prev_x) * self.freq

        edx = self._dx_filter.filter(dx, np.full_like(dx, self._alpha(self.freq, self.dcutoff)))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        alpha = self._alpha(self.freq, cutoff)
        return self._x_filter.filter(x, alpha)
