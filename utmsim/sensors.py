"""Sensor measurement model (requirements PL-6, PL-7, AC-1).

Each physical sensor has its own gain and offset error, coloured noise, and a small unmodelled
nonlinearity. Detectors never see these parameters.
"""

import numpy as np


def ar1(rng, n_rows, n, phi, sigma):
    """Stationary AR(1) noise, shape (n_rows, n), marginal std = sigma."""
    phi = np.asarray(phi, dtype=float).reshape(-1, 1) * np.ones((n_rows, 1))
    e = rng.standard_normal((n_rows, n))
    out = np.empty((n_rows, n))
    out[:, 0] = e[:, 0]
    scale = np.sqrt(1.0 - phi[:, 0] ** 2)
    for t in range(1, n):
        out[:, t] = phi[:, 0] * out[:, t - 1] + scale * e[:, t]
    return out * np.asarray(sigma, dtype=float).reshape(-1, 1)


class SensorArray:
    """Measurement errors for all sensors of one modality."""

    def __init__(self, rng, n_sensors, scale, noise_frac, mismatch, span=None):
        self.n = n_sensors
        self.scale = float(scale)                       # typical signal level of the modality
        self.span = float(span if span is not None else scale)   # meaningful variation range
        self.gain = 1.0 + rng.normal(0.0, 0.01, n_sensors) * min(1.0, self.span / self.scale)
        self.offset = rng.normal(0.0, 0.005 * self.span, n_sensors)
        # Noise level (Table 2: 0.1-10% "of signal") is taken relative to the measurement SPAN,
        # e.g. a speed encoder's noise is a fraction of the slip range, not of 1,800 rpm.
        self.sigma = noise_frac * self.span * rng.uniform(0.7, 1.3, n_sensors)
        self.phi = rng.uniform(0.2, 0.8, n_sensors)     # noise colour
        self.nonlin = mismatch * rng.normal(0.0, 0.02, n_sensors) / max(self.scale, 1e-9)

    def measure(self, rng, true):
        """true: (n_sensors, n) physical values at each sensor location."""
        x = np.asarray(true, dtype=float)
        dev = x - self.scale
        y = self.gain[:, None] * x + self.offset[:, None] + self.nonlin[:, None] * dev ** 2
        return y + ar1(rng, self.n, x.shape[1], self.phi, self.sigma)
