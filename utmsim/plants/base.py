"""Common plant interface (requirements PL-4..PL-8, AC-1).

A plant has two sides:
  * the GENERATOR (true dynamics, per-run perturbed parameters, unmodelled terms), which produces
    true physical signals at each sensor location;
  * the NOMINAL model (`Nominal`), which is the only knowledge the detectors get: simplified
    equations with nominal parameters, used for physics checks (Algorithm 2), the soft sensor
    (Algorithm 3) and by a K2 attacker.
"""

import numpy as np

from ..sensors import SensorArray


def perturb(rng, nominal: dict, keys, mismatch, rel=0.10):
    """Generator parameters = nominal * (1 + mismatch * U(-rel, rel)) for the listed keys."""
    out = dict(nominal)
    for k in keys:
        out[k] = nominal[k] * (1.0 + mismatch * rng.uniform(-rel, rel))
    return out


class Plant:
    name = "base"
    scales: dict = {}          # modality -> typical signal level
    spans: dict = {}           # modality -> meaningful variation range (sensor noise, severity)
    fixed_counts: dict = {}    # modalities with a fixed number of sensors
    feature_modalities: tuple = ()
    target_modality: dict = {}  # mechanism -> modality it corrupts

    def __init__(self, spec, rng, mismatch):
        self.spec = spec
        self.rng = rng
        self.mismatch = float(mismatch)
        n = int(spec["sensors_per_modality"])
        self.counts = {m: self.fixed_counts.get(m, n) for m in self.scales}
        self.sensors = {m: SensorArray(rng, c, self.scales[m], spec["noise_frac"], self.mismatch,
                                       span=self.spans.get(m, self.scales[m]))
                        for m, c in self.counts.items()}
        self.fault_onset = spec.get("fault_onset_h")
        self.fault_rate = float(rng.uniform(0.5, 1.5)) if self.fault_onset is not None else 0.0

    def fault_level(self, t_h):
        """0 before the process-fault onset, then grows (fraction of full degradation)."""
        if self.fault_onset is None or t_h < self.fault_onset:
            return 0.0
        return float(min(1.0, self.fault_rate * (t_h - self.fault_onset) / 2000.0))

    # -- generator ----------------------------------------------------------------------------
    def step_slow(self, t_h, dt_h):
        raise NotImplementedError

    def true_window(self, t_h, fs, n, rng):
        """Return (true signals dict modality -> (n_sensors, n), context dict). The context holds
        information a real controller would know (program phase, setpoint) and is visible to the
        detectors."""
        raise NotImplementedError

    def groups(self):
        """Redundancy groups: for each modality, index arrays of sensors measuring the SAME
        quantity (used only for voting-type checks). Default: the whole modality."""
        return {m: [np.arange(c)] for m, c in self.counts.items()}

    def measure(self, true, rng):
        return {m: self.sensors[m].measure(rng, true[m]) for m in true}


class Nominal:
    """Detector-side knowledge. Subclasses implement `residuals` and `soft_sensor`."""

    relation_groups: tuple = ()
    fault_explainable: tuple = ()      # v2 (B3)

    def __init__(self):
        self.state = {}

    def residuals(self, readings, ctx, t_h):
        """Return dict group -> 1-D array of residual entries (one per checked sensor)."""
        raise NotImplementedError

    def soft_sensor(self, readings, ctx):
        """Return (inputs vector, target scalar) for the Algorithm 3 learner."""
        raise NotImplementedError

    def k2_spoof(self, readings, ctx, delta, attacker_state):
        """K2 attacker: modify readings so a fake change `delta` is consistent with this nominal
        model across all modalities it relates."""
        raise NotImplementedError


def med(x):
    return np.median(x, axis=-1)


def tile(row, n_rows, factors=None):
    out = np.tile(np.asarray(row, dtype=float), (n_rows, 1))
    if factors is not None:
        out *= np.asarray(factors, dtype=float)[:, None]
    return out
