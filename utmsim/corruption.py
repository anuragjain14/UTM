"""Sensor-corruption mechanisms (requirements §3.1, §3.2, EV-3).

Natural and adversarial mechanisms each have their OWN generative law. For gradual attacks the
attacker can be `naive` (its own linear/abrupt law) or `mimic` (deliberately imitating the
natural law). The attribution experiment (E-ATTR) is therefore informative in both directions.

Labels produced here go to the output files only; detectors never receive them.
"""

import numpy as np

SEVERITY = {"low": 1.0, "medium": 3.0, "high": 6.0}


class Corruption:
    def __init__(self, spec, plant, nominal, rng):
        self.mech = spec["mechanism"]
        self.origin = spec["origin"]
        self.onset = float(spec["onset_h"])
        self.k = spec.get("magnitude_k")                 # v2 detectability mode (A2)
        self.S = float(self.k) if self.k is not None else SEVERITY[spec["severity"]]
        self.knowledge = spec.get("attacker_knowledge")
        self.mode = spec.get("attack_mode")
        self.rng = rng
        self.nominal = nominal
        self.mod = plant.target_modality[self.mech]
        sa = plant.sensors[self.mod]
        self._sigma = sa.sigma
        n_s = sa.n
        if self.mech in ("A3",) or (self.mech == "A4" and self.knowledge in ("K1", "K2")):
            self.idx = np.arange(n_s)                       # whole modality
        elif self.mech == "A4":
            self.idx = np.sort(rng.choice(n_s, size=min(2, n_s), replace=False))
        else:
            self.idx = np.array([int(rng.integers(n_s))])
        self.ref = float(max(np.median(sa.sigma), 0.005 * sa.span))    # per-sensor reference unit
        self.sign = 1 if rng.random() < 0.5 else -1
        self.mag = 2.0 * self.S * self.ref * self.sign   # v1 unit; replaced at onset in detectability mode
        self.hist = []                                   # clean medians of affected sensors (A2)
        self.tau = float(rng.uniform(300.0, 2000.0))     # natural (or mimicked) time constant
        self.T_ramp = float(rng.uniform(200.0, 2000.0))  # naive attack ramp duration
        self.daily = float(rng.uniform(0.05, 0.15))      # natural drift: temperature coefficient
        self.daily_phase = float(rng.uniform(0, 2 * np.pi))
        self.gain_step = 0.02 * self.S * (1 if rng.random() < 0.5 else -1)
        self.p_stuck = 0.1 + 0.3 * min(self.S, 6.0) / 6.0
        self.record_at_h = None                          # A3: warm-up window to record
        self.recorded = None
        self.last_med = None
        self.k_state = {}
        self.last_t = None

    # ---- warm-up hooks --------------------------------------------------------------------
    def choose_record_time(self, warmup_times):
        if self.mech == "A3":
            self.record_at_h = float(warmup_times[int(self.rng.integers(len(warmup_times)))])

    def observe_clean(self, t_h, readings):
        if self.record_at_h is not None and abs(t_h - self.record_at_h) < 1e-9:
            self.recorded = readings[self.mod].copy()
        self.last_med = np.median(readings[self.mod], axis=1)
        if self.k is not None:
            self.hist.append(self.last_med[self.idx])

    def _set_operating_magnitude(self, sigma):
        """Detectability mode: magnitude = k x the affected sensors' operating variability (std of
        clean window medians before onset), floored at the sensor noise."""
        if self.k is None or getattr(self, "_mag_set", False):
            return
        if len(self.hist) >= 3:
            op = float(np.median(np.std(np.vstack(self.hist), axis=0)))
        else:
            op = 0.0
        self.op_ref = max(op, float(np.median(sigma)), 1e-12)
        self.mag = self.k * self.op_ref * self.sign
        self._mag_set = True

    # ---- laws -----------------------------------------------------------------------------
    def natural_drift(self, dt_h, t_h):
        return self.mag * (1.0 - np.exp(-dt_h / self.tau)) * (
            1.0 + self.daily * np.sin(2 * np.pi * t_h / 24.0 + self.daily_phase))

    def apply(self, readings, ctx, t_h):
        """Return (readings, compromised). `readings` is modified in place on a copy."""
        if t_h < self.onset:
            self.observe_clean(t_h, readings)
            return readings, False
        self._set_operating_magnitude(self._sigma)
        out = {m: v.copy() for m, v in readings.items()}
        x = out[self.mod]
        i = self.idx
        dt = t_h - self.onset
        m = self.mech
        compromised = True
        if m == "N1":
            x[i] += self.natural_drift(dt, t_h)
        elif m == "A1":
            if self.mode == "mimic":
                x[i] += self.natural_drift(dt, t_h)
            else:
                x[i] += self.mag * min(1.0, dt / self.T_ramp)
        elif m == "N2":
            x[i] = x[i] * (1.0 + self.gain_step) + self.mag
        elif m == "A5":
            x[i] += self.mag
        elif m == "N3":
            grow = min(1.0, dt / (3.0 * self.tau))
            x[i] += self.rng.normal(0.0, abs(self.mag) * grow + 1e-12, x[i].shape)
        elif m == "A2":
            if self.mode == "mimic":
                grow = min(1.0, dt / (3.0 * self.tau))
                x[i] += self.rng.normal(0.0, abs(self.mag) * grow + 1e-12, x[i].shape)
            else:
                mu = x[i].mean(axis=1, keepdims=True)
                x[i] = mu + (x[i] - mu) * (1.0 + self.S / 2.0)
        elif m == "N4":
            g = 1.0 - 0.08 * self.S * min(1.0, dt / self.tau)
            x[i] = x[i] * g + self.rng.normal(0.0, 0.3 * abs(self.mag) * min(1.0, dt / self.tau) + 1e-12, x[i].shape)
        elif m == "N5":
            if self.rng.random() < self.p_stuck:
                x[i] = self.last_med[i][:, None] if self.last_med is not None else x[i].mean(axis=1, keepdims=True)
            else:
                compromised = False
                self.last_med = np.median(readings[self.mod], axis=1)
        elif m == "N6":
            q = abs(self.mag) * min(1.0, dt / self.tau) + 1e-12
            x[i] = np.round(x[i] / q) * q + self.rng.uniform(-q / 2, q / 2, x[i].shape)
        elif m == "A3":
            if self.recorded is not None:
                x[:] = self.recorded
        elif m == "A4":
            ramp = min(1.0, dt / 48.0)
            if self.knowledge == "K2":
                self.k_state["dt_h"] = 0.0 if self.last_t is None else t_h - self.last_t
                delta = self.k2_delta() * ramp
                out = self.nominal.k2_spoof(out, ctx, delta, self.k_state)
            else:
                x[i] += self.mag * ramp
        self.last_t = t_h
        return out, compromised

    def k2_delta(self):
        """Fake-change magnitude in the plant's own units (load fraction, K, rad)."""
        unit = {"MotorNominal": 0.05, "ReactorNominal": 0.5, "RobotNominal": 0.01}[type(self.nominal).__name__]
        return unit * self.S * np.sign(self.mag)

    def label(self):
        return {"origin": self.origin, "mechanism": self.mech, "modality": self.mod,
                "affected_sensors": ";".join(str(int(k)) for k in self.idx)}
