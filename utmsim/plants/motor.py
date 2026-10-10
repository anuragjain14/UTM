"""PL-1: 50 HP (~37 kW) induction motor with vibration, acoustic, thermal and electrical sensing.

Sensors report condition-monitoring quantities at the sampling rate: vibration velocity RMS
(mm/s), acoustic level (dB), winding/housing temperature (degC), phase current RMS (A), shaft speed
(rpm), ambient temperature (degC).
"""

import numpy as np

from ..sensors import ar1
from .base import Nominal, Plant, med, perturb, tile

NOM = {
    "w_sync": 1800.0,   # rpm
    "s_rated": 0.0167,  # rated slip
    "w_rated": 1770.0,
    "I_rated": 57.0,    # A at 460 V, 50 HP
    "I_nl_frac": 0.30,
    "v0": 1.2,          # mm/s at rated speed, no load term
    "kappa_L": 0.40,    # vibration load sensitivity
    "a0": 60.0,         # dB
    "c_a": 4.0,         # dB per mm/s
    "R_th": 0.035,      # K/W
    "k_cu": 0.30,       # W/A^2 (copper loss coefficient)
    "P_f0": 300.0,      # W friction/windage
    "tau_th_h": 1.5,
}


class MotorPlant(Plant):
    name = "motor"
    scales = {"vibration": 1.5, "acoustic": 66.0, "temperature": 70.0, "current": 45.0,
              "speed": 1770.0, "ambient": 20.0}
    spans = {"vibration": 1.5, "acoustic": 10.0, "temperature": 40.0, "current": 40.0,
             "speed": 30.0, "ambient": 10.0}
    fixed_counts = {"speed": 2, "ambient": 1}
    feature_modalities = ("vibration", "acoustic", "temperature", "current", "speed")
    target_modality = {"N1": "current", "N2": "current", "N3": "vibration", "N4": "vibration",
                       "N5": "temperature", "N6": "speed", "A1": "current", "A2": "vibration",
                       "A3": "current", "A4": "current", "A5": "current"}

    def __init__(self, spec, rng, mismatch):
        super().__init__(spec, rng, mismatch)
        self.p = perturb(rng, NOM, ["s_rated", "I_rated", "I_nl_frac", "v0", "kappa_L", "c_a",
                                    "R_th", "k_cu", "P_f0", "tau_th_h"], mismatch)
        self.gamma = 1.0 + mismatch * rng.uniform(-0.08, 0.08)          # unmodelled current law
        self.kappa_T = mismatch * rng.uniform(0.0, 0.10)                # unmodelled thermal-vib term
        self.loc_vib = rng.uniform(0.85, 1.15, self.counts["vibration"])
        self.loc_temp = rng.uniform(-3.0, 3.0, self.counts["temperature"])
        self.amb_phase = rng.uniform(0, 2 * np.pi)
        self.L_op, self.t_next = float(rng.uniform(0.45, 0.95)), float(rng.exponential(24.0))
        self.theta = 45.0
        self.h = 1.0

    def ambient(self, t_h):
        return 20.0 + 5.0 * np.sin(2 * np.pi * t_h / 24.0 + self.amb_phase)

    def current_law(self, L):
        p = self.p
        I_nl = p["I_nl_frac"] * p["I_rated"]
        return I_nl + (p["I_rated"] - I_nl) * np.clip(L, 0.0, None) ** self.gamma

    def step_slow(self, t_h, dt_h):
        if t_h >= self.t_next:
            self.L_op = float(self.rng.uniform(0.45, 0.95))
            self.t_next = t_h + float(self.rng.exponential(24.0))
        self.h = 1.0 - 0.7 * self.fault_level(t_h)        # P1 bearing wear
        p = self.p
        I = self.current_law(self.L_op)
        P_loss = p["k_cu"] * I ** 2 + p["P_f0"] * (1.0 + 2.0 * (1.0 - self.h))
        theta_ss = self.ambient(t_h) + p["R_th"] * P_loss
        self.theta = theta_ss + (self.theta - theta_ss) * np.exp(-dt_h / p["tau_th_h"])

    def true_window(self, t_h, fs, n, rng):
        p = self.p
        ell = ar1(rng, 1, n, np.exp(-1.0 / (fs * 0.5)), 0.03)[0]
        L = np.clip(self.L_op + ell, 0.05, 1.1)
        w = p["w_sync"] * (1.0 - p["s_rated"] * L)
        I = self.current_law(L)
        v = (p["v0"] * (w / p["w_rated"]) ** 2 * (1.0 + p["kappa_L"] * L)
             * (1.0 + 2.0 * (1.0 - self.h) ** 1.5) * (1.0 + self.kappa_T * (self.theta - 60.0) / 60.0))
        v = v * (1.0 + ar1(rng, 1, n, np.exp(-1.0 / (fs * 0.05)), 0.03)[0])
        a = p["a0"] + p["c_a"] * v + ar1(rng, 1, n, np.exp(-1.0 / (fs * 0.05)), 0.3)[0]
        theta = self.theta + 0.02 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
        true = {
            "vibration": tile(v, self.counts["vibration"], self.loc_vib),
            "acoustic": tile(a, self.counts["acoustic"]),
            "temperature": tile(theta, self.counts["temperature"]) + self.loc_temp[:, None],
            "current": tile(I, self.counts["current"]),
            "speed": tile(w, self.counts["speed"]),
            "ambient": tile(np.full(n, self.ambient(t_h)), 1),
        }
        return true, {}


class MotorNominal(Nominal):
    relation_groups = ("R1_current_speed", "R2_acoustic_vibration", "R3_thermal", "R4_vibration_speed")
    # v2 (B3): relations that a known process fault (P1/P2/P3) shifts consistently across sensors
    fault_explainable = ("R3_thermal", "R4_vibration_speed")

    def __init__(self):
        super().__init__()
        self.p = dict(NOM)
        self.theta_hat = None
        self.last_t = None

    def load_from_speed(self, w):
        return np.clip((1.0 - w / self.p["w_sync"]) / self.p["s_rated"], 0.0, 1.2)

    def current(self, L):
        I_nl = self.p["I_nl_frac"] * self.p["I_rated"]
        return I_nl + (self.p["I_rated"] - I_nl) * L

    def vib(self, w, L):
        return self.p["v0"] * (w / self.p["w_rated"]) ** 2 * (1.0 + self.p["kappa_L"] * L)

    def residuals(self, r, ctx, t_h):
        p = self.p
        w = float(np.median(med(r["speed"])))
        L = float(self.load_from_speed(w))
        I_med, v_med, a_med = med(r["current"]), med(r["vibration"]), med(r["acoustic"])
        out = {
            "R1_current_speed": I_med - self.current(L),
            "R2_acoustic_vibration": np.concatenate([a_med - (p["a0"] + p["c_a"] * v_med.mean()),
                                                     v_med - (a_med.mean() - p["a0"]) / p["c_a"]]),
            "R4_vibration_speed": v_med - self.vib(w, L),
        }
        amb = float(np.median(r["ambient"]))
        I_mean = float(I_med.mean())
        theta_ss = amb + p["R_th"] * (p["k_cu"] * I_mean ** 2 + p["P_f0"])
        if self.theta_hat is None:
            self.theta_hat = float(np.median(med(r["temperature"])))
        else:
            dt = max(t_h - self.last_t, 0.0)
            self.theta_hat = theta_ss + (self.theta_hat - theta_ss) * np.exp(-dt / p["tau_th_h"])
        self.last_t = t_h
        out["R3_thermal"] = med(r["temperature"]) - self.theta_hat
        return out

    def soft_sensor(self, r, ctx):
        x = np.array([1.0, float(np.median(med(r["speed"]))) / self.p["w_sync"],
                      float(med(r["vibration"]).mean()) / self.p["v0"],
                      float(med(r["acoustic"]).mean()) / self.p["a0"]])
        return x, float(med(r["current"]).mean())

    def k2_spoof(self, r, ctx, delta, st):
        """Fake a load change `delta` (fraction of rated load) consistently in current, speed,
        vibration, acoustic and (via the nominal thermal model) temperature."""
        p = self.p
        out = {m: v.copy() for m, v in r.items()}
        w = med(r["speed"]).mean()
        L = self.load_from_speed(w)
        dw = -p["w_sync"] * p["s_rated"] * delta
        out["speed"] += dw
        out["current"] += self.current(L + delta) - self.current(L)
        dv = self.vib(w + dw, L + delta) - self.vib(w, L)
        out["vibration"] += dv
        out["acoustic"] += p["c_a"] * dv
        I0 = float(med(r["current"]).mean())
        I1 = I0 + float(self.current(L + delta) - self.current(L))
        target = p["R_th"] * p["k_cu"] * (I1 ** 2 - I0 ** 2)
        dt = st.get("dt_h", 0.0)
        st["dtheta"] = target + (st.get("dtheta", 0.0) - target) * np.exp(-dt / p["tau_th_h"])
        out["temperature"] += st["dtheta"]
        return out
