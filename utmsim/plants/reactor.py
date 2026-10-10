"""PL-2: jacketed batch reactor with temperature, pressure, coolant-flow, level and feed-flow
sensing. 10-hour batch cycle: fill (feed on), heat, react (exothermic, controlled at setpoint by
coolant flow), drain."""

import numpy as np

from ..sensors import ar1
from .base import Nominal, Plant, med, perturb, tile

NOM = {
    "cycle_h": 10.0,
    "A": 3.0,            # m^2 cross-section
    "F_in": 2.0,         # m^3/h during fill
    "F_drain": 1.0,      # m^3/h during drain
    "P0": 1.5,           # bar at T0, empty headspace reference
    "T0": 25.0,
    "b_PT": 0.012,       # bar/K
    "c_PL": 0.05,        # bar/m (headspace compression with level)
    "F0_hold": 4.0,      # m^3/h coolant base flow while holding temperature
    "F0_react": 9.0,     # m^3/h coolant base flow while reacting
    "Kc": 1.5,           # m^3/h per K controller gain
}

PHASES = [(0.0, 1.0, "fill", 40.0), (1.0, 2.0, "heat", 80.0), (2.0, 8.0, "react", 80.0),
          (8.0, 10.0, "drain", 40.0)]


def phase_at(t_h, cycle_h):
    x = t_h % cycle_h
    for a, b, name, sp in PHASES:
        if a <= x < b:
            return name, sp
    return "drain", 40.0


class ReactorPlant(Plant):
    name = "reactor"
    scales = {"temperature": 70.0, "pressure": 2.3, "coolant_flow": 8.0, "level": 1.5,
              "feed_flow": 2.0}
    spans = {"temperature": 50.0, "pressure": 1.0, "coolant_flow": 8.0, "level": 2.0, "feed_flow": 2.0}
    fixed_counts = {"feed_flow": 2}
    feature_modalities = ("temperature", "pressure", "coolant_flow", "level")
    target_modality = {"N1": "temperature", "N2": "pressure", "N3": "coolant_flow", "N5": "level",
                       "A1": "temperature", "A2": "coolant_flow", "A3": "temperature",
                       "A4": "temperature", "A5": "pressure"}

    def __init__(self, spec, rng, mismatch):
        super().__init__(spec, rng, mismatch)
        self.p = perturb(rng, NOM, ["A", "F_in", "F_drain", "P0", "b_PT", "c_PL", "F0_hold",
                                    "F0_react", "Kc"], mismatch)
        self.vap = mismatch * rng.uniform(0.0, 0.004)         # unmodelled vapour-pressure curvature
        self.T, self.Lv = 40.0, 0.2
        self.loc_T = rng.uniform(-0.8, 0.8, self.counts["temperature"])

    def step_slow(self, t_h, dt_h):
        phase, sp = phase_at(t_h, self.p["cycle_h"])
        foul = self.fault_level(t_h)                       # P2: heat-transfer degradation
        offset = 3.0 * foul if phase in ("react", "heat") else 0.0
        self.T = (sp + offset) + (self.T - (sp + offset)) * np.exp(-dt_h / 0.3)
        if phase == "fill":
            self.Lv = min(2.2, self.Lv + self.p["F_in"] * dt_h / self.p["A"])
        elif phase == "drain":
            self.Lv = max(0.2, self.Lv - self.p["F_drain"] * dt_h / self.p["A"])

    def true_window(self, t_h, fs, n, rng):
        p = self.p
        phase, sp = phase_at(t_h, p["cycle_h"])
        dur_h = n / fs / 3600.0
        tt = np.arange(n) / fs / 3600.0
        dT = ar1(rng, 1, n, np.exp(-1.0 / (fs * 60.0)), 0.3)[0]       # reaction-heat fluctuation
        T = self.T + dT
        foul = self.fault_level(t_h)
        F0 = (p["F0_react"] if phase == "react" else p["F0_hold"]) * (1.0 + 0.6 * foul)
        Fc = F0 + p["Kc"] * (T - sp) + ar1(rng, 1, n, 0.9, 0.1)[0]
        F_in = np.full(n, p["F_in"] if phase == "fill" else 0.0)
        slope = (p["F_in"] if phase == "fill" else -p["F_drain"] if phase == "drain" else 0.0) / p["A"]
        Lv = np.clip(self.Lv + slope * tt, 0.0, None)
        P = p["P0"] + p["b_PT"] * (T - p["T0"]) + self.vap * (T - p["T0"]) ** 2 + p["c_PL"] * Lv
        true = {
            "temperature": tile(T, self.counts["temperature"]) + self.loc_T[:, None],
            "pressure": tile(P, self.counts["pressure"]),
            "coolant_flow": tile(Fc, self.counts["coolant_flow"]),
            "level": tile(Lv, self.counts["level"]),
            "feed_flow": tile(F_in, self.counts["feed_flow"]),
        }
        return true, {"phase": phase, "setpoint": sp, "dur_h": dur_h}


class ReactorNominal(Nominal):
    relation_groups = ("R1_mass_balance", "R2_pressure_temperature", "R3_coolant_controller")
    # v2 (B3): relations that a known process fault (P1/P2/P3) shifts consistently across sensors
    fault_explainable = ("R3_coolant_controller",)

    def __init__(self):
        super().__init__()
        self.p = dict(NOM)

    def residuals(self, r, ctx, t_h):
        p = self.p
        n = r["level"].shape[1]
        tt = np.arange(n) / max(n - 1, 1) * ctx["dur_h"]
        slopes = np.polyfit(tt, r["level"].T, 1)[0]                      # per level sensor (m/h)
        F_in = float(np.median(med(r["feed_flow"])))
        drain = p["F_drain"] if ctx["phase"] == "drain" else 0.0
        T = float(med(r["temperature"]).mean())
        Lv = float(med(r["level"]).mean())
        P_exp = p["P0"] + p["b_PT"] * (T - p["T0"]) + p["c_PL"] * Lv
        P_med = med(r["pressure"])
        F0 = p["F0_react"] if ctx["phase"] == "react" else p["F0_hold"]
        return {
            "R1_mass_balance": slopes - (F_in - drain) / p["A"],
            "R2_pressure_temperature": np.concatenate([P_med - P_exp,
                                                       med(r["temperature"]) - (p["T0"] + (P_med.mean() - p["P0"] - p["c_PL"] * Lv) / p["b_PT"])]),
            "R3_coolant_controller": med(r["coolant_flow"]) - (F0 + p["Kc"] * (T - ctx["setpoint"])),
        }

    def soft_sensor(self, r, ctx):
        x = np.array([1.0, float(med(r["temperature"]).mean()) / 80.0, float(med(r["level"]).mean())])
        return x, float(med(r["pressure"]).mean())

    def k2_spoof(self, r, ctx, delta, st):
        """Fake a temperature change `delta` (K), consistent in pressure and coolant flow."""
        out = {m: v.copy() for m, v in r.items()}
        out["temperature"] += delta
        out["pressure"] += self.p["b_PT"] * delta
        out["coolant_flow"] += self.p["Kc"] * delta
        return out
