"""PL-3: 6-DOF robotic assembly cell with joint encoders, joint motor currents, a wrist
force/torque sensor and a vision system measuring the end-effector position. The arm runs a
periodic pick-and-place program; the program (and payload) changes every few hours."""

import numpy as np

from ..sensors import ar1
from .base import Nominal, Plant, med, perturb, tile

G = 9.81
NOM = {
    "L2": 0.42, "L3": 0.39, "h0": 0.16,       # m (link lengths, base height)
    "m2": 8.0, "m3": 4.0,                      # kg
    "kt": 3.0,                                 # Nm per A at the joint (motor constant x gear ratio)
    "J": 0.4,                                  # kg m^2 effective joint inertia (joints 2,3)
    "f_c": 1.0,                                # Nm Coulomb friction
    "cycle_s": 8.0,
}
N_JOINTS = 6


def fk(q, p):
    """Position of the end effector from joints 1-3 (base yaw, shoulder, elbow)."""
    r = p["L2"] * np.cos(q[1]) + p["L3"] * np.cos(q[1] + q[2])
    return np.stack([np.cos(q[0]) * r, np.sin(q[0]) * r,
                     p["h0"] + p["L2"] * np.sin(q[1]) + p["L3"] * np.sin(q[1] + q[2])])


def gravity(q, p, payload):
    t2 = ((p["m2"] * p["L2"] / 2 + p["m3"] * p["L2"] + payload * p["L2"]) * G * np.cos(q[1])
          + (p["m3"] * p["L3"] / 2 + payload * p["L3"]) * G * np.cos(q[1] + q[2]))
    t3 = (p["m3"] * p["L3"] / 2 + payload * p["L3"]) * G * np.cos(q[1] + q[2])
    return t2, t3


class RobotPlant(Plant):
    name = "robot"
    scales = {"position": 1.0, "current": 20.0, "force": 20.0, "vision": 0.5}
    spans = {"position": 0.05, "current": 20.0, "force": 20.0, "vision": 0.05}
    feature_modalities = ("position", "current", "force", "vision")
    target_modality = {"N1": "vision", "N2": "force", "N3": "current", "N4": "force",
                       "N5": "position", "N6": "position", "A1": "vision", "A2": "current",
                       "A3": "position", "A4": "position", "A5": "force"}

    def __init__(self, spec, rng, mismatch):
        n = int(spec["sensors_per_modality"])
        self.fixed_counts = {"position": max(N_JOINTS, n), "current": max(N_JOINTS, n),
                             "force": 6, "vision": 3 * max(1, n // 6)}
        super().__init__(spec, rng, mismatch)
        self.p = perturb(rng, NOM, ["L2", "L3", "h0", "m2", "m3", "kt", "J", "f_c"], mismatch, rel=0.05)
        self.q_off = mismatch * rng.normal(0.0, 0.003, N_JOINTS)        # unmodelled joint offsets
        self.visc = mismatch * rng.uniform(0.0, 0.5)                    # unmodelled viscous friction
        self.friction_gain = 1.0
        self.joint_of = {"position": np.arange(self.counts["position"]) % N_JOINTS,
                         "current": np.arange(self.counts["current"]) % N_JOINTS}
        self.new_program(0.0)

    def groups(self):
        g = {m: [np.where(self.joint_of[m] == j)[0] for j in range(N_JOINTS)] for m in ("position", "current")}
        g["vision"] = [np.arange(a, self.counts["vision"], 3) for a in range(3)]   # x, y, z axes
        g["force"] = [np.array([k]) for k in range(6)]                           # no redundancy
        return g

    def new_program(self, t_h):
        r = self.rng
        self.q0 = np.array([0.0, 0.6, -0.9, 0.0, 0.3, 0.0]) + r.uniform(-0.15, 0.15, N_JOINTS)
        self.amp = r.uniform(0.15, 0.45, N_JOINTS)
        self.phi = r.uniform(0, 2 * np.pi, N_JOINTS)
        self.payload = float(r.uniform(0.5, 3.0))
        self.t_next = t_h + float(r.exponential(12.0))

    def step_slow(self, t_h, dt_h):
        if t_h >= self.t_next:
            self.new_program(t_h)
        self.friction_gain = 1.0 + 2.5 * self.fault_level(t_h)           # P3 joint friction

    def trajectory(self, t_s):
        w = 2 * np.pi / self.p["cycle_s"]
        arg = w * t_s[None, :] + self.phi[:, None]
        q = self.q0[:, None] + self.amp[:, None] * np.sin(arg)
        qd = self.amp[:, None] * w * np.cos(arg)
        qdd = -self.amp[:, None] * w ** 2 * np.sin(arg)
        holding = (np.sin(w * t_s) > 0).astype(float)
        return q, qd, qdd, holding

    def true_window(self, t_h, fs, n, rng):
        p = self.p
        t_s = (t_h * 3600.0) + np.arange(n) / fs
        q, qd, qdd, hold = self.trajectory(t_s)
        payload = self.payload * hold
        t2, t3 = gravity(q + self.q_off[:, None], p, payload)
        tau = p["J"] * qdd + p["f_c"] * self.friction_gain * np.tanh(qd / 0.05) + self.visc * qd
        tau[1] += t2
        tau[2] += t3
        cur = tau / p["kt"]
        pos = fk(q + self.q_off[:, None], p)
        contact = (hold * (rng.random(n) < 0.02)) * rng.normal(0, 3.0, n)   # insertion contacts
        force = np.zeros((6, n))
        force[2] = -payload * G + contact
        force[:2] = ar1(rng, 2, n, 0.5, 0.2)
        force[3:] = ar1(rng, 3, n, 0.5, 0.05)
        true = {
            "position": q[self.joint_of["position"]],
            "current": cur[self.joint_of["current"]],
            "force": force,
            "vision": np.tile(pos, (self.counts["vision"] // 3, 1)),
        }
        ctx = {"t_s": t_s, "holding": hold, "payload": self.payload, "fs": fs}
        return true, ctx


class RobotNominal(Nominal):
    relation_groups = ("R1_kinematics", "R2_dynamics", "R3_payload")

    def __init__(self, joint_of):
        super().__init__()
        self.p = dict(NOM)
        self.joint_of = joint_of

    def joints(self, r):
        enc = r["position"]
        return np.stack([np.median(enc[self.joint_of["position"] == j], axis=0) for j in range(N_JOINTS)])

    def residuals(self, r, ctx, t_h):
        p = self.p
        q = self.joints(r)
        fs = ctx["fs"]
        qd = np.gradient(q, 1.0 / fs, axis=1)
        qdd = np.gradient(qd, 1.0 / fs, axis=1)
        k = max(1, int(fs * 0.05))
        kern = np.ones(k) / k
        qdd = np.stack([np.convolve(row, kern, mode="same") for row in qdd])
        pos = fk(q, p)
        vis = r["vision"]
        kin = np.median(vis - np.tile(pos, (vis.shape[0] // 3, 1)), axis=1)
        payload = ctx["payload"] * ctx["holding"]
        t2, t3 = gravity(q, p, payload)
        tau = p["J"] * qdd + p["f_c"] * np.tanh(qd / 0.05)
        tau[1] += t2
        tau[2] += t3
        cur_nom = tau / p["kt"]
        cur = r["current"]
        dyn = np.median(cur - cur_nom[self.joint_of["current"]], axis=1)
        fz = r["force"][2]
        hold = ctx["holding"] > 0.5
        pay = np.array([np.median(fz[hold] + ctx["payload"] * G) if hold.any() else 0.0,
                        np.median(fz[~hold]) if (~hold).any() else 0.0])
        return {"R1_kinematics": kin, "R2_dynamics": dyn, "R3_payload": pay}

    def soft_sensor(self, r, ctx):
        q = self.joints(r)
        x = np.array([1.0, float(np.mean(np.cos(q[1]))), float(np.mean(np.cos(q[1] + q[2]))),
                      float(np.mean(ctx["holding"]))])
        cur2 = r["current"][self.joint_of["current"] == 1]
        return x, float(np.sqrt(np.mean(cur2 ** 2)))

    def k2_spoof(self, r, ctx, delta, st):
        """Fake a shoulder-joint offset `delta` (rad) consistent in encoders, vision (nominal FK)
        and joint currents (nominal gravity)."""
        p = self.p
        out = {m: v.copy() for m, v in r.items()}
        q = self.joints(r)
        q2 = q.copy()
        q2[1] += delta
        out["position"][self.joint_of["position"] == 1] += delta
        dpos = fk(q2, p) - fk(q, p)
        out["vision"] += np.tile(dpos, (out["vision"].shape[0] // 3, 1))
        payload = ctx["payload"] * ctx["holding"]
        a2, a3 = gravity(q, p, payload)
        b2, b3 = gravity(q2, p, payload)
        out["current"][self.joint_of["current"] == 1] += (b2 - a2) / p["kt"]
        out["current"][self.joint_of["current"] == 2] += (b3 - a3) / p["kt"]
        return out
