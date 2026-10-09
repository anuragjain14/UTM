"""Campaign configuration and deterministic run list (requirements X-1, X-2, EV-1..EV-3)."""

import copy
import hashlib
import tomllib
from pathlib import Path

import numpy as np

SCENARIOS = ("motor", "reactor", "robot")
MECHANISMS = ("N1", "N2", "N3", "N4", "N5", "N6", "A1", "A2", "A3", "A4", "A5")
ORIGIN = {m: ("natural" if m.startswith("N") else "adversarial") for m in MECHANISMS}
# Mechanisms that need a specific sensor type (N4 mounting: vibration/force; N6 encoder wear).
APPLICABLE = {
    "motor": set(MECHANISMS),
    "reactor": set(MECHANISMS) - {"N4", "N6"},
    "robot": set(MECHANISMS),
}
PROCESS_FAULT = {"motor": "P1_bearing_wear", "reactor": "P2_fouling", "robot": "P3_joint_friction"}

DEFAULTS = {
    "campaign": {
        "name": "unnamed", "kind": "test", "base_seed": 0,
        "window_samples": 512,      # samples per high-rate window (MR-2)
        "eval_every_h": 2.0,        # K (MR-2): one decision window every 2 h of operation
        "warmup_h": 192.0,          # clean commissioning period (fit 2/3 + normalization 1/3)
        "warmup_every_h": 2.0,      # same spacing as operation, so step statistics match
        "mismatch": 1.0,            # generator-vs-nominal model mismatch scale (AC-4)
        "keep_raw": 0,              # raw windows kept per run (MR-4)
    },
    "ranges": {                     # Table 2 of the manuscript
        "horizon_h": [1000.0, 10000.0],
        "onset_h": [100.0, 5000.0],
        "sensors_per_modality": [4, 16],
        "noise_frac": [0.001, 0.10],
    },
    "condition_weights": {"clean": 0.10, "process_only": 0.08, "corruption": 0.76, "combo": 0.06},
    "mechanism_weights": {},        # empty = uniform over applicable mechanisms
    "severity_weights": {"low": 1.0, "medium": 1.0, "high": 1.0},
    "scenarios": {},
}

FS_RANGE = {"motor": [200.0, 1000.0], "reactor": [1.0, 10.0], "robot": [100.0, 500.0]}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_campaign(path) -> dict:
    raw = Path(path).read_bytes()
    camp = _merge(DEFAULTS, tomllib.loads(raw.decode("utf-8")))
    camp["_path"] = str(path)
    camp["_sha256"] = hashlib.sha256(raw).hexdigest()
    if not camp["scenarios"]:
        raise ValueError("campaign defines no [scenarios.*] tables")
    for s in camp["scenarios"]:
        if s not in SCENARIOS:
            raise ValueError(f"unknown scenario {s!r}; expected one of {SCENARIOS}")
    return camp


def _pick(rng, weights: dict):
    keys = sorted(weights)
    p = np.array([float(weights[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _ranges(camp, scen):
    r = _merge(camp["ranges"], {k: v for k, v in camp["scenarios"][scen].items() if k in camp["ranges"]})
    r["fs_hz"] = camp["scenarios"][scen].get("fs_hz", FS_RANGE[scen])
    return r


def build_run_list(camp: dict) -> list:
    """Deterministic list of run specifications. Run i is defined by (campaign file, i) alone."""
    runs, gidx = [], 0
    base = int(camp["campaign"]["base_seed"])
    for s_idx, scen in enumerate(sorted(camp["scenarios"])):
        sc = camp["scenarios"][scen]
        n = int(sc["runs"])
        rg = _ranges(camp, scen)
        mech_w = {m: camp["mechanism_weights"].get(m, 1.0) for m in sorted(APPLICABLE[scen])}
        mech_w = {m: w for m, w in mech_w.items() if w > 0}
        cond_w = _merge(camp["condition_weights"], sc.get("condition_weights", {}))
        for i in range(n):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([base, 7_000 + s_idx, i])))
            kind = _pick(rng, cond_w)
            horizon = float(rng.uniform(*rg["horizon_h"]))
            onset_lo = max(float(rg["onset_h"][0]), camp["campaign"]["warmup_h"] + 4.0)
            onset_hi = min(float(rg["onset_h"][1]), horizon - 200.0)
            mech = knowledge = mode = None
            if kind in ("corruption", "combo"):
                mech = _pick(rng, mech_w)
                if mech == "A4":
                    knowledge = _pick(rng, {"K0": 1, "K1": 1, "K2": 1})
                if mech in ("A1", "A2"):
                    mode = _pick(rng, {"naive": 1, "mimic": 1})
            spec = {
                "run_index": gidx,
                "run_id": f"{camp['campaign']['name']}-{scen}-{i:05d}",
                "scenario": scen,
                "kind": kind,
                "mechanism": mech,
                "origin": ORIGIN.get(mech, "none") if mech else "none",
                "attacker_knowledge": knowledge,
                "attack_mode": mode,
                "severity": _pick(rng, camp["severity_weights"]) if mech else None,
                "process_fault": PROCESS_FAULT[scen] if kind in ("process_only", "combo") else None,
                "horizon_h": round(horizon, 3),
                "onset_h": round(float(rng.uniform(onset_lo, onset_hi)), 3) if mech else None,
                "fault_onset_h": round(float(rng.uniform(onset_lo, onset_hi)), 3) if kind in ("process_only", "combo") else None,
                "fs_hz": round(float(rng.uniform(*rg["fs_hz"])), 3),
                "sensors_per_modality": int(rng.integers(rg["sensors_per_modality"][0], rg["sensors_per_modality"][1] + 1)),
                "noise_frac": round(float(np.exp(rng.uniform(np.log(rg["noise_frac"][0]), np.log(rg["noise_frac"][1])))), 6),
            }
            runs.append(spec)
            gidx += 1
    return runs


def shard(runs: list, spec: str) -> list:
    """`--shard i/N` (1-based i): runs whose position mod N == i-1 (requirement X-3)."""
    i, n = (int(x) for x in spec.split("/"))
    if not (1 <= i <= n):
        raise ValueError(f"bad shard {spec!r}")
    return [r for k, r in enumerate(runs) if k % n == i - 1]
