"""Revised architecture v2 (follow-up B): scoring and calibration.

  B1  temporal accumulation: per-run EWMA of the composite gate score, s_t = lam*x_t + (1-lam)*s_{t-1},
      s_0 = 0 (the commissioning median of every z-score);
  B2  hybrid gate: the composite adds the redundancy-vote deviation and the PCA residual to the
      cross-modal, physics and learner scores;
  B3  fault-aware physics (`z_phys_fa`, see detectors.PhysicsFA) replaces the v1 physics score.

Weights (simplex grid, step 1/4) and lam are chosen on the v2 CALIBRATION campaign only.
Ablations are tuned the same way, each on its own grid, so every configuration gets its best shot.
"""

from itertools import product

import numpy as np

from .metrics import auc

LAMBDAS = [1.0, 0.5, 0.2, 0.1, 0.05]
V2_INPUTS = ["cm", "phys_fa", "learn2", "rel_vote", "pca"]
VARIANTS = {                                   # name -> (inputs, lambdas searched)
    "proposed_v2": (V2_INPUTS, LAMBDAS),
    "v2_no_accum": (V2_INPUTS, [1.0]),
    "v2_no_fa": (["cm", "phys", "learn2", "rel_vote", "pca"], LAMBDAS),
    "v2_no_hybrid": (["cm", "phys_fa", "learn2"], LAMBDAS),
}


def has_v2(d):
    return "z_phys_fa" in d


def simplex(n, steps=4):
    """All weight vectors of length n with entries in {0, 1/steps, ..., 1} summing to 1."""
    return [tuple(c / steps for c in v) for v in product(range(steps + 1), repeat=n) if sum(v) == steps]


def ewma_by_run(x, runs, lam):
    """EWMA restarted at every run. Rows of one run must be contiguous and in time order (as
    loaded by report.load). Vectorized across runs: one numpy step per time index."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    if lam >= 1.0:
        return x
    runs = np.asarray(runs)
    starts = np.flatnonzero(np.r_[True, runs[1:] != runs[:-1]])
    lens = np.diff(np.r_[starts, len(x)])
    pos = np.arange(len(x)) - np.repeat(starts, lens)
    row = np.repeat(np.arange(len(starts)), lens)
    M = np.zeros((len(starts), int(lens.max())))
    M[row, pos] = x
    S = np.empty_like(M)
    s = np.zeros(len(starts))
    for t in range(M.shape[1]):
        s = lam * M[:, t] + (1.0 - lam) * s
        S[:, t] = s
    return S[row, pos]


def _components(d, inputs, lam, cache):
    out = []
    for k in inputs:
        if (k, lam) not in cache:
            cache[(k, lam)] = ewma_by_run(d["z_" + k], d["run"], lam)
        out.append(cache[(k, lam)])
    return out


def score(d, cfg, cache=None):
    """Composite of one v2 configuration. EWMA is linear, so EWMA(sum w_i z_i) = sum w_i EWMA(z_i)."""
    comps = _components(d, cfg["inputs"], cfg["lambda"], {} if cache is None else cache)
    return sum(w * c for w, c in zip(cfg["w"], comps))


def calibrate_v2(d, y):
    """Best (weights, lambda) per variant by window-level AUC on calibration data."""
    cache, out = {}, {}
    for name, (inputs, lams) in VARIANTS.items():
        best = None
        for lam in lams:
            comps = _components(d, inputs, lam, cache)
            for w in simplex(len(inputs)):
                a = auc(sum(wi * c for wi, c in zip(w, comps)), y)
                if np.isfinite(a) and (best is None or a > best[0]):
                    best = (a, list(w), lam)
        out[name] = {"inputs": inputs, "w": best[1], "lambda": best[2], "calibration_auc": best[0]}
    return out


def architectures_v2(d, cfgs):
    cache = {}
    return {name: score(d, cfg, cache) for name, cfg in cfgs.items()}
