"""Metrics and statistics (requirements §7). Pure numpy.

Uncertainty is computed by bootstrap over RUNS (windows within a run are not independent).
DeLong's test is provided for paired AUC comparison on the same windows; because windows within a
run are correlated, the run-level bootstrap of the AUC difference is the primary interval.
"""

import math

import numpy as np


def _midrank(x):
    """Average ranks (1-based) with ties, vectorized."""
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    start = end - counts
    return ((start + end - 1) / 2.0 + 1.0)[inv]


def auc(scores, labels):
    """Mann-Whitney AUC with tie handling. Returns nan if a class is missing."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _midrank(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def delong(s_a, s_b, labels):
    """Paired DeLong test (DeLong et al. 1988; fast algorithm of Sun & Xu 2014).
    Returns (auc_a, auc_b, diff, z, two-sided p)."""
    y = np.asarray(labels, dtype=bool)
    S = np.vstack([np.asarray(s_a, float), np.asarray(s_b, float)])
    ok = ~np.isnan(S).any(axis=0)
    S, y = S[:, ok], y[ok]
    pos, neg = S[:, y], S[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        return (float("nan"),) * 5
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in S[:, np.r_[np.where(y)[0], np.where(~y)[0]]]])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float(diff), float("nan"), float("nan")
    z = diff / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(aucs[0]), float(aucs[1]), float(diff), float(z), float(p)


def confusion(decision, labels):
    d = np.asarray(decision, bool)
    y = np.asarray(labels, bool)
    tp, fp = int((d & y).sum()), int((d & ~y).sum())
    fn, tn = int((~d & y).sum()), int((~d & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr, "accuracy": acc}


def bootstrap_runs(run_ids, stat_fn, B=200, seed=0, alpha=0.05):
    """Percentile CI of stat_fn(index_array) resampling whole runs with replacement."""
    run_ids = np.asarray(run_ids)
    uniq, inv = np.unique(run_ids, return_inverse=True)
    by_run = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(B)):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([by_run[k] for k in pick])
        v = stat_fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


class FastAUC:
    """AUC under run-level bootstrap weights in O(N) per replicate.

    Resampling whole runs with replacement is equivalent to weighting each window by how many
    times its run was drawn. Scores are sorted once; each replicate is then a weighted
    Mann-Whitney sum over tie groups (ties count 1/2), identical to `auc` on the concatenated
    resample.
    """

    def __init__(self, scores, labels, runs):
        s = np.asarray(scores, dtype=float)
        ok = ~np.isnan(s)
        self.ok = ok
        _, self.group = np.unique(s[ok], return_inverse=True)
        self.ngroups = int(self.group.max()) + 1 if ok.any() else 0
        self.y = np.asarray(labels, dtype=bool)[ok]
        self.runs = np.asarray(runs)[ok]

    def __call__(self, run_weight):
        w = np.asarray(run_weight, dtype=float)[self.runs]
        wp = np.bincount(self.group, weights=w * self.y, minlength=self.ngroups)
        wn = np.bincount(self.group, weights=w * ~self.y, minlength=self.ngroups)
        P, N = wp.sum(), wn.sum()
        if P == 0 or N == 0:
            return float("nan")
        neg_below = np.cumsum(wn) - wn
        return float((wp * (neg_below + 0.5 * wn)).sum() / (P * N))


def bootstrap_auc(fast_list, n_runs, combine=None, B=200, seed=0, alpha=0.05):
    """Percentile CI over run-level bootstrap replicates for one AUC (or a combination of several,
    e.g. a difference), using the same replicate weights for every term (paired)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(B)):
        wts = np.bincount(rng.integers(n_runs, size=n_runs), minlength=n_runs)
        a = [f(wts) for f in fast_list]
        v = combine(*a) if combine else a[0]
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
