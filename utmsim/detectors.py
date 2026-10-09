"""Detectors (requirements §6). All are fitted only on clean commissioning (warm-up) windows of the
same run and never receive labels (requirement T-6).

Proposed framework: CrossModal (Algorithm 1), Physics (Algorithm 2), GatedLearner (Algorithm 3).
Baselines: Security (B2 path), Reliability (B3 path), Cusum (B7), PCAResidual (B6),
Isolation Forest / One-class SVM (B4/B5, scikit-learn if installed).
"""

import numpy as np

try:  # optional dependency
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


def robust_std(x, axis=0):
    x = np.asarray(x, dtype=float)
    mad = np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)
    return 1.4826 * mad


def channels(readings, modalities):
    return np.concatenate([readings[m] for m in modalities], axis=0)


def corr(X):
    """Correlation matrix robust to constant channels (constant -> zero correlation)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = np.sqrt((Xc ** 2).mean(axis=1))
    ok = sd > 1e-12
    Z = np.zeros_like(Xc)
    Z[ok] = Xc[ok] / sd[ok, None]
    return (Z @ Z.T) / X.shape[1]


def group_dev(meds, groups):
    """Each sensor's deviation from the median of its redundancy group (0 if alone)."""
    d = np.zeros_like(meds)
    for g in groups:
        if len(g) > 1:
            d[g] = meds[g] - np.median(meds[g])
    return d


def features(readings, modalities, groups):
    """Window feature vector for learned baselines: per modality level, spread, and the largest
    deviation of one sensor from its redundancy-group median."""
    out = []
    for m in modalities:
        x = readings[m]
        meds = np.median(x, axis=1)
        out += [float(meds.mean()), float(x.std(axis=1).mean()), float(np.max(np.abs(group_dev(meds, groups[m]))))]
    return np.array(out)


# --------------------------------------------------------------------------------------------
class CrossModal:
    """Algorithm 1: score = sum_{i<j} |C_current - C_baseline| / sigma (paper Sec. 3.3)."""

    def __init__(self, modalities):
        self.mods = modalities

    def fit(self, windows):
        Cs = np.stack([corr(channels(w, self.mods)) for w in windows])
        self.C = Cs.mean(axis=0)
        self.sig = np.maximum(Cs.std(axis=0), 0.02)
        self.iu = np.triu_indices(self.C.shape[0], k=1)

    def score(self, w):
        d = np.abs(corr(channels(w, self.mods)) - self.C) / self.sig
        return float(d[self.iu].mean())


class Physics:
    """Algorithm 2: weighted normalized constraint violations from the NOMINAL relations."""

    def __init__(self, groups):
        self.groups = groups

    def fit(self, residual_dicts):
        self.c, self.s = {}, {}
        for g in self.groups:
            R = np.stack([r[g] for r in residual_dicts])
            self.c[g] = np.median(R, axis=0)
            s = robust_std(R, axis=0)
            self.s[g] = np.maximum(s, max(0.1 * float(np.median(s)), 1e-9))

    def score(self, res):
        per = {g: float(np.max(np.abs(res[g] - self.c[g]) / self.s[g])) for g in self.groups}
        return float(np.mean(list(per.values()))), per


class GatedLearner:
    """Algorithm 3: soft-sensor linear model, normalized-LMS updates with an EWC penalty
    lambda * sum F_i (theta_i - theta*_i)^2 (Kirkpatrick et al. 2017). Two copies run side by
    side: `gated` updates only when the a-priori gate rule passes; `ungated` (B0) always updates."""

    LAMBDA = 1.0
    LR = 0.05
    GATE_Z = 3.0          # a-priori gate rule: mean(z_crossmodal, z_physics) < 3 (fixed before runs)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        A = X.T @ X + 1e-6 * np.eye(X.shape[1])
        self.theta_star = np.linalg.solve(A, X.T @ y)
        self.F = np.diag(A) / len(y)
        self.F = self.F / max(self.F.max(), 1e-12)
        self.gated = self.theta_star.copy()
        self.ungated = self.theta_star.copy()
        r = y - X @ self.theta_star
        self.res_c, self.res_s = float(np.median(r)), float(max(robust_std(r), 1e-9))

    def _update(self, th, x, y):
        g = (x @ th - y) * x / (x @ x + 1e-12) + self.LAMBDA * self.F * (th - self.theta_star)
        return th - self.LR * g

    def score(self, x, y):
        return abs((y - x @ self.gated) - self.res_c) / self.res_s

    def step(self, x, y, gate_ok):
        if gate_ok:
            self.gated = self._update(self.gated, x, y)
        self.ungated = self._update(self.ungated, x, y)

    def errors(self, x_clean, y_clean):
        return float((x_clean @ self.gated - y_clean) ** 2), float((x_clean @ self.ungated - y_clean) ** 2)


# --------------------------------------------------------------------------------------------
CUSUM_K, CUSUM_H = 0.5, 5.0        # standard choices (allowance 0.5 sigma, decision interval 5 sigma)


def _cusum_step(obj, z):
    """Two-sided Page CUSUM per sensor; returns the max statistic, then resets sensors that
    crossed the decision interval (as an operator would after acknowledging an alarm)."""
    obj.sp = np.maximum(0.0, obj.sp + z - CUSUM_K)
    obj.sn = np.maximum(0.0, obj.sn - z - CUSUM_K)
    stat = np.maximum(obj.sp, obj.sn)
    out = float(np.max(stat))
    alarm = stat > CUSUM_H
    obj.sp[alarm] = 0.0
    obj.sn[alarm] = 0.0
    return out


class Security:
    """B2 path: attack-oriented checks: step/spike between consecutive windows and replay
    (near-duplicate of a stored historical window)."""

    N_STORE = 64              # history buffer: all fit windows

    def __init__(self, modalities):
        self.mods = modalities

    def fit(self, windows):
        meds = np.stack([np.concatenate([np.median(w[m], axis=1) for m in self.mods]) for w in windows])
        self.step_s = np.maximum(robust_std(np.diff(meds, axis=0), axis=0), 1e-9)
        self.prev = meds[-1]
        keep = windows[-self.N_STORE:]
        Xs = np.stack([channels(w, self.mods) for w in keep])               # (k, ch, n)
        Xs = Xs - Xs.mean(axis=2, keepdims=True)
        self.store = Xs / np.maximum(np.linalg.norm(Xs, axis=2, keepdims=True), 1e-12)

    def score(self, w):
        med = np.concatenate([np.median(w[m], axis=1) for m in self.mods])
        step = float(np.max(np.abs(med - self.prev) / self.step_s))
        self.prev = med
        X = channels(w, self.mods)
        X = X - X.mean(axis=1, keepdims=True)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        sim = np.einsum("kcn,cn->kc", self.store, X)                         # per stored window, channel
        replay = float(np.max(sim))                    # any channel matching a stored window
        return step, replay


class Reliability:
    """B3 path: redundant-sensor median voting + CUSUM on each sensor's deviation from its
    redundancy-group median (robust to legitimate operating-point changes)."""

    def __init__(self, modalities, groups):
        self.mods = modalities
        self.groups = groups

    def _dev(self, w):
        return np.concatenate([group_dev(np.median(w[m], axis=1), self.groups[m]) for m in self.mods])

    def fit(self, windows):
        D = np.stack([self._dev(w) for w in windows])
        self.c = np.median(D, axis=0)
        self.s = np.maximum(robust_std(D, axis=0), 1e-9)
        self.sp = np.zeros(D.shape[1])
        self.sn = np.zeros(D.shape[1])

    def score(self, w):
        z = (self._dev(w) - self.c) / self.s
        return float(np.max(np.abs(z))), _cusum_step(self, z)


class Cusum:
    """B7: classical per-sensor two-sided CUSUM on standardized window medians."""

    def __init__(self, modalities):
        self.mods = modalities

    def _med(self, w):
        return np.concatenate([np.median(w[m], axis=1) for m in self.mods])

    def fit(self, windows):
        M = np.stack([self._med(w) for w in windows])
        self.c = np.median(M, axis=0)
        self.s = np.maximum(robust_std(M, axis=0), 1e-9)
        self.sp = np.zeros(M.shape[1])
        self.sn = np.zeros(M.shape[1])

    def score(self, w):
        z = (self._med(w) - self.c) / self.s
        return _cusum_step(self, z)


class Learned:
    """B4 Isolation Forest, B5 One-class SVM, B6 PCA residual (Q statistic), on window features."""

    def __init__(self, seed):
        self.seed = int(seed)

    def fit(self, F):
        F = np.asarray(F)
        self.mu = F.mean(axis=0)
        self.sd = np.maximum(F.std(axis=0), 1e-9)
        Z = (F - self.mu) / self.sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        var = S ** 2 / max((S ** 2).sum(), 1e-12)
        k = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
        self.P = Vt[:k].T
        self.iforest = self.ocsvm = None
        if HAVE_SKLEARN:
            self.iforest = IsolationForest(n_estimators=100, random_state=self.seed).fit(Z)
            self.ocsvm = OneClassSVM(nu=0.05, gamma="scale").fit(Z)

    def score(self, f):
        z = (np.asarray(f) - self.mu) / self.sd
        q = float(np.sum((z - self.P @ (self.P.T @ z)) ** 2))
        if self.iforest is None:
            return q, float("nan"), float("nan")
        z2 = z[None, :]
        return q, float(-self.iforest.score_samples(z2)[0]), float(-self.ocsvm.decision_function(z2)[0])
