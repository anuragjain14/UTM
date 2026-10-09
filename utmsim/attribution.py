"""E-ATTR (requirements MP-2): can a classifier tell natural from adversarial corruption, using only
what the defender observes? Per corrupted run, features summarize the post-onset trajectory of the
window features and detector z-scores (level shift, trend, daily periodicity, spread). Pairs:

    N1 drift      vs A1 gradual bias (naive)   and  (mimic)
    N2 calibration vs A5 FDI step
    N3 noise growth vs A2 noise amplification (naive)  and  (mimic)

Classifier: L2 logistic regression (numpy), 5-fold stratified CV; plus scikit-learn gradient boosting
if installed. Accuracy with a Wilson 95% CI over runs.
"""

import numpy as np

from .metrics import wilson

PAIRS = [("N1", "A1", "naive"), ("N1", "A1", "mimic"), ("N2", "A5", None),
         ("N3", "A2", "naive"), ("N3", "A2", "mimic")]
MIN_PER_CLASS = 8


def run_features(d, k):
    m = (d["run"] == k) & np.isfinite(d["hours_since_onset"])
    if m.sum() < 6:
        return None
    hs = d["hours_since_onset"][m].astype(float)
    th = d["t_h"][m].astype(float)
    cols = sorted(c for c in d if c.startswith("f") and c[1:].isdigit()) + sorted(c for c in d if c.startswith("z_"))
    pre = (d["run"] == k) & ~np.isfinite(d["hours_since_onset"])
    feats = []
    X = np.column_stack([np.ones_like(hs), hs, np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24)])
    for c in cols:
        v = d[c][m].astype(float)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0)
        base = d[c][pre].astype(float)
        base = base[np.isfinite(base)]
        mu = float(base.mean()) if len(base) else 0.0
        sd = float(base.std()) if len(base) > 2 else 1.0
        sd = sd if sd > 1e-9 else 1.0
        z = (v - mu) / sd
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        resid = z - X @ coef
        feats += [z.mean(), coef[1] * 1000.0, np.hypot(coef[2], coef[3]), resid.std(),
                  z[: max(1, len(z) // 4)].mean() - z[-max(1, len(z) // 4):].mean()]
    return np.array(feats, dtype=float)


def _logreg_cv(X, y, folds=5, seed=0, l2=1.0, iters=400, lr=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        c = idx[y == cls]
        rng.shuffle(c)
        fold[c] = np.arange(len(c)) % folds
    pred = np.empty(len(y), dtype=int)
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        A = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        w = np.zeros(A.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
            g = A.T @ (p - y[tr]) / len(p) + l2 * np.r_[0, w[1:]] / len(p)
            w -= lr * g
        At = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd])
        pred[te] = (At @ w > 0).astype(int)
    return pred


def attribution_table(d, specs):
    rows = []
    for nat, adv, mode in PAIRS:
        Xs, ys = [], []
        for k, s in enumerate(specs):
            sp = s["spec"]
            if sp["mechanism"] == nat or (sp["mechanism"] == adv and (mode is None or sp["attack_mode"] == mode)):
                f = run_features(d, k)
                if f is not None:
                    Xs.append(f)
                    ys.append(1 if sp["mechanism"] == adv else 0)
        y = np.array(ys)
        label = f"{nat} vs {adv}" + (f" ({mode})" if mode else "")
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        base = {"pair": label, "n_natural": n0, "n_adversarial": n1,
                "chance": max(n0, n1) / max(n0 + n1, 1)}
        if n0 < MIN_PER_CLASS or n1 < MIN_PER_CLASS:
            rows.append({**base, "classifier": f"insufficient runs (< {MIN_PER_CLASS} per class)",
                         "accuracy": float("nan"), "acc_lo": float("nan"), "acc_hi": float("nan")})
            continue
        X = np.nan_to_num(np.vstack(Xs))
        pred = _logreg_cv(X, y)
        k = int((pred == y).sum())
        lo, hi = wilson(k, len(y))
        rows.append({**base, "classifier": "logistic regression (5-fold CV)", "accuracy": k / len(y),
                     "acc_lo": lo, "acc_hi": hi})
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            pg = cross_val_predict(GradientBoostingClassifier(random_state=0), X, y,
                                   cv=StratifiedKFold(5, shuffle=True, random_state=0))
            k2 = int((pg == y).sum())
            lo2, hi2 = wilson(k2, len(y))
            rows.append({**base, "classifier": "gradient boosting (5-fold CV)", "accuracy": k2 / len(y),
                         "acc_lo": lo2, "acc_hi": hi2})
        except Exception:
            pass
    return rows
