"""One simulation run (requirements MR-1..MR-5, §6, D-6).

Phases per run:
  1. commissioning windows (clean): first 2/3 fit every detector, last 1/3 give the robust
     normalization (median/MAD) that turns every raw score into a z-score;
  2. operation windows: corruption and process faults may be active; every architecture scores
     the SAME window (paired design, G5); the learner updates online.
Only operation windows are written out.
"""

import resource
import time
from pathlib import Path

import numpy as np

from . import __version__
from .corruption import Corruption
from .detectors import (HAVE_SKLEARN, CrossModal, Cusum, GatedLearner, Learned, Physics,
                        Reliability, Security, features, robust_std)
from .io import environment, git_commit, manifest, write_csv_gz, write_json
from .plants import make
from .rng import run_streams, stream_seeds

SCORE_KEYS = ["cm", "phys", "learn", "sec_step", "sec_replay", "rel_vote", "rel_cusum", "cusum",
              "pca", "iforest", "ocsvm"]


def _raw_scores(det, r, res, x, y, f):
    t = {}
    out = {}
    t0 = time.perf_counter_ns(); out["cm"] = det["cm"].score(r); t["cm"] = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns(); out["phys"], per = det["phys"].score(res); t["phys"] = time.perf_counter_ns() - t0
    for g, v in per.items():
        out["phys_" + g] = v
    t0 = time.perf_counter_ns(); out["learn"] = det["learn"].score(x, y); t["learn"] = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns(); out["sec_step"], out["sec_replay"] = det["sec"].score(r); t["sec"] = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns(); out["rel_vote"], out["rel_cusum"] = det["rel"].score(r); t["rel"] = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns(); out["cusum"] = det["cusum"].score(r); t["cusum"] = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns(); out["pca"], out["iforest"], out["ocsvm"] = det["learned"].score(f); t["learned"] = time.perf_counter_ns() - t0
    return out, t


def run_one(spec, camp, out_dir, repo_dir=None):
    t_start = time.time()
    c = camp["campaign"]
    rs = run_streams(c["base_seed"], spec["run_index"])
    plant, nominal = make(spec, rs["plant"], c["mismatch"])
    corr = Corruption(spec, plant, nominal, rs["corruption"]) if spec["mechanism"] else None
    fs, n = float(spec["fs_hz"]), int(c["window_samples"])
    warm = np.arange(0.0, c["warmup_h"], c["warmup_every_h"])
    oper = np.arange(c["warmup_h"], spec["horizon_h"], c["eval_every_h"])
    n_fit = int(round(len(warm) * 2 / 3))
    if corr:
        corr.choose_record_time(warm)
    mods = plant.feature_modalities
    groups = plant.groups()

    t_slow = 0.0

    def advance(t):
        nonlocal t_slow
        while t_slow + 1.0 <= t + 1e-9:
            t_slow += 1.0
            plant.step_slow(t_slow, 1.0)

    def window(t):
        advance(t)
        true, ctx = plant.true_window(t, fs, n, rs["window"])
        clean = plant.measure(true, rs["sensors"])
        if corr:
            obs, comp = corr.apply(clean, ctx, t)
        else:
            obs, comp = clean, False
        return clean, obs, ctx, comp

    # ---- commissioning -----------------------------------------------------------------------
    W, RES, XY, FEAT = [], [], [], []
    for t in warm:
        clean, obs, ctx, _ = window(t)
        W.append(obs)
        RES.append(nominal.residuals(obs, ctx, t))
        XY.append(nominal.soft_sensor(obs, ctx))
        FEAT.append(features(obs, mods, groups))
    det = {"cm": CrossModal(mods), "phys": Physics(nominal.relation_groups), "learn": GatedLearner(),
           "sec": Security(mods), "rel": Reliability(mods, groups), "cusum": Cusum(mods),
           "learned": Learned(rs["detectors"].integers(2 ** 31))}
    det["cm"].fit(W[:n_fit])
    det["phys"].fit(RES[:n_fit])
    det["learn"].fit([xy[0] for xy in XY[:n_fit]], [xy[1] for xy in XY[:n_fit]])
    det["sec"].fit(W[:n_fit])
    det["rel"].fit(W[:n_fit])
    det["cusum"].fit(W[:n_fit])
    det["learned"].fit(FEAT[:n_fit])
    norm_raw = [_raw_scores(det, W[k], RES[k], *XY[k], FEAT[k])[0] for k in range(n_fit, len(warm))]
    keys = list(norm_raw[0])
    norm = {}
    for k in keys:
        v = np.array([d[k] for d in norm_raw], dtype=float)
        if np.all(np.isnan(v)):
            norm[k] = (np.nan, np.nan)
        else:
            norm[k] = (float(np.nanmedian(v)), float(max(robust_std(v[~np.isnan(v)]), 1e-9)))
    del W, RES, XY, FEAT

    # ---- operation -------------------------------------------------------------------------
    rows, trows, raw_keep = [], [], []
    onset = spec["onset_h"]
    for idx, t in enumerate(oper):
        clean, obs, ctx, comp = window(t)
        res = nominal.residuals(obs, ctx, t)
        x, y = nominal.soft_sensor(obs, ctx)
        f = features(obs, mods, groups)
        raw, tm = _raw_scores(det, obs, res, x, y, f)
        z = {k: (raw[k] - norm[k][0]) / norm[k][1] for k in keys}
        gate_ok = bool(np.mean([z["cm"], z["phys"]]) < GatedLearner.GATE_Z)
        xc, yc = nominal.soft_sensor(clean, ctx)
        e_g, e_u = det["learn"].errors(xc, yc)
        det["learn"].step(x, y, gate_ok)
        row = {
            "run_id": spec["run_id"], "idx": idx, "t_h": float(t),
            "compromised": bool(comp),
            "origin": spec["origin"] if comp else "none",
            "mechanism": spec["mechanism"] if comp else "none",
            "process_fault": bool(plant.fault_onset is not None and t >= plant.fault_onset),
            "hours_since_onset": (float(t) - onset) if (onset is not None and t >= onset) else None,
            "gate_ok": gate_ok, "err_gated": e_g, "err_ungated": e_u,
        }
        for k in keys:
            row["z_" + k] = z[k]
        for j, v in enumerate(f):
            row[f"f{j}"] = float(v)
        rows.append(row)
        trows.append({"idx": idx, **{f"ns_{k}": v for k, v in tm.items()}})
        if len(raw_keep) < int(c["keep_raw"]) and (onset is not None and t >= onset):
            raw_keep.append((t, obs))

    # ---- outputs -----------------------------------------------------------------------------
    run_dir = Path(out_dir) / spec["scenario"] / spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    header = (["run_id", "idx", "t_h", "compromised", "origin", "mechanism", "process_fault",
               "hours_since_onset", "gate_ok", "err_gated", "err_ungated"]
              + ["z_" + k for k in keys] + [f"f{j}" for j in range(len(rows[0]) - 11 - len(keys))])
    write_csv_gz(run_dir / "windows.csv.gz", header, rows)
    write_csv_gz(run_dir / "timing.csv.gz", list(trows[0]), trows)
    files = [run_dir / "windows.csv.gz", run_dir / "timing.csv.gz"]
    if raw_keep:
        np.savez_compressed(run_dir / "raw.npz", **{f"t{t:.0f}_{m}": v for t, r in raw_keep for m, v in r.items()})
        files.append(run_dir / "raw.npz")
    summary = {
        "spec": spec,
        "campaign": {"name": c["name"], "kind": c["kind"], "sha256": camp["_sha256"],
                     "mismatch": c["mismatch"], "window_samples": n, "eval_every_h": c["eval_every_h"],
                     "warmup_h": c["warmup_h"]},
        "seeds": stream_seeds(c["base_seed"], spec["run_index"]),
        "n_windows": len(rows), "n_compromised": int(sum(r["compromised"] for r in rows)),
        "corruption": corr.label() if corr else None,
        "sensor_counts": plant.counts,
        "sklearn_baselines": HAVE_SKLEARN,
        "simulator_version": __version__,
    }
    write_json(run_dir / "summary.json", summary)
    files.append(run_dir / "summary.json")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if environment()["platform"].startswith("macOS") else rss / 1024
    manifest(run_dir, files, {
        "run_id": spec["run_id"], "code_commit": git_commit(repo_dir or Path(__file__).resolve().parents[1]),
        "environment": environment(), "wall_s": round(time.time() - t_start, 3),
        "peak_rss_mb": round(rss_mb, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return run_dir
