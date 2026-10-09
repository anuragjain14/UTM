"""Calibration and reporting: every number the paper reports is produced here, from stored runs.

  calibrate  -> thresholds.json   (gate weights + per-architecture thresholds at a fixed FPR,
                                    from CALIBRATION runs only; requirements AC-3, G4)
  report     -> tables (CSV + report.md) for outputs O1, O5-O10, O12-O14
"""

import json
from pathlib import Path

import numpy as np

from .io import read_csv_gz
from .metrics import auc, bootstrap_runs, confusion, delong, wilson

BASE_COLS = ["compromised", "process_fault", "hours_since_onset", "gate_ok", "err_gated", "err_ungated", "t_h"]
GROUP_COLS = {"motor": ["R1_current_speed", "R2_acoustic_vibration", "R3_thermal", "R4_vibration_speed"],
              "reactor": ["R1_mass_balance", "R2_pressure_temperature", "R3_coolant_controller"],
              "robot": ["R1_kinematics", "R2_dynamics", "R3_payload"]}
WEIGHT_GRID = [w for w in [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]]
BASELINES = ["B1_separated", "B2_security", "B3_reliability", "B4_iforest", "B5_ocsvm", "B6_pca", "B7_cusum"]
ABLATION = ["cm_only", "phys_only", "learn_only", "gate_cm_phys", "proposed"]


# ---------------------------------------------------------------------------------------------
def load(results_dir, scenario):
    """Load all runs of one scenario into column arrays (+ per-row run index and run specs)."""
    root = Path(results_dir) / scenario
    cols, run_of, specs, manifests, timing = {}, [], [], [], []
    for k, d in enumerate(sorted(p for p in root.iterdir() if (p / "summary.json").exists())):
        w = read_csv_gz(d / "windows.csv.gz")
        n = len(w["idx"])
        specs.append(json.loads((d / "summary.json").read_text()))
        manifests.append(json.loads((d / "manifest.json").read_text()))
        t = read_csv_gz(d / "timing.csv.gz")
        timing.append({c: np.array(v, dtype=float) for c, v in t.items() if c.startswith("ns_")})
        for c, v in w.items():
            if c in ("run_id", "idx"):
                continue
            if c in ("origin", "mechanism"):
                arr = np.array(v, dtype=object)
            else:
                arr = np.array([x if x != "" else "nan" for x in v], dtype=np.float32)
            cols.setdefault(c, []).append(arr)
        run_of.append(np.full(n, k, dtype=np.int32))
    if not specs:
        return None
    data = {c: np.concatenate(v) for c, v in cols.items()}
    data["run"] = np.concatenate(run_of)
    data["y"] = data["compromised"] > 0.5
    return {"data": data, "specs": specs, "manifests": manifests, "timing": timing}


def architectures(d, weights):
    """Composite scores for every architecture (requirement §6). Higher = more suspicious."""
    Z = lambda k: d["z_" + k].astype(np.float64)
    wc, wp, wl = weights["proposed"]
    gc, gp = weights["gate_cm_phys"]
    sec = np.fmax(Z("sec_step"), Z("sec_replay"))
    rel = np.fmax(Z("rel_vote"), Z("rel_cusum"))
    a = {
        "proposed": wc * Z("cm") + wp * Z("phys") + wl * Z("learn"),
        "gate_cm_phys": gc * Z("cm") + gp * Z("phys"),
        "cm_only": Z("cm"), "phys_only": Z("phys"), "learn_only": Z("learn"),
        "B1_separated": np.fmax(sec, rel), "B2_security": sec, "B3_reliability": rel,
        "B4_iforest": Z("iforest"), "B5_ocsvm": Z("ocsvm"), "B6_pca": Z("pca"), "B7_cusum": Z("cusum"),
    }
    for g in [k[7:] for k in d if k.startswith("z_phys_")]:
        a["phys_" + g] = Z("phys_" + g)
    return a


# ---------------------------------------------------------------------------------------------
def calibrate(results_dir, fpr=0.01):
    out = {"fpr_target": fpr, "source": str(results_dir), "scenarios": {}}
    for scen in ("motor", "reactor", "robot"):
        L = load(results_dir, scen)
        if L is None:
            continue
        d, y = L["data"], L["data"]["y"]
        best = None
        for w in WEIGHT_GRID:
            s = w[0] * d["z_cm"] + w[1] * d["z_phys"] + w[2] * d["z_learn"]
            a = auc(s, y)
            if np.isfinite(a) and (best is None or a > best[0]):
                best = (a, w)
        best2 = None
        for k in range(5):
            w = (k / 4, 1 - k / 4)
            a = auc(w[0] * d["z_cm"] + w[1] * d["z_phys"], y)
            if np.isfinite(a) and (best2 is None or a > best2[0]):
                best2 = (a, w)
        weights = {"proposed": list(best[1]) if best else [1 / 3] * 3,
                   "gate_cm_phys": list(best2[1]) if best2 else [0.5, 0.5]}
        arch = architectures(d, weights)
        thr = {}
        for name, s in arch.items():
            neg = s[~y]
            neg = neg[np.isfinite(neg)]
            thr[name] = float(np.quantile(neg, 1 - fpr)) if len(neg) else float("nan")
        out["scenarios"][scen] = {"weights": weights, "thresholds": thr,
                                  "calibration_auc_proposed": best[0] if best else None,
                                  "n_runs": len(L["specs"]), "n_windows": int(len(y)),
                                  "n_negative_windows": int((~y).sum()),
                                  "campaign_sha256": sorted({s["campaign"]["sha256"] for s in L["specs"]})}
    return out


# ---------------------------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(lo, hi, nd=3):
    return f"[{_fmt(lo, nd)}–{_fmt(hi, nd)}]"


def detection_delay(d, dec, specs):
    """Per corrupted run: hours from onset to first alarm (None if never alarmed)."""
    out = []
    for k, s in enumerate(specs):
        if not s["spec"]["mechanism"]:
            continue
        m = (d["run"] == k) & np.isfinite(d["hours_since_onset"])
        if not m.any():
            continue
        hs = d["hours_since_onset"][m]
        al = dec[m]
        out.append(float(hs[al].min()) if al.any() else None)
    return out


def report(results_dir, thresholds, out_dir, B=200, attribution=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    kinds = set()
    for scen in ("motor", "reactor", "robot"):
        if scen not in thresholds["scenarios"]:
            continue
        L = load(results_dir, scen)
        if L is None:
            continue
        d, specs = L["data"], L["specs"]
        kinds |= {s["campaign"]["kind"] for s in specs}
        y = d["y"]
        cal = thresholds["scenarios"][scen]
        arch = architectures(d, cal["weights"])
        dec = {a: np.nan_to_num(s, nan=-np.inf) > cal["thresholds"][a] for a, s in arch.items()}
        every = specs[0]["campaign"]["eval_every_h"]
        runs = d["run"]
        md.append(f"\n## {scen}\n\n{len(specs)} runs, {len(y):,} operation windows "
                  f"({int(y.sum()):,} compromised). Thresholds at {thresholds['fpr_target']:.0%} FPR "
                  f"from calibration ({cal['n_runs']} runs). Gate weights (cm, phys, learn) = {cal['weights']['proposed']}.\n")

        # O1 main detection table + DeLong / bootstrap dAUC
        rows = []
        for a in ["proposed"] + BASELINES:
            s, dd = arch[a], dec[a]
            if not np.isfinite(s).any():
                rows.append({"architecture": a, "note": "not run (scikit-learn not installed)"})
                continue
            cm = confusion(dd, y)
            A = auc(s, y)
            lo, hi = bootstrap_runs(runs, lambda ix: auc(s[ix], y[ix]), B=B)
            neg_h = (~y).sum() * every
            fa = cm["fp"] / neg_h * 1000 if neg_h else float("nan")
            dl = detection_delay(d, dd, specs)
            got = [x for x in dl if x is not None]
            row = {"architecture": a, "precision": cm["precision"], "recall": cm["recall"], "f1": cm["f1"],
                   "auc": A, "auc_lo": lo, "auc_hi": hi, "fpr_test": cm["fpr"], "false_alarms_per_1000h": fa,
                   "median_delay_h": float(np.median(got)) if got else float("nan"),
                   "delay_q1": float(np.quantile(got, 0.25)) if got else float("nan"),
                   "delay_q3": float(np.quantile(got, 0.75)) if got else float("nan"),
                   "events_detected": f"{len(got)}/{len(dl)}"}
            if a != "proposed":
                _, _, diff, z, p = delong(arch["proposed"], s, y)
                blo, bhi = bootstrap_runs(runs, lambda ix: auc(arch["proposed"][ix], y[ix]) - auc(s[ix], y[ix]), B=B, seed=1)
                row.update({"dAUC_vs_proposed": -diff if np.isfinite(diff) else diff, "dAUC_boot_lo": -bhi, "dAUC_boot_hi": -blo, "delong_p": p})
            rows.append(row)
        _write_csv(out_dir / f"O1_detection_{scen}.csv", rows)
        md.append("### O1 Detection (operation windows)\n\n| Architecture | Precision | Recall | F1 | AUC [95% CI] | FPR (test) | False alarms / 1,000 h | Median delay h [IQR] | Events detected | ΔAUC (baseline − proposed) [boot CI] (DeLong p) |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if "note" in r:
                md.append(f"| {r['architecture']} | {r['note']} |||||||||")
                continue
            dpart = "" if r["architecture"] == "proposed" else f"{_fmt(r['dAUC_vs_proposed'])} {_ci(r['dAUC_boot_lo'], r['dAUC_boot_hi'])} (p={_fmt(r['delong_p'], 4)})"
            md.append(f"| {r['architecture']} | {_fmt(r['precision'], 2)} | {_fmt(r['recall'], 2)} | {_fmt(r['f1'], 2)} | "
                      f"{_fmt(r['auc'])} {_ci(r['auc_lo'], r['auc_hi'])} | {_fmt(100 * r['fpr_test'], 1)}% | "
                      f"{_fmt(r['false_alarms_per_1000h'], 1)} | {_fmt(r['median_delay_h'], 0)} [{_fmt(r['delay_q1'], 0)}–{_fmt(r['delay_q3'], 0)}] | "
                      f"{r['events_detected']} | {dpart} |")

        # O5 recall per mechanism (+ attack mode / knowledge)
        def sub_label(k):
            sp = specs[k]["spec"]
            m = sp["mechanism"]
            if m == "A4":
                return f"A4_{sp['attacker_knowledge']}"
            if m in ("A1", "A2"):
                return f"{m}_{sp['attack_mode']}"
            return m
        lab = np.array([sub_label(k) if specs[k]["spec"]["mechanism"] else "none" for k in runs.tolist()], dtype=object)
        lab[~y] = "none"
        cols5 = ["proposed", "cm_only", "phys_only"] + BASELINES
        rows5 = []
        for m in sorted(set(lab) - {"none"}):
            mm = lab == m
            r = {"mechanism": m, "n_windows": int(mm.sum()), "n_runs": int(len(set(runs[mm].tolist())))}
            for a in cols5:
                r[a] = float(dec[a][mm].mean()) if np.isfinite(arch[a]).any() else float("nan")
            rows5.append(r)
        _write_csv(out_dir / f"O5_recall_by_mechanism_{scen}.csv", rows5)
        md.append("\n### O5 / O7 Recall by mechanism (window level)\n\n| Mechanism | runs | " + " | ".join(cols5) + " |\n|---|---|" + "---|" * len(cols5))
        for r in rows5:
            md.append(f"| {r['mechanism']} | {r['n_runs']} | " + " | ".join(_fmt(r[a], 2) for a in cols5) + " |")

        # O8 / O4 operational cost
        pf_clean = (d["process_fault"] > 0.5) & ~y
        rows8 = []
        for a in ["proposed", "gate_cm_phys"] + BASELINES:
            if not np.isfinite(arch[a]).any():
                continue
            dd = dec[a]
            expo = []
            for k, s in enumerate(specs):
                if s["spec"]["mechanism"]:
                    m = (runs == k) & y
                    if m.any():
                        expo.append(float((~dd[m]).sum() * every))
            rows8.append({"architecture": a,
                          "wrongful_quarantine_during_process_faults": float(dd[pf_clean].mean()) if pf_clean.any() else float("nan"),
                          "n_process_fault_windows": int(pf_clean.sum()),
                          "unnecessary_quarantine_h_per_1000h": float(dd[~y].sum() * every / max((~y).sum() * every, 1) * 1000),
                          "safety_exposure_h_per_event_mean": float(np.mean(expo)) if expo else float("nan")})
        rows8.append({"architecture": "B0_no_checks", "wrongful_quarantine_during_process_faults": 0.0,
                      "n_process_fault_windows": int(pf_clean.sum()), "unnecessary_quarantine_h_per_1000h": 0.0,
                      "safety_exposure_h_per_event_mean": float(np.mean([float(((runs == k) & y).sum() * every) for k, s in enumerate(specs) if s["spec"]["mechanism"]] or [np.nan]))})
        _write_csv(out_dir / f"O8_operational_{scen}.csv", rows8)
        md.append("\n### O8 / O4 Operational cost\n\n| Architecture | Wrongful quarantine during process faults | Unnecessary quarantine h / 1,000 h | Safety exposure h / event |\n|---|---|---|---|")
        for r in rows8:
            md.append(f"| {r['architecture']} | {_fmt(100 * r['wrongful_quarantine_during_process_faults'], 1)}% (n={r['n_process_fault_windows']}) | "
                      f"{_fmt(r['unnecessary_quarantine_h_per_1000h'], 1)} | {_fmt(r['safety_exposure_h_per_event_mean'], 1)} |")

        # O9 physics by constraint group
        rows9 = []
        for g in GROUP_COLS[scen]:
            a = "phys_" + g
            if a in arch:
                cm = confusion(dec[a], y)
                rows9.append({"constraint": g, "recall": cm["recall"], "fpr": cm["fpr"], "auc": auc(arch[a], y)})
        _write_csv(out_dir / f"O9_physics_constraints_{scen}.csv", rows9)
        md.append("\n### O9 Physics constraints\n\n| Constraint | Recall | FPR | AUC |\n|---|---|---|---|")
        for r in rows9:
            md.append(f"| {r['constraint']} | {_fmt(r['recall'], 2)} | {_fmt(100 * r['fpr'], 1)}% | {_fmt(r['auc'])} |")

        # O10 ablation
        full = auc(arch["proposed"], y)
        rows10 = []
        for a in ABLATION:
            A = auc(arch[a], y)
            lo, hi = bootstrap_runs(runs, lambda ix: auc(arch[a][ix], y[ix]), B=B, seed=2)
            rows10.append({"configuration": a, "auc": A, "auc_lo": lo, "auc_hi": hi,
                           "recall_at_threshold": confusion(dec[a], y)["recall"], "dAUC_vs_full": A - full})
        _write_csv(out_dir / f"O10_ablation_{scen}.csv", rows10)
        md.append("\n### O10 Ablation (identical windows)\n\n| Configuration | AUC [95% CI] | Recall @ threshold | ΔAUC vs proposed |\n|---|---|---|---|")
        for r in rows10:
            md.append(f"| {r['configuration']} | {_fmt(r['auc'])} {_ci(r['auc_lo'], r['auc_hi'])} | {_fmt(r['recall_at_threshold'], 2)} | {_fmt(r['dAUC_vs_full'])} |")

        # O12 learning (gated vs ungated soft-sensor error on CLEAN data, by hours since onset)
        hs = d["hours_since_onset"]
        bins = [0, 100, 250, 500, 1000, 2000, 5000, 1e9]
        rows12 = []
        for origin in ("natural", "adversarial"):
            om = np.array([specs[k]["spec"]["origin"] == origin for k in runs.tolist()]) & np.isfinite(hs)
            for b0, b1 in zip(bins[:-1], bins[1:]):
                m = om & (hs >= b0) & (hs < b1)
                if m.any():
                    rows12.append({"origin": origin, "hours_since_onset_from": b0, "to": b1, "n_windows": int(m.sum()),
                                   "mse_gated_median": float(np.median(d["err_gated"][m])),
                                   "mse_ungated_median": float(np.median(d["err_ungated"][m])),
                                   "gate_rejection_rate": float(1 - d["gate_ok"][m].mean())})
        _write_csv(out_dir / f"O12_learning_{scen}.csv", rows12)
        md.append("\n### O12 Gated vs ungated learning (soft-sensor squared error on clean data)\n\n| Origin | Hours since onset | Windows | Median error gated | Median error ungated | Gate rejection rate |\n|---|---|---|---|---|---|")
        for r in rows12:
            md.append(f"| {r['origin']} | {r['hours_since_onset_from']:.0f}–{r['to']:.0f} | {r['n_windows']} | {r['mse_gated_median']:.4g} | {r['mse_ungated_median']:.4g} | {_fmt(r['gate_rejection_rate'], 2)} |")

        # O13 compute time and memory
        rows13 = []
        for comp in ("ns_cm", "ns_phys", "ns_learn", "ns_sec", "ns_rel", "ns_cusum", "ns_learned"):
            v = np.concatenate([t[comp] for t in L["timing"] if comp in t]) / 1e6
            rows13.append({"component": comp[3:], "p50_ms": float(np.median(v)), "p99_ms": float(np.quantile(v, 0.99)), "max_ms": float(v.max())})
        rss = [m.get("peak_rss_mb") for m in L["manifests"] if m.get("peak_rss_mb") is not None]
        _write_csv(out_dir / f"O13_compute_{scen}.csv", rows13)
        md.append("\n### O13 Compute time per decision (measured)\n\n| Component | p50 ms | p99 ms | max ms |\n|---|---|---|---|")
        for r in rows13:
            md.append(f"| {r['component']} | {r['p50_ms']:.3f} | {r['p99_ms']:.3f} | {r['max_ms']:.3f} |")
        md.append(f"\nPeak worker memory (max over runs): {max(rss):.0f} MB.\n" if rss else "")

        if attribution:
            from .attribution import attribution_table
            rows6 = attribution_table(d, specs)
            _write_csv(out_dir / f"O6_attribution_{scen}.csv", rows6)
            md.append("\n### O6 Attribution on matched pairs (can natural vs adversarial be told apart?)\n\n| Pair | Runs (nat/adv) | Classifier | Accuracy [95% Wilson CI] | Chance |\n|---|---|---|---|---|")
            for r in rows6:
                md.append(f"| {r['pair']} | {r['n_natural']}/{r['n_adversarial']} | {r['classifier']} | {_fmt(r['accuracy'], 2)} {_ci(r['acc_lo'], r['acc_hi'], 2)} | {_fmt(r['chance'], 2)} |")

    # O14 campaign summary
    env_rows = summarize_environment(results_dir)
    _write_csv(out_dir / "O14_campaign.csv", env_rows)
    md.append("\n## O14 Campaign and environment (from run manifests)\n\n| Item | Value |\n|---|---|")
    for r in env_rows:
        md.append(f"| {r['item']} | {r['value']} |")

    banner = ("> ⚠️ **SMOKE / CALIBRATION OUTPUT, NOT RESULTS.** Do not use these numbers in the paper.\n"
              if kinds & {"smoke", "calibration"} else
              "> Generated by `utmsim report` from stored runs. These tables are the only valid source of "
              "simulation numbers for the manuscript.\n")
    head = [f"# UTM simulation report\n", banner, f"\nResults: `{results_dir}`  ·  thresholds: `{thresholds.get('source')}` "
            f"(FPR target {thresholds['fpr_target']:.0%})  ·  bootstrap B = {B} (resampling runs)\n"]
    (out_dir / "report.md").write_text("\n".join(head + md) + "\n")
    return out_dir / "report.md"


def summarize_environment(results_dir):
    mans, kinds, nwin, nruns = [], set(), 0, 0
    for p in Path(results_dir).glob("*/*/manifest.json"):
        mans.append(json.loads(p.read_text()))
        s = json.loads((p.parent / "summary.json").read_text())
        kinds.add(s["campaign"]["kind"])
        nwin += s["n_windows"]
        nruns += 1
    if not mans:
        return []
    envs = {(m["environment"]["cpu_model"], m["environment"]["cpu_count"], m["environment"]["platform"]) for m in mans}
    vers = {(m["environment"]["python"], m["environment"]["numpy"], m["environment"]["scikit_learn"]) for m in mans}
    commits = sorted({m["code_commit"] for m in mans})
    wall = sum(m["wall_s"] for m in mans)
    return [
        {"item": "Campaign kind(s)", "value": ", ".join(sorted(kinds))},
        {"item": "Runs", "value": nruns},
        {"item": "Operation windows evaluated", "value": f"{nwin:,}"},
        {"item": "Hardware", "value": "; ".join(f"{c} ({n} cores), {p}" for c, n, p in sorted(envs))},
        {"item": "Software", "value": "; ".join(f"Python {a}, numpy {b}, scikit-learn {c or 'not installed'}" for a, b, c in sorted(vers, key=str))},
        {"item": "Total compute (sum of run wall times)", "value": f"{wall / 3600:.2f} core-hours"},
        {"item": "Code version(s)", "value": ", ".join(commits)},
    ]


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k))) for k in keys))
    Path(path).write_text("\n".join(lines) + "\n")
