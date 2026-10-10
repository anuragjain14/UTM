import csv, glob, html, json, re, zipfile
from pathlib import Path

SIM = Path("/Users/parulmann/Documents/PAPERS_KAMAL_PERSONAL/Working /WithAnurag/UnifiedThreatModeling/simulator")
KM = SIM.parent / "KM_working_"
V3 = KM / "Unified Threat Modeling for Industrial AI Systems_v3_merged.docx"
OUT = KM / "Unified Threat Modeling for Industrial AI Systems_v4_results.docx"
REP = SIM / "report_test"
HL = "green"
SC = ("motor", "reactor", "robot")
SCN = {"motor": "Motor", "reactor": "Reactor", "robot": "Robot"}
ARCHN = {"proposed": "Proposed (unified)", "B1_separated": "B1 Separated", "B2_security": "B2 Security-focused",
         "B3_reliability": "B3 Reliability-focused", "B4_iforest": "B4 Isolation Forest", "B5_ocsvm": "B5 One-class SVM",
         "B6_pca": "B6 PCA residual", "B7_cusum": "B7 CUSUM", "gate_cm_phys": "Cross-modal + physics (no learning)",
         "cm_only": "Cross-modal only", "phys_only": "Physics only", "learn_only": "Learning residual only", "B0_no_checks": "B0 No checks"}
SHORT = {"B1_separated": "B1", "B2_security": "B2", "B3_reliability": "B3", "B4_iforest": "B4", "B5_ocsvm": "B5", "B6_pca": "B6", "B7_cusum": "B7"}

def rd(name, scen=None):
    p = REP / (f"{name}_{scen}.csv" if scen else f"{name}.csv")
    return list(csv.DictReader(open(p)))

def f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else f"{v:.{nd}f}"

O1 = {s: {r["architecture"]: r for r in rd("O1_detection", s)} for s in SC}
O5 = {s: {r["mechanism"]: r for r in rd("O5_recall_by_mechanism", s)} for s in SC}
O6 = {s: rd("O6_attribution", s) for s in SC}
O8 = {s: {r["architecture"]: r for r in rd("O8_operational", s)} for s in SC}
O9 = {s: rd("O9_physics_constraints", s) for s in SC}
O10 = {s: {r["configuration"]: r for r in rd("O10_ablation", s)} for s in SC}
O12 = {s: rd("O12_learning", s) for s in SC}
O13 = {s: {r["component"]: r for r in rd("O13_compute", s)} for s in SC}
O14 = {r["item"]: r["value"] for r in rd("O14_campaign")}
TH = json.load(open(SIM / "thresholds.json"))
MMREP = {"0×": SIM / "report_mismatch_0x", "1×": REP, "2×": SIM / "report_mismatch_2x"}
MM = {m: {sc: {r["architecture"]: r for r in csv.DictReader(open(p / f"O1_detection_{sc}.csv"))} for sc in ("motor", "reactor", "robot")} for m, p in MMREP.items()}
MM8 = {m: {sc: {r["architecture"]: r for r in csv.DictReader(open(p / f"O8_operational_{sc}.csv"))} for sc in ("motor", "reactor", "robot")} for m, p in MMREP.items()}
cal_core_h = sum(json.load(open(p))["wall_s"] for p in glob.glob(str(SIM / "results/calibration/*/*/manifest.json"))) / 3600
test_core_h = sum(json.load(open(p))["wall_s"] for p in glob.glob(str(SIM / "results/test/*/*/manifest.json"))) / 3600
peak = max(json.load(open(p))["peak_rss_mb"] for p in glob.glob(str(SIM / "results/test/*/*/manifest.json")))
def wall(log):
    m = re.search(r"done in (\d+)s", (SIM / "logs" / log).read_text()); return int(m.group(1))
cal_wall, test_wall = wall("calibration.log"), wall("test.log")
nwin = {s: sum(json.load(open(p))["n_windows"] for p in glob.glob(str(SIM / f"results/test/{s}/*/summary.json"))) for s in SC}
ncorr = {s: sum(1 for p in glob.glob(str(SIM / f"results/test/{s}/*/summary.json")) if json.load(open(p))["spec"]["mechanism"]) for s in SC}

def auc(s, a): return float(O1[s][a]["auc"])
def ci(s, a): return f"{f(O1[s][a]['auc'])} [{f(O1[s][a]['auc_lo'])}–{f(O1[s][a]['auc_hi'])}]"
def rec(s, a): return float(O1[s][a]["recall"])
def mech(s, m, a): return float(O5[s][m][a]) if m in O5[s] else None
def wq(s, a): return 100 * float(O8[s][a]["wrongful_quarantine_during_process_faults"])

# ---------------------------------------------------------------- docx helpers
zin = zipfile.ZipFile(V3)
doc = zin.read("word/document.xml").decode("utf8")
b0 = doc.index("<w:body>") + 8; b1 = doc.rindex("<w:sectPr")
head, tail = doc[:b0], doc[b1:]
blocks = re.findall(r"(<w:tbl>.*?</w:tbl>|<w:p[ >].*?</w:p>|<w:p/>)", doc[b0:b1], re.S)
ptext = lambda b: html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", b)))
def idx(prefix): return next(i for i, b in enumerate(blocks) if ptext(b).startswith(prefix))
def ppr(b): m = re.search(r"<w:pPr>.*?</w:pPr>", b, re.S); return m.group(0) if m else ""
def rpr(b): m = re.search(r"<w:r>(<w:rPr>.*?</w:rPr>)", b, re.S); return m.group(1) if m else ""
AFTER = ['<w:u ', '<w:u/', '<w:effect', '<w:bdr', '<w:shd', '<w:fitText', '<w:vertAlign', '<w:rtl', '<w:cs', '<w:em ', '<w:lang', '<w:eastAsianLayout', '<w:specVanish', '<w:oMath', '<w14:']
def hl(r):
    r = re.sub(r'<w:highlight [^>]*/>', '', r or '<w:rPr></w:rPr>'); pos = len(r) - len('</w:rPr>')
    for t in AFTER:
        k = r.find(t)
        if k != -1 and k < pos: pos = k
    return r[:pos] + f'<w:highlight w:val="{HL}"/>' + r[pos:]
def run(t, r): return '<w:r>' + r + '<w:t xml:space="preserve">' + html.escape(t, quote=False) + '</w:t></w:r>'

H2 = blocks[idx("3.6 Experimental Design")]; H3 = blocks[idx("3.6.1 Simulation Framework")]
BODY = blocks[idx("A simulation framework was developed")]; CAP = blocks[idx("Table 4.")]
H1 = blocks[idx("5. CONCLUSION")]
cap_r = re.findall(r"<w:r>(<w:rPr>.*?</w:rPr>)", CAP, re.S)
def P(tmpl, text, bold_lead=None):
    r0 = rpr(tmpl)
    segs = ""
    if bold_lead:
        rb = r0.replace("<w:rPr>", '<w:rPr><w:b w:val="1"/><w:bCs w:val="1"/>', 1)
        segs += run(bold_lead, hl(rb))
    return "<w:p>" + ppr(tmpl) + segs + run(text, hl(r0)) + "</w:p>"
def body(t, lead=None): return P(BODY, t, lead)
def h2(t): return P(H2, t)
def h3(t): return P(H3, t)
def caption(num, text): return "<w:p>" + ppr(CAP) + run(f"Table {num}. ", hl(cap_r[0])) + run(text, hl(cap_r[1])) + "</w:p>"

TBL = blocks[idx("ApproachAccuracyPrecision")]
TBLPR = re.search(r"<w:tblPr>.*?</w:tblPr>", TBL, re.S).group(0)
rows = re.findall(r"<w:tr[ >].*?</w:tr>", TBL, re.S)
HC = re.search(r"<w:tc>.*?</w:tc>", rows[0], re.S).group(0); BC = re.search(r"<w:tc>.*?</w:tc>", rows[1], re.S).group(0)
HTRPR = re.search(r"<w:tblPrEx>.*?</w:trPr>", rows[0], re.S).group(0); BTRPR = re.search(r"<w:tblPrEx>.*?</w:trPr>", rows[1], re.S).group(0)
def cell(tmpl, text, w, center=False):
    tcpr = re.sub(r'<w:tcW [^>]*/>', f'<w:tcW w:type="dxa" w:w="{w}"/>', re.search(r"<w:tcPr>.*?</w:tcPr>", tmpl, re.S).group(0))
    p = re.search(r"<w:p>.*?</w:p>", tmpl, re.S).group(0)
    pp = ppr(p) if center else ppr(p).replace('<w:jc w:val="center"/>', '')
    if center and '<w:jc' not in pp: pp = pp.replace('</w:pPr>', '<w:jc w:val="center"/></w:pPr>')
    rr = rpr(p).replace('<w:sz w:val="20"/><w:szCs w:val="20"/>', '<w:sz w:val="17"/><w:szCs w:val="17"/>')
    return "<w:tc>" + tcpr + "<w:p>" + pp + run(str(text), hl(rr)) + "</w:p></w:tc>"
def table(header, data, widths):
    assert len(header) == len(widths) and abs(sum(widths) - 9360) < 5, (len(header), len(widths), sum(widths))
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
    out = "<w:tbl>" + TBLPR + grid
    out += "<w:tr>" + HTRPR + "".join(cell(HC, h, w, True) for h, w in zip(header, widths)) + "</w:tr>"
    for r in data:
        out += "<w:tr>" + BTRPR + "".join(cell(BC, v, w, k > 0) for k, (v, w) in enumerate(zip(r, widths))) + "</w:tr>"
    return out + "</w:tbl>"

# ---------------------------------------------------------------- derived numbers for text
w = {s: TH["scenarios"][s]["weights"]["proposed"] for s in SC}
best_base = {s: max((a for a in SHORT if O1[s][a]["auc"] not in ("", "nan")), key=lambda a: auc(s, a)) for s in SC}
lr = {}
for s in SC:
    adv = [r for r in O12[s] if r["origin"] == "adversarial"]
    lr[s] = [float(r["mse_ungated_median"]) / float(r["mse_gated_median"]) for r in adv]
lr_lo, lr_hi = min(min(lr["motor"]), min(lr["reactor"])), max(max(lr["motor"]), max(lr["reactor"]))
base_wq = [wq(s, a) for s in SC for a in ("B1_separated", "B2_security", "B3_reliability")]
def mech_row_best(s, m):
    if m not in O5[s]: return None
    vals = {a: float(O5[s][m][a]) for a in SHORT if O5[s][m].get(a) not in ("", "nan", None)}
    a = max(vals, key=vals.get); return a, vals[a]
rep_base = max(mech_row_best(s, "A3")[1] for s in ("reactor", "robot"))
rep_motor = mech_row_best("motor", "A3")

new = []
# ================= 3.6 =================
new += [h2("3.6 Experimental Design and Validation"),
        h3("3.6.1 Simulation Framework"),
        body("The architecture is evaluated in a pre-specified simulation campaign. The study is simulation-based and does not claim validation on a physical production line. Each run simulates one plant over a horizon of 1,000–10,000 h on two time scales: slow processes (operating-point changes, thermal states, degradation, gradual corruption) evolve on a 1-h grid, and every 2 h a high-rate decision window of 512 samples is generated for every sensor, at the run's sampling rate. One integrity decision is made per window. The first 192 h of every run are a clean commissioning period: the first two-thirds fit every detector, and the last third provide each detector's robust normalization (median and median absolute deviation), so that every score is a z-score. Corruption and process faults can therefore begin from 196 h. Table 2 lists the parameter ranges, drawn independently per run."),
        body("To avoid circular evaluation, data are generated by a different model from the one the detectors use. Each run perturbs the generator's physical parameters by up to ±10% (±5% for the robot) and adds dynamics that are absent from the nominal model (described below). The detectors only know the nominal equations and the clean commissioning data; they never see the generator's parameters or the corruption labels."),
        table(["Parameter", "Range / setting", "Notes"], [
            ["Horizon", "1,000–10,000 h (uniform)", "Per run"],
            ["Decision windows", "Every 2 h; 512 samples", "Commissioning: first 192 h, always clean"],
            ["Sampling rate", "Motor 200–1,000 Hz; reactor 1–10 Hz; robot 100–500 Hz", "Window length 0.5–2.6 s, 51–512 s and 1–5 s"],
            ["Sensors per modality", "4–16 (uniform)", "Fixed counts: motor speed 2, ambient 1; robot force 6"],
            ["Sensor noise", "0.1–10% of the measurement span (log-uniform)", "Per-sensor gain/offset error, coloured noise, small nonlinearity"],
            ["Corruption onset", "196–5,000 h (uniform)", "Process-fault onset drawn independently"],
            ["Severity", "Low / medium / high (S = 1, 3, 6)", "Magnitude unit 2·S·max(sensor σ, 0.5% of span)"],
            ["Model mismatch", "Generator parameters ±10% of nominal (robot ±5%) + unmodelled terms", "Mismatch factor 1 (sweep at 0× and 2× in Section 4.8)"],
            ["Run mix", "Clean 10%, process fault only 8%, sensor corruption 76%, corruption + process fault 6%", "Mechanism, attack mode and attacker knowledge drawn uniformly"],
        ], [2000, 4060, 3300]),
        caption(2, "Simulation parameters and design of the campaign."),
        h3("3.6.2 Plants, Faults and Corruption Mechanisms"),
        body("Three reduced-order plants are simulated (Figure 5). The motor (50 HP) has a load schedule with fast load fluctuation, slip-dependent speed, a current law, vibration that depends on speed, load and bearing health, acoustic emission coupled to vibration, and a first-order thermal model driven by losses and ambient temperature; unmodelled terms are a non-unit current-law exponent, a thermal–vibration coupling and sensor-location factors. The batch reactor runs a 10-h fill–heat–react–drain cycle with temperature control by coolant flow; unmodelled vapour-pressure curvature is added. The robotic cell is a six-degree-of-freedom arm running periodic pick-and-place programs whose trajectory and payload change about every 12 h, with gravity, inertia and friction torques, a wrist force sensor and a vision system; unmodelled joint offsets and viscous friction are added. The nominal relations used by the physics layer are, for the motor: current versus speed-inferred load, acoustic versus vibration, temperature versus a nominal thermal model driven by the measured current, and vibration versus speed and load; for the reactor: mass balance, pressure versus temperature and level, and coolant flow versus the controller law; and for the robot: vision versus forward kinematics, joint currents versus nominal dynamics, and wrist force versus the known payload."),
        body("Genuine process faults (motor bearing wear, reactor heat-transfer fouling, robot joint-friction increase) change the process while the data remain correct; they are not counted as integrity positives, and the results report how often each architecture wrongly quarantines correct data during them. Eleven sensor-corruption mechanisms are injected. Natural: N1 saturating, temperature-modulated drift; N2 calibration step (gain and offset); N3 noise growth; N4 mounting degradation; N5 intermittent stuck-at; N6 encoder wear. Adversarial: A1 gradual bias; A2 noise amplification; A3 replay of a window recorded during commissioning; A4 coordinated manipulation; A5 false-data-injection step. Natural and adversarial mechanisms follow their own generative laws. For A1 and A2 the attacker is either naive, using its own linear or abrupt law, or mimicking, deliberately reproducing the natural law of N1 or N3. For A4 the attacker has one of three knowledge levels: K0 offsets one or two sensors; K1 applies the same offset to all sensors of a modality, preserving their correlations; K2 propagates a fake physical change through the nominal model to every related modality."),
        blocks[idx("Figure 5.") - 1],
        "<w:p>" + ppr(blocks[idx("Figure 5.")]) + run("Figure 5. ", hl(rpr(blocks[idx("Figure 5.")]))) + run("Illustrative configuration of the three simulated plants and the injection points. Sensor counts in the simulation are drawn per run (4–16 per modality; Table 2).", hl(re.findall(r"<w:r>(<w:rPr>.*?</w:rPr>)", blocks[idx('Figure 5.')], re.S)[-1])) + "</w:p>",
        h3("3.6.3 Detectors, Baselines and Thresholds"),
        body(f"The proposed architecture combines the z-scores of cross-modal verification (Algorithm 1, the mean normalized deviation of the window's correlation matrix from its commissioning baseline), physics-informed validation (Algorithm 2, the mean over relation groups of the largest normalized residual) and the residual of the gated soft-sensor model (Algorithm 3). Their weights were chosen on calibration runs from a 15-point grid by maximizing the AUC: motor {w['motor']}, reactor {w['reactor']}, robot {w['robot']} (cross-modal, physics, learning). The learner is a linear soft sensor updated by normalized least mean squares with an Elastic Weight Consolidation penalty (λ = 1); its gated copy is updated only when the mean of the cross-modal and physics z-scores is below 3, a rule fixed before any run. Seven baselines see the same windows: B1 separated (the maximum of B2 and B3), B2 security-focused (step change between consecutive windows and replay similarity to stored commissioning windows), B3 reliability-focused (redundant-sensor voting and a Page CUSUM on each sensor's deviation from its redundancy group), B4 Isolation Forest, B5 one-class SVM and B6 PCA residual (Q statistic) on window features, and B7 a classical per-sensor Page CUSUM (allowance 0.5, decision interval 5, reset after alarm)."),
        body(f"Every architecture's threshold was set at a 1% false-positive rate on the non-corrupted windows of a separate calibration campaign of 900 runs (300 per plant) with disjoint random seeds. Weights and thresholds were then fixed before the test campaign of 3,000 runs (1,000 per plant) was executed: calibration finished at 02:46 UTC, the thresholds were written at 02:47 UTC and the first test run completed at 03:28 UTC on 9 October 2026."),
        h3("3.6.4 Evaluation Metrics and Statistics"),
        body("A window is a positive if a sensor-corruption mechanism is active in it (from onset onward; for N5 only when the sensor is stuck); windows with only a process fault are negatives. We report the AUC, and recall, precision and false-positive rate at the fixed threshold; false alarms per 1,000 h of non-corrupted operation; the number of corrupted runs with at least one alarm and the median delay to the first alarm; recall by mechanism; the rate of wrongful quarantine during genuine process faults; the soft-sensor error on the uncorrupted readings of the same windows for the gated and ungated learners; and the measured compute time per decision. 95% confidence intervals are obtained by bootstrap over runs (200 replicates), because windows within a run are correlated; for the same reason DeLong p-values, which treat windows as independent, are reported but not used for inference. Whether natural and adversarial corruption can be told apart is tested on matched pairs (N1 vs A1, N2 vs A5, N3 vs A2): per corrupted run, summary features of the post-onset trajectory of observable features and scores are classified with logistic regression and gradient boosting under 5-fold stratified cross-validation, and accuracy is compared with the majority-class rate."),
        h3("3.6.5 Implementation and Reproducibility"),
        body("The simulator is written in Python with numpy; scikit-learn provides baselines B4 and B5 and the gradient-boosting classifier. Each run is fully determined by the campaign file and its index, writes a manifest with the software environment, wall time, peak memory and SHA-256 checksums of all outputs, and is byte-for-byte reproducible. All 900 calibration and 3,000 test runs completed without error and passed checksum verification. Table 3 summarizes the execution environment."),
        table(["Item", "Value"], [
            ["Hardware", O14["Hardware"]],
            ["Software", O14["Software"]],
            ["Runs", "Calibration 900 (300 per plant); test 3,000 (1,000 per plant)"],
            ["Decision windows (test)", f"{int(O14['Operation windows evaluated'].replace(',', '')):,} (motor {nwin['motor']:,}; reactor {nwin['reactor']:,}; robot {nwin['robot']:,})"],
            ["Compute", f"Calibration {cal_core_h:.1f} core-hours ({cal_wall / 60:.0f} min wall); test {test_core_h:.1f} core-hours ({test_wall / 60:.0f} min wall); 14 worker processes"],
            ["Peak memory per worker", f"{peak:.0f} MB"],
        ], [2600, 6760]),
        caption(3, "Execution environment and cost of the campaigns (from the run manifests and logs)."),
        ]
k_ext = idx("3.6.7 Planned External")
new += [P(H3, "3.6.6 Planned External Validation on Public ICS Data")] + blocks[k_ext + 1:idx("4. RESULTS AND DISCUSSION")]

# ================= 4 =================
s_ = lambda s: SCN[s]
new += [blocks[idx("4. RESULTS AND DISCUSSION")], h2("4.1 Detection Performance")]
new.append(body(
    f"Table 4 summarizes detection on the test campaign. Gradual, low-magnitude corruption was hard for every architecture. At the 1% false-positive operating point, the proposed architecture flagged {100*rec('motor','proposed'):.0f}%, {100*rec('reactor','proposed'):.0f}% and {100*rec('robot','proposed'):.0f}% of corrupted windows (motor, reactor, robot), with AUCs of {ci('motor','proposed')}, {ci('reactor','proposed')} and {ci('robot','proposed')}. "
    f"The test false-positive rates ({f(100*float(O1['motor']['proposed']['fpr_test']),1)}%, {f(100*float(O1['reactor']['proposed']['fpr_test']),1)}% and {f(100*float(O1['robot']['proposed']['fpr_test']),1)}%) stayed close to the 1% calibration target, indicating that thresholds transferred from calibration to test. "
    f"The proposed architecture had the highest AUC on the robot, where every baseline was lower (bootstrap intervals of the difference exclude zero except for B7). On the reactor it was statistically indistinguishable from B1, B3 and B5, and on both the motor and the reactor the PCA-residual baseline (B6) achieved a higher AUC ({ci('motor','B6_pca')} and {ci('reactor','B6_pca')}). "
    f"Detection improved with severity and with time since onset; for example, on the motor the proposed AUC rose from 0.58 for low-severity corruption in its first 100 h to 0.79 for high-severity corruption after 1,000 h, while B6 rose from 0.56 to 0.84."))
rows4 = []
for s in SC:
    for a in ["proposed"] + list(SHORT):
        r = O1[s][a]
        d = "" if a == "proposed" else f"{f(r.get('dAUC_vs_proposed'))} [{f(r.get('dAUC_boot_lo'))}, {f(r.get('dAUC_boot_hi'))}]"
        rows4.append([s_(s) if a == "proposed" else "", ARCHN[a], ci(s, a), f(r["recall"], 2), f(r["precision"], 2),
                      f"{f(100*float(r['fpr_test']),1)}%", f(r["false_alarms_per_1000h"], 1), r["events_detected"], d])
new += [table(["Plant", "Architecture", "AUC [95% CI]", "Recall", "Precision", "FPR", "False alarms /1,000 h", "Runs alarmed", "ΔAUC vs proposed [95% CI]"],
              rows4, [760, 1660, 1440, 640, 760, 620, 900, 900, 1680]),
        caption(4, "Integrity detection on the test campaign (1,000 runs per plant). Thresholds at 1% FPR from calibration; 95% CIs by bootstrap over runs; ΔAUC = baseline − proposed. Runs alarmed: corrupted runs with at least one alarm after onset.")]

new.append(h2("4.2 Detection by Mechanism and Attacker Knowledge"))
mrows = []
order = ["N1", "N2", "N3", "N4", "N5", "N6", "A1_naive", "A1_mimic", "A2_naive", "A2_mimic", "A3", "A4_K0", "A4_K1", "A4_K2", "A5"]
for m in order:
    row = [m.replace("_", " ")]
    for s in SC:
        if m not in O5[s]:
            row += ["n/a", "n/a"]; continue
        b = mech_row_best(s, m)
        row += [f(O5[s][m]["proposed"], 2), f"{b[1]:.2f} ({SHORT[b[0]]})"]
    mrows.append(row)
k1 = mech(s:="reactor", "A4_K1", "proposed")
new.append(body(
    f"Table 5 breaks recall down by mechanism. Most natural mechanisms and the gradual attacks (A1, A2) were rarely flagged by any architecture: recall was mostly below 0.10, with the PCA-residual baseline reaching 0.15–0.28 on some of them. The proposed architecture's clearest advantage was on replay (A3): recall {f(mech('reactor','A3','proposed'),2)} on the reactor and {f(mech('robot','A3','proposed'),2)} on the robot, against at most {rep_base:.2f} for any baseline, because replayed data lose their correlation with the live modalities and violate the current physical relations. On the motor, where only the current sensors were replayed, the security path detected replay better ({ARCHN[rep_motor[0]]}: {rep_motor[1]:.2f} vs {f(mech('motor','A3','proposed'),2)}). "
    f"On the reactor it also detected correlation-preserving coordinated attacks (A4 K1, recall {k1:.2f}, baselines ≤ {mech_row_best('reactor','A4_K1')[1]:.2f}), which the physics relations expose even though cross-modal correlations are intact. Attackers with knowledge of the nominal physics (A4 K2) were rarely detected ({f(mech('motor','A4_K2','proposed'),2)}, {f(mech('reactor','A4_K2','proposed'),2)} and {f(mech('robot','A4_K2','proposed'),2)}), confirming that the physics layer offers little protection against an adversary who knows the model. "
    f"Baselines performed better on several single-sensor offsets and natural faults (e.g., B6 on N2 and N5 in the reactor, and B1/B3 voting on single-sensor offsets in the motor), where redundancy and feature-space residuals are more sensitive than the physics relations."))
new += [table(["Mechanism", "Motor: proposed", "Motor: best baseline", "Reactor: proposed", "Reactor: best baseline", "Robot: proposed", "Robot: best baseline"],
              mrows, [1500, 1250, 1400, 1250, 1400, 1250, 1310]),
        caption(5, "Window-level recall by mechanism at the fixed thresholds. Best baseline = the highest recall among B1–B7 (in brackets). N4 and N6 are not applicable to the reactor.")]

new.append(h2("4.3 Physics Constraints and Ablation"))
prow = [[s_(s), r["constraint"].split("_", 1)[1].replace("_", " "), f(r["auc"]), f(r["recall"], 2), f"{f(100*float(r['fpr']),1)}%"] for s in SC for r in O9[s]]
new.append(body(f"Individual physics relations differed widely in usefulness (Table 6). The motor's current–speed relation was the most informative single check (AUC {f(O9['motor'][0]['auc'])}), whereas several relations were close to chance on their own, mostly because legitimate operating-point changes and model mismatch widen their commissioning tolerances. The factorial ablation (Table 7) shows that the best combination differed by plant: on the motor the physics layer carried most of the performance and the learning residual added little, on the reactor the learning residual was the strongest single component, and on the robot the physics layer alone matched the full architecture, the calibration having assigned it all the weight."))
new += [table(["Plant", "Relation", "AUC", "Recall", "FPR"], prow, [1400, 3560, 1500, 1400, 1500]),
        caption(6, "Detection by individual physics relation (each thresholded at 1% FPR on calibration windows).")]
arows = []
for a in ["cm_only", "phys_only", "learn_only", "gate_cm_phys", "proposed"]:
    arows.append([ARCHN[a]] + [f"{f(O10[s][a]['auc'])} [{f(O10[s][a]['auc_lo'])}–{f(O10[s][a]['auc_hi'])}]" for s in SC])
new += [table(["Configuration", "Motor AUC [95% CI]", "Reactor AUC [95% CI]", "Robot AUC [95% CI]"], arows, [3060, 2100, 2100, 2100]),
        caption(7, "Factorial ablation on identical windows. The proposed configuration uses the calibration weights given in Section 3.6.3.")]

new.append(h2("4.4 Operational Cost and Safety"))
orows = [[ARCHN[a]] + sum([[f"{wq(s, a):.1f}%", f(O8[s][a]["unnecessary_quarantine_h_per_1000h"], 1)] for s in SC], [])
         for a in ["proposed", "gate_cm_phys"] + list(SHORT) + ["B0_no_checks"]]
expo = {s: (float(O8[s]['proposed']['safety_exposure_h_per_event_mean']), float(O8[s]['B0_no_checks']['safety_exposure_h_per_event_mean'])) for s in SC}
new.append(body(
    f"The integrity layer has a cost (Table 8). During genuine process faults, the proposed architecture wrongly quarantined {wq('motor','proposed'):.1f}%, {wq('reactor','proposed'):.1f}% and {wq('robot','proposed'):.1f}% of correct windows, against {min(base_wq):.1f}–{max(base_wq):.1f}% for B1–B3. The physics relations cannot separate a genuine change of the process, such as bearing wear raising vibration, from corrupted data; this directly limits the claim that unified validation is safe to deploy without fault-aware models. "
    f"Because recall was low, the hours during which corrupted data were trusted were reduced only slightly relative to no checks (mean per corruption event: motor {expo['motor'][0]:,.0f} h vs {expo['motor'][1]:,.0f} h; reactor {expo['reactor'][0]:,.0f} h vs {expo['reactor'][1]:,.0f} h; robot {expo['robot'][0]:,.0f} h vs {expo['robot'][1]:,.0f} h)."))
new += [table(["Architecture", "Motor WQ", "Motor UQ", "Reactor WQ", "Reactor UQ", "Robot WQ", "Robot UQ"], orows, [2760, 1100, 1100, 1100, 1100, 1100, 1100]),
        caption(8, "Operational cost. WQ: share of correct windows quarantined during genuine process faults. UQ: unnecessary quarantine, hours per 1,000 h of non-corrupted operation.")]

new.append(h2("4.5 Gated Learning"))
lrows = []
for s in SC:
    for r in O12[s]:
        if (r["hours_since_onset_from"], r["to"]) in (("0", "100"), ("500", "1000"), ("2000", "5000")):
            g, u = float(r["mse_gated_median"]), float(r["mse_ungated_median"])
            lrows.append([s_(s), r["origin"], f"{r['hours_since_onset_from']}–{r['to']} h", f"{g:.3g}", f"{u:.3g}", f"{u/g:.2f}", f(r["gate_rejection_rate"], 2)])
new.append(body(
    f"The validation gate protected the soft-sensor model against adversarial corruption on two of the three plants (Table 9). On the motor and the reactor, the ungated learner's error on clean data after an attack was {lr_lo:.1f}–{lr_hi:.1f} times that of the gated learner, and the ratio grew with time since onset. Under natural corruption the differences were smaller: the gated learner was about 1.4 times more accurate on the motor late in the runs, and slightly less accurate on the reactor. On the robot the gate gave no benefit: the gated model was slightly less accurate, as the gate also rejected 10–42% of windows, which slows legitimate adaptation."))
new += [table(["Plant", "Origin", "Since onset", "Error (gated)", "Error (ungated)", "Ungated / gated", "Gate rejection"], lrows, [1100, 1300, 1400, 1400, 1460, 1350, 1350]),
        caption(9, "Median squared error of the soft-sensor model on the uncorrupted readings of the same windows, for runs with natural or adversarial corruption.")]

new.append(h2("4.6 Can Natural and Adversarial Corruption Be Told Apart?"))
trows = []
for s in SC:
    by = {}
    for r in O6[s]:
        by.setdefault(r["pair"], {})["lr" if r["classifier"].startswith("logistic") else "gb"] = r
    for pair, d in by.items():
        lrr, gbr = d.get("lr"), d.get("gb")
        trows.append([s_(s), pair, f"{lrr['n_natural']}/{lrr['n_adversarial']}",
                      f"{f(lrr['accuracy'],2)} [{f(lrr['acc_lo'],2)}–{f(lrr['acc_hi'],2)}]",
                      f"{f(gbr['accuracy'],2)} [{f(gbr['acc_lo'],2)}–{f(gbr['acc_hi'],2)}]" if gbr else "—", f(lrr["chance"], 2)])
new.append(body("Table 10 tests the premise of the paper empirically. Naive attacks were distinguishable from the corresponding natural mechanisms on the motor and the reactor: gradient boosting exceeded the majority-class rate for N1 vs naive A1 and for N3 vs naive A2, and N2 vs A5 was distinguishable on both plants. Attackers that imitated the natural law were not distinguishable: accuracy for N1 vs mimicking A1 and N3 vs mimicking A2 was at or below the majority-class rate on all three plants. On the robot, only the naive noise attack was distinguishable. Attribution from observations is therefore possible against naive adversaries but not against adversaries who imitate natural degradation, which is the case the attribution-independent integrity decision is designed for."))
new += [table(["Plant", "Pair", "Runs (nat./adv.)", "Logistic regression [95% CI]", "Gradient boosting [95% CI]", "Majority-class rate"], trows, [1000, 2100, 1200, 1820, 1820, 1420]),
        caption(10, "Attribution accuracy on matched pairs (5-fold stratified cross-validation over runs; Wilson 95% CIs).")]

new.append(h2("4.7 Computational Cost"))
comps = [("cm", "Cross-modal (Alg. 1)"), ("phys", "Physics (Alg. 2)"), ("learn", "Learning residual (Alg. 3)"), ("sec", "B2 security path"), ("rel", "B3 reliability path"), ("cusum", "B7 CUSUM"), ("learned", "B4–B6 learned detectors")]
crows = [[n] + [f"{float(O13[s][c]['p50_ms']):.3f} / {float(O13[s][c]['p99_ms']):.3f}" for s in SC] for c, n in comps]
p99sum = {s: sum(float(O13[s][c]["p99_ms"]) for c in ("cm", "phys", "learn")) for s in SC}
new.append(body(f"The proposed components are inexpensive (Table 11): the sum of the 99th-percentile times of the three components was below {max(p99sum.values()):.2f} ms per decision on all plants, measured on the hardware in Table 3. The learned baselines (B4–B6 together) took about 3 ms at the median. Rare maxima of tens of milliseconds reflect scheduling on a shared workstation rather than the algorithms."))
new += [table(["Component", "Motor p50 / p99 (ms)", "Reactor p50 / p99 (ms)", "Robot p50 / p99 (ms)"], crows, [3060, 2100, 2100, 2100]),
        caption(11, "Measured compute time per decision.")]

new.append(h2("4.8 Robustness to Model Mismatch"))
mm_auc = lambda m, sc, a: float(MM[m][sc][a]["auc"])
mm_ci = lambda m, sc, a: f"{f(MM[m][sc][a]['auc'])} [{f(MM[m][sc][a]['auc_lo'])}–{f(MM[m][sc][a]['auc_hi'])}]"
mm_fpr = lambda m, sc: 100 * float(MM[m][sc]["proposed"]["fpr_test"])
best0 = {sc: max(SHORT, key=lambda a: mm_auc("0×", sc, a)) for sc in SC}
new.append(body(
    f"To quantify how much the physics-based checks depend on model fidelity, the test design was repeated with 900 additional runs at each of two other mismatch levels: 0× (generator identical to the nominal model, apart from sensor errors) and 2× (twice the default parameter perturbation and unmodelled terms), using the same code version and the same calibration thresholds. "
    f"With an exact nominal model the proposed architecture became the best detector on all three plants (AUC {mm_ci('0×','motor','proposed')}, {mm_ci('0×','reactor','proposed')} and {mm_ci('0×','robot','proposed')}; best baselines {SHORT[best0['motor']]} {f(MM['0×']['motor'][best0['motor']]['auc'])}, {SHORT[best0['reactor']]} {f(MM['0×']['reactor'][best0['reactor']]['auc'])} and {SHORT[best0['robot']]} {f(MM['0×']['robot'][best0['robot']]['auc'])}); the paired run-level bootstrap interval of every baseline-minus-proposed difference excluded zero, the closest being B6 on the motor at {f(MM['0×']['motor']['B6_pca']['dAUC_vs_proposed'])} [{f(MM['0×']['motor']['B6_pca']['dAUC_boot_lo'])}, {f(MM['0×']['motor']['B6_pca']['dAUC_boot_hi'])}], whereas at the default mismatch it fell to the level of the baselines, which changed much less (Table 12). Doubling the mismatch cost little beyond the default level. "
    f"Model fidelity is therefore the main factor limiting the physics-based approach. Thresholds calibrated at 1× did not transfer to other mismatch levels (proposed test FPR on the motor {mm_fpr('0×','motor'):.1f}% at 0× and {mm_fpr('2×','motor'):.1f}% at 2×), so this comparison uses the threshold-free AUC. "
    f"The tight tolerances learned with an exact model also made genuine process faults look like violations: on the motor, wrongful quarantine during bearing wear rose to {100*float(MM8['0×']['motor']['proposed']['wrongful_quarantine_during_process_faults']):.1f}% at 0×, which motivates fault-aware physics relations."))
mrows12 = []
for sc in SC:
    for a in ["proposed", "B1_separated", "B3_reliability", "B6_pca", "B7_cusum"]:
        mrows12.append([SCN[sc] if a == "proposed" else "", ARCHN[a]] + [mm_ci(m, sc, a) for m in ("0×", "1×", "2×")])
new += [table(["Plant", "Architecture", "AUC at 0× mismatch", "AUC at 1× (default)", "AUC at 2× mismatch"], mrows12, [900, 2220, 2080, 2080, 2080]),
        caption(12, "Detection AUC [95% bootstrap CI over runs] by detector-model mismatch. 1×: test campaign (1,000 runs per plant); 0× and 2×: 300 runs per plant each; same code version and thresholds.")]

new.append(h2("4.9 Statistical Notes"))
new.append(body("All intervals are bootstrap intervals over runs. DeLong p-values were close to zero even for AUC differences whose run-level bootstrap interval included zero (e.g., B1 vs proposed on the reactor, p < 0.001 with an interval of −0.012 to 0.032), because the DeLong test treats the millions of correlated windows as independent; inference in this paper therefore rests on the bootstrap intervals only. The final version will use 1,000 bootstrap replicates (200 are used here)."))

new.append(h2("4.10 Discussion"))
new.append(body("The results support three claims. First, an integrity layer built from physics consistency and cross-modal correlation detects attacks that preserve plausible per-sensor values but break relations between modalities: replay and correlation-preserving coordinated attacks, which specialized security and reliability baselines largely missed. Second, gating incremental learning on that integrity decision prevents corrupted data from degrading a learned model, which matters for adaptive maintenance models. Third, natural and adversarial corruption cannot be separated reliably when the adversary imitates natural degradation, so an integrity decision that does not depend on attribution is warranted in that case.", "What the results support. "))
new.append(body("The results do not support a general detection advantage. Window-level detection of gradual, low-magnitude corruption was poor for every architecture, a feature-space PCA residual outperformed the proposed combination on two of three plants, and single-sensor offsets were often better caught by redundancy voting. They also show a real safety cost: the physics layer wrongly quarantined about a tenth of correct data during genuine process faults, several times more than the baselines.", "What the results do not support. "))
new.append(body("The evaluation is simulation-based, with reduced-order plants whose fidelity limits external validity; severity is defined relative to sensor noise, so many corruptions are small relative to normal process variation, which partly explains the low recall; window-level labels count every window from onset, including the early period of gradual mechanisms when the effect is still negligible; performance depends strongly on the fidelity of the nominal model (Section 4.8), and thresholds calibrated at one mismatch level did not transfer to others; the public-data validation (Section 3.6.6) is pending; and the physics relations do not model process faults, which causes the wrongful quarantines reported above.", "Limitations. "))
new.append(body("A deployable design should combine the physics and correlation checks with redundancy voting and a feature-space residual detector, since their strengths are complementary; should model expected process faults explicitly so that genuine degradation is not quarantined; and should use the integrity decision to gate model updates, where the benefit was clearest; and should invest in the fidelity of the nominal plant model, since an exact model made the physics-based architecture the best detector on every plant.", "Practical implications. "))
new.append(body(f"Low recall means that most corrupted windows would be trusted by any of the evaluated architectures at a 1% false-positive budget, and wrongful quarantine during genuine faults can itself remove information operators need. Neither the proposed architecture nor the baselines should be used as the sole safeguard in safety-critical control; certified safety functions must remain independent of them.", "Ethical considerations. "))
new.append(h2("4.11 Reproducibility and Data Availability"))
new.append(body("The simulator, campaign files, calibration thresholds, run manifests with checksums, and all per-window outputs are released with the paper ([repository link and Zenodo DOI: TO ADD]); every table in this section is regenerated from those files by a single command (utmsim report)."))

# ---------------------------------------------------------------- assemble
i36, i5 = idx("3.6 Experimental Design"), idx("5. CONCLUSION")
blocks2 = blocks[:i36] + new + blocks[i5:]
iab = next(i for i, b in enumerate(blocks2) if ptext(b).startswith("Industrial AI systems increasingly make"))
abstract = (f"Industrial AI systems increasingly make safety-critical decisions from sensor streams at the edge, where corruption caused by natural degradation can be observationally similar to adversarial manipulation. Building on control-theoretic results on attack detectability and identifiability, this paper develops an integrity-first architecture for detection without attribution that combines cross-modal consistency checking, physics-informed validation against nominal models, and gated incremental learning. We evaluate it in a pre-specified simulation campaign covering an induction motor, a batch reactor and a robotic cell, eleven natural and adversarial corruption mechanisms and seven baselines (3,000 test runs; {int(O14['Operation windows evaluated'].replace(',', ''))/1e6:.2f} million decision windows; thresholds fixed at a 1% false-positive rate on 900 separate calibration runs). Gradual, low-magnitude corruption was hard for every method: the proposed architecture reached AUCs of {f(O1['motor']['proposed']['auc'],2)}, {f(O1['reactor']['proposed']['auc'],2)} and {f(O1['robot']['proposed']['auc'],2)}, comparable to the best baselines, and a PCA-residual detector was better on two plants. Its advantages were specific: on the reactor and robot it detected replayed data (recall {f(mech('reactor','A3','proposed'),2)} and {f(mech('robot','A3','proposed'),2)} versus at most {rep_base:.2f} for any baseline) and correlation-preserving coordinated attacks ({k1:.2f} on the reactor), and its validation gate kept a learned soft sensor {lr_lo:.1f}–{lr_hi:.1f} times more accurate under attack. It also wrongly quarantined {min(wq(s,'proposed') for s in SC):.1f}–{max(wq(s,'proposed') for s in SC):.1f}% of correct data during genuine process faults. With an exact nominal model it became the best detector on all three plants (AUC {f(MM['0×']['motor']['proposed']['auc'],2)}–{f(MM['0×']['reactor']['proposed']['auc'],2)}), identifying model fidelity as the limiting factor. Natural and adversarial corruption were distinguishable for naive attackers but not for attackers imitating natural degradation.")
blocks2[iab] = "<w:p>" + ppr(blocks2[iab]) + run(abstract, hl(rpr(blocks2[iab]))) + "</w:p>"
def repl_par(prefix, text, lead=None):
    k = next(i for i, b in enumerate(blocks2) if ptext(b).startswith(prefix))
    blocks2[k] = P(blocks2[k], text, lead)
repl_par("Key findings.", f"Across 3,000 pre-specified simulation runs, the unified architecture detected replay and correlation-preserving coordinated attacks that specialized baselines largely missed, and its validation gate protected a learned soft sensor against adversarial corruption on two of three plants. It did not provide a general detection advantage: window-level detection of gradual, low-magnitude corruption was poor for every architecture (AUC {f(O1['motor']['proposed']['auc'],2)}–{f(O1['reactor']['proposed']['auc'],2)} on motor and reactor, {f(O1['robot']['proposed']['auc'],2)} on the robot for the proposed method), a PCA-residual detector performed better on two plants, and the physics layer wrongly quarantined about a tenth of correct data during genuine process faults. Natural and adversarial corruption could be told apart for naive attackers but not for attackers imitating natural degradation.", "Key findings. ")
repl_par("Architectural implications.", "The results support an integrity decision that does not wait for attribution, specifically against adversaries who imitate natural degradation, and support using that decision to gate model learning. They also show that a deployable integrity layer must combine physics and correlation checks with redundancy voting and feature-space residuals, and must model expected process faults so that genuine degradation is not mistaken for corruption.", "Architectural implications. ")
repl_par("Future research directions.", "The next steps are the public-data validation already specified (Section 3.6.6), plant-model identification to close the fidelity gap found in Section 4.8, fault-aware physics relations that distinguish process faults from corrupted data, a combined architecture with redundancy voting and a feature-space residual detector, severity definitions calibrated to normal process variation, and evaluation on a physical testbed.", "Future research directions. ")
iz = next(i for i, b in enumerate(blocks2) if ptext(b).startswith("Industrial AI; Cyber-Physical"))
out_xml = head + "".join(blocks2) + tail
with zipfile.ZipFile(V3) as zi, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        if it.filename in ("word/media/image6.png", "word/media/image7.png"):
            continue                                   # Figures 6 and 7 plotted the old, unsupported numbers
        data = zi.read(it.filename)
        if it.filename == "word/document.xml":
            data = out_xml.encode("utf8")
        if it.filename == "word/_rels/document.xml.rels":
            data = re.sub(rb'<Relationship [^>]*Target="media/image[67]\.png"[^>]*/>', b"", data)
        if it.filename == "[Content_Types].xml":
            pass
        zo.writestr(it, data)
print("written", OUT, "blocks", len(blocks2))
