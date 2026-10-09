# Models, mechanisms and design decisions

This file is the source text for the rewritten manuscript §3.6.1–§3.6.5. Every item is implemented
in `utmsim/` as described here.

## 1. Time structure (multi-rate; requirements MR-1..MR-5)

- Slow state on a 1-h grid over the run horizon (1,000–10,000 h).
- One decision **window** every **2 h**: 512 samples at the run's sampling rate (motor 200–1,000 Hz,
  reactor 1–10 Hz, robot 100–500 Hz). Window length therefore differs by plant: ~0.5–2.6 s for the
  motor, ~51–512 s for the reactor, ~1–5 s for the robot.
- **Commissioning:** the first 192 h (96 windows) are always clean. The first 2/3 fit every detector;
  the last 1/3 provide each detector's robust normalization (median, MAD), so every score becomes a
  z-score. Hence corruption onsets start at **196 h** (Table 2 said 100 h; the paper must say this).

## 2. Plants (generator side) and nominal models (detector side)

Each run perturbs the generator's parameters by `mismatch × U(−10%, +10%)` (robot ±5%) and adds
unmodelled terms. Detectors only know the nominal equations (requirement AC-1).

| Plant | Generator | Unmodelled terms | Nominal relations (Algorithm 2) |
|---|---|---|---|
| Motor (50 HP) | load schedule (changes ~daily) + fast load fluctuation; slip-speed; current law I(L); vibration ∝ (ω/ω_r)²(1+κ_L L)·f(bearing health); acoustic = a0 + c_a·v; first-order thermal model with ambient daily cycle | current-law exponent γ ≠ 1; thermal–vibration coupling; sensor-location factors | R1 current vs speed-inferred load; R2 acoustic vs vibration; R3 temperature vs nominal thermal model driven by measured current; R4 vibration vs speed/load |
| Batch reactor | 10-h cycle (fill, heat, react, drain); temperature control by coolant flow; level from feed/drain; pressure from temperature and level | vapour-pressure curvature | R1 level slope vs feed/drain (mass balance); R2 pressure vs temperature/level; R3 coolant flow vs controller law |
| Robot cell | 6-DOF periodic pick-and-place; program and payload change every ~12 h; gravity + inertia + friction torques; force sensor sees payload when holding plus insertion contacts | joint offsets; viscous friction | R1 vision vs forward kinematics of encoders; R2 joint currents vs nominal dynamics; R3 wrist force vs known payload/gripper state |

**Sensors** (`sensors.py`): per-sensor gain and offset error, AR(1) coloured noise, small quadratic
nonlinearity. **Noise level = noise_frac (0.1–10%, log-uniform) × the modality's measurement span**
(e.g. motor speed span 30 rpm, robot encoder span 0.05 rad). Using the span, not the absolute level,
avoids physically implausible noise (±18 rpm on a speed encoder at 1% of 1,800 rpm).

**Process faults** (data stay correct, the process changes; requirement §3.3 / Q1): P1 bearing wear
(motor), P2 heat-transfer fouling (reactor), P3 joint friction increase (robot). They are **not**
integrity positives; output O8 reports how often correct data are wrongly quarantined during them.

## 3. Corruption mechanisms (`corruption.py`)

Severity S ∈ {1, 3, 6} (low, medium, high). Magnitude unit = 2·S·max(sensor σ, 0.5% of span).

| ID | Law |
|---|---|
| N1 drift | m·(1 − e^(−Δt/τ))·(1 + c·sin(daily)), τ ~ U(300, 2000) h: saturating, temperature-modulated |
| N2 calibration | gain step ±2·S% + offset step m |
| N3 noise growth | additive noise, σ grows linearly to |m| over 3τ |
| N4 mounting degradation | gain loss up to 8·S% + added noise (motor vibration, robot force) |
| N5 stuck-at | each window after onset, stuck at last median with p = 0.1 + 0.3·S/6 (compromised only when stuck) |
| N6 encoder wear | quantization + backlash noise growing over τ (motor speed, robot encoders) |
| A1 gradual bias | **naive:** linear ramp to m over U(200, 2000) h; **mimic:** identical law to N1 |
| A2 noise amplification | **naive:** abrupt ×(1 + S/2) of within-window variation; **mimic:** identical law to N3 |
| A3 replay | the whole target modality replaced by a window recorded during commissioning |
| A4 coordinated | **K0:** offset on 1–2 sensors; **K1:** same offset on all sensors of the modality (correlations kept); **K2:** fake physical change propagated through the NOMINAL model to every related modality. Ramps in over 48 h |
| A5 FDI step | pure offset step m |

Matched pairs for E-ATTR: N1↔A1 (naive, mimic), N2↔A5, N3↔A2 (naive, mimic). Natural and adversarial
mechanisms have their own laws; `mimic` attackers copy the natural law on purpose. *(Changed from
requirement MP-1, which drew both from the same distribution and would have made the attribution
test circular.)*

## 4. Detectors and architectures (`detectors.py`, `report.py`)

| Name | Definition |
|---|---|
| Cross-modal (Alg. 1) | mean over channel pairs of \|C_window − C_baseline\| / σ_baseline (paper: sum; mean is a monotone rescaling) |
| Physics (Alg. 2) | per relation group: max over entries of \|r − c\|/s (c, s = commissioning median and robust std); total = mean over groups |
| Learn (Alg. 3) | soft-sensor linear model (motor: current from speed, vibration, acoustic; reactor: pressure from temperature, level; robot: joint-2 RMS current from configuration and gripper state); normalized-LMS updates with EWC penalty (λ = 1); gated copy updates only if mean(z_cm, z_phys) < 3 (rule fixed a priori); ungated copy always updates |
| **Proposed** | w_cm·z_cm + w_phys·z_phys + w_learn·z_learn, weights from a 15-point grid maximizing AUC on **calibration** runs |
| B1 separated | max(B2, B3): security and reliability paths in parallel |
| B2 security | max(step z between consecutive windows, replay similarity to stored commissioning windows) |
| B3 reliability | max(redundancy-vote deviation, Page CUSUM of vote deviation) |
| B4 / B5 | Isolation Forest / one-class SVM on window features (scikit-learn) |
| B6 | PCA residual (Q statistic, 90% variance) on window features |
| B7 | classical per-sensor Page CUSUM on standardized medians (k = 0.5, h = 5, reset after alarm) |

Every architecture's threshold is set at **1% FPR on calibration windows** (same rule for all) and
then frozen (AC-3, G4).

## 5. Statistics

AUC (Mann–Whitney); 95% CIs by bootstrap **over runs**; DeLong test for paired AUC differences
(reported alongside the run-level bootstrap, which is primary because windows within a run are
correlated); Wilson CIs for attribution accuracy. No p-values over Monte Carlo runs as evidence of
generalization; no Cohen's d.

## 6. Open-question defaults adopted (requirements §12)

Q1 process faults are not integrity positives · Q2 1,000 test + 300 calibration runs per scenario ·
Q3 baselines as above · Q4 1% FPR on calibration · Q5 measured compute time, no control-loop latency ·
Q6 availability impact and safety exposure proxies (O8) · Q7 linear soft sensor with exact EWC ·
Q8 no matplotlib (figure data as CSV) · Q9 code in `UnifiedThreatModeling/simulator/` ·
Q10 own machines or cloud · Q11 K0–K2.
