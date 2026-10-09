# UTM simulator

Simulation framework for *Unified Threat Modeling for Industrial AI Systems*. It produces, by
actually running, every simulation number in the paper (§3.6, §4). Specification:
`../KM_working_/simulator_requirements.md`. Output plan: `../KM_working_/simulation_outputs_spec.md`.

> **Rules (requirements G1–G7).** Every reported number must come from `utmsim report` on the frozen
> test campaign. The paper changes to match the results, never the reverse. Don't tune anything
> after seeing test results. Smoke and calibration reports are stamped *NOT RESULTS*.

## 1. Setup (once per machine)

```bash
cd simulator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock            # exact versions, kept with the results
python -m unittest discover -s tests -v   # 18 tests, ~10 s
git init && git add -A && git commit -m "UTM simulator"   # so manifests record a commit
```

Core needs only **numpy**. **scikit-learn** adds baselines B4 (Isolation Forest), B5 (one-class SVM)
and the gradient-boosting attribution classifier; without it those rows show "not run".

## 2. Run order

```bash
# a) smoke test: whole pipeline in seconds (numbers are NOT results)
python -m utmsim run --campaign campaigns/smoke.toml --workers 4
python -m utmsim calibrate --results results/smoke --out thresholds_smoke.json
python -m utmsim report --results results/smoke --thresholds thresholds_smoke.json --out report_smoke

# b) calibration: 900 runs; chooses gate weights + thresholds at 1% FPR
python -m utmsim run --campaign campaigns/calibration.toml --workers 14
python -m utmsim verify --campaign campaigns/calibration.toml
python -m utmsim calibrate --results results/calibration --out thresholds.json

# c) FREEZE: commit code + campaigns + thresholds.json before the test campaign (G4)
git add -A && git commit -m "Freeze for test campaign"

# d) test: 3,000 runs
python -m utmsim run --campaign campaigns/test.toml --workers 14
python -m utmsim verify --campaign campaigns/test.toml
python -m utmsim report --results results/test --thresholds thresholds.json --out report_test --bootstrap 1000

# e) robustness: model-mismatch sweep (same thresholds)
for m in 0 2; do python -m utmsim run --campaign campaigns/mismatch_${m}x.toml --workers 14
  python -m utmsim report --results results/mismatch_${m}x --thresholds thresholds.json --out report_mismatch_${m}x; done
```

Cost measured on an Apple M3 Max: worst-case run (10,000 h, 1,000 Hz, 16 sensors/modality) about 40 s,
≤ 160 MB; average about 20 s. So calibration + test is about 20 core-hours, roughly 1.5 h on 14 workers.

## 3. Several machines or cloud

Runs are independent and deterministic. Any machine can run any shard; results merge by copying folders.

```bash
# machine i of N (1-based)
python -m utmsim run --campaign campaigns/test.toml --shard i/N --workers 8
# then copy results/test/ from every machine into one folder (rsync, aws s3 sync, gsutil rsync)
python -m utmsim verify --campaign campaigns/test.toml     # all runs present, checksums OK
```

- **Resume:** re-running a shard skips runs whose output verifies, so spot/preemptible instances are
  safe to restart. Use `--force` only to deliberately recompute.
- **Container:** `docker build -t utmsim .`; for an array job use
  `run --campaign campaigns/test.toml --shard ${INDEX}/${COUNT}` (AWS Batch: `AWS_BATCH_JOB_ARRAY_INDEX + 1`).
- Use **your own** machines or cloud account (requirement G6).

## 4. Outputs

`results/<campaign>/<scenario>/<run_id>/`:

| File | Content |
|---|---|
| `windows.csv.gz` | one row per 2-h operation window: labels (evaluation only), z-scores of every detector, learner errors, window features. Byte-identical on re-run |
| `timing.csv.gz` | compute time per component per window (ns) |
| `summary.json` | run spec, seeds, counts, corruption details |
| `manifest.json` | code commit, environment (CPU, Python, numpy, scikit-learn), wall time, peak memory, sha256 of every file |

`utmsim report` writes `report.md` + one CSV per output (O1, O5–O10, O12–O14); see
`simulation_outputs_spec.md` for what each one feeds in the paper.

## 5. What is simulated

See `docs/models.md`: plant equations, sensor model, mechanisms, detectors, baselines and the
decisions taken where the paper was silent.
