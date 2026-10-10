# Runbook: running the UTM simulation campaign

**Machine:** one Mac with ≥ 8 cores, ≥ 16 GB RAM and ≥ 10 GB free disk. Your 16-core M3 Max is
ideal: about 1.5–2 h for calibration + test.
**Must be your own machine** (not Apple-issued or Apple-managed). Otherwise use option B at the end.
**Total time:** about 3–4 h, mostly unattended. Do it in one sitting, on mains power.

Every command below runs in Terminal. The folder name contains a space, so keep the quotes.

---

## Step 0: Open the folder

```bash
cd "/Users/parulmann/Documents/PAPERS_KAMAL_PERSONAL/Working /WithAnurag/UnifiedThreatModeling/simulator"
mkdir -p logs
```

## Step 1: Set up Python (once, ~5 min)

```bash
python3 -m venv .venv
source .venv/bin/activate          # the prompt now starts with (.venv); repeat this line in any new Terminal
pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock
python -c "import numpy, sklearn; print('numpy', numpy.__version__, 'scikit-learn', sklearn.__version__)"
```

If `pip install` fails on Python 3.14 (scikit-learn not yet available for it), use 3.12:

```bash
brew install python@3.12
rm -rf .venv && /opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip freeze > requirements.lock
```

## Step 2: Run the tests (~10 s)

```bash
python -m unittest discover -s tests
```

Expect: `Ran 26 tests … OK`. **Stop here if anything fails** and send me the output.

## Step 3: Start version control (once)

Use your personal name and email (not an Apple account):

```bash
git init
git config user.name  "Kamal Mann"
git config user.email "kamalmann@ieee.org"
git add -A
git commit -m "UTM simulator v0.1.0"
```

## Step 4: Smoke test (~1 min)

```bash
python -m utmsim run --campaign campaigns/smoke.toml --workers 4
python -m utmsim calibrate --results results/smoke --out thresholds_smoke.json
python -m utmsim report --results results/smoke --thresholds thresholds_smoke.json --out report_smoke --bootstrap 20
open report_smoke/report.md
```

Check: 12 runs "ok", 0 errors. The report starts with "SMOKE / CALIBRATION OUTPUT, NOT RESULTS", and
the B4/B5 rows have numbers (scikit-learn is working). **These numbers mean nothing**; don't look at
them for conclusions.

## Step 5: Keep the Mac awake

Plug in power. Every long command below is prefixed with `caffeinate -dimsu`, which prevents sleep
while it runs. Don't close the lid unless an external display is connected.

## Step 6: Calibration campaign (900 runs, ~25–35 min)

```bash
caffeinate -dimsu python -m utmsim run --campaign campaigns/calibration.toml --workers 14 2>&1 | tee logs/calibration.log
python -m utmsim verify --campaign campaigns/calibration.toml
python -m utmsim calibrate --results results/calibration --out thresholds.json
```

- `verify` must say `900/900 runs present with valid checksums`.
- `calibrate` prints the chosen gate weights per scenario. Note them; nothing else to do.
- If interrupted (crash, sleep, Ctrl-C), **run the same command again**: finished runs are skipped.
- Any run with status `error` leaves `results/calibration/<scenario>/<run>/error.txt`. Send it to me
  before continuing.

## Step 7: Freeze (1 min, essential)

This proves the settings were fixed before the test campaign ran (requirement G4).

```bash
git add -A
git commit -m "Freeze: code, campaigns and thresholds.json before test campaign"
git log --oneline | head -3
shasum -a 256 thresholds.json campaigns/test.toml | tee logs/freeze_checksums.txt
```

**From here on, change nothing**: no code, no campaign file, no `thresholds.json`. If something has
to change, it means going back to Step 6.

## Step 8: Test campaign (3,000 runs, ~1–1.5 h)

```bash
caffeinate -dimsu python -m utmsim run --campaign campaigns/test.toml --workers 14 2>&1 | tee logs/test.log
python -m utmsim verify --campaign campaigns/test.toml
```

`verify` must say `3000/3000 runs present with valid checksums`. Same resume and error rules as Step 6.

## Step 9: Report (~15–40 min)

Start with 200 bootstrap samples, and run 1,000 for the final paper numbers later (it takes longer):

```bash
caffeinate -dimsu python -m utmsim report --results results/test --thresholds thresholds.json --out report_test --bootstrap 200 2>&1 | tee logs/report.log
open report_test/report.md
```

Final version, can run overnight:

```bash
caffeinate -dimsu python -m utmsim report --results results/test --thresholds thresholds.json --out report_test_B1000 --bootstrap 1000
```

## Step 10: Robustness sweep (optional but recommended, 1,800 runs, ~40–60 min)

```bash
for m in 0 2; do
  caffeinate -dimsu python -m utmsim run --campaign campaigns/mismatch_${m}x.toml --workers 14 2>&1 | tee logs/mismatch_${m}x.log
  python -m utmsim verify --campaign campaigns/mismatch_${m}x.toml
  python -m utmsim report --results results/mismatch_${m}x --thresholds thresholds.json --out report_mismatch_${m}x --bootstrap 200
done
```

## Step 11: Archive and hand over

```bash
git log -1 --format=%H > logs/code_commit.txt
tar czf ../utm_results_$(date +%Y%m%d).tar.gz results logs thresholds.json requirements.lock report_test* report_mismatch_* campaigns
shasum -a 256 ../utm_results_*.tar.gz | tee -a logs/archive_checksum.txt
```

Send me: `report_test/report.md` (plus `report_mismatch_0x/2x/report.md` if you ran Step 10) and
`logs/code_commit.txt`. I'll update §3.6 and §4 of the paper **only** from those reports. Keep the
`.tar.gz` safe: it's the evidence behind every number (and later goes to Zenodo).

## Step 12: v2 campaigns (follow-up A2/A3 + B; about 3 h, mostly unattended)

Run from the **git repo** so every manifest records the frozen commit (not `-dirty`). `thresholds.json`
(v1) must be present and unchanged.

```bash
python -m unittest discover -s tests            # expect: Ran 26 tests ... OK
caffeinate -dimsu python -m utmsim run --campaign campaigns/calibration_v2.toml --workers 14 2>&1 | tee logs/calibration_v2.log
python -m utmsim verify --campaign campaigns/calibration_v2.toml
python -m utmsim calibrate --results results/calibration_v2 --base thresholds.json --out thresholds_v2.json 2>&1 | tee logs/calibrate_v2.log
```

`calibrate` prints the chosen v2 weights and EWMA λ per plant. **Then freeze:**

```bash
git add -A && git commit -m "Freeze v2: code, campaigns, thresholds_v2.json before test_v2"
git tag -a v2-freeze -m "v2 frozen before test_v2"
shasum -a 256 thresholds.json thresholds_v2.json campaigns/test_v2.toml campaigns/mismatch_v2_*.toml | tee logs/freeze_v2_checksums.txt
```

From here on, change nothing. Then test and report:

```bash
for c in test_v2 mismatch_v2_0x mismatch_v2_2x; do
  caffeinate -dimsu python -m utmsim run --campaign campaigns/$c.toml --workers 14 2>&1 | tee logs/$c.log
  python -m utmsim verify --campaign campaigns/$c.toml
  caffeinate -dimsu python -m utmsim report --results results/$c --thresholds thresholds_v2.json --out report_$c --bootstrap 200 2>&1 | tee logs/report_$c.log
done
```

Commit the results (`git add -f results/*_v2* logs report_*v2*`) and tag `v2-campaign`.



| Symptom | Fix |
|---|---|
| `command not found: python` | Run `source .venv/bin/activate` again (new Terminal window) |
| Mac got slow, fans loud | Normal. Use `--workers 10` to leave more headroom; it only changes speed, not results |
| Run stopped halfway | Re-run the same command; completed runs are skipped |
| `verify` reports missing runs | Re-run the `run` command for that campaign, then `verify` again |
| Disk full | Results take about 2–4 GB. Free space, then re-run |
| B4/B5 rows say "not run" | scikit-learn isn't installed in the venv: Step 1 |

---

## Option B: not your own Mac, or you want it faster

**Single cloud VM (simplest).** Rent one compute-optimized VM with 32 vCPUs (e.g. AWS c7i.8xlarge or
GCP c3-standard-44) on **your personal account**; about 1 h, a few dollars. Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv git
# copy the simulator folder to the VM (scp or git), then Steps 1–11 with --workers 30
# no caffeinate on Linux: run inside `tmux` so it survives disconnects
```

**Several machines.** Same code and same `thresholds.json` on each. Machine *i* of *N* runs:

```bash
python -m utmsim run --campaign campaigns/test.toml --shard i/N --workers <cores-2>
```

Then copy every machine's `results/test/` into one folder (`rsync -a`), and run `verify` and
`report` once on the merged folder. Calibration must be finished and frozen **before** any machine
starts the test shards.
