"""Command line: utmsim run | calibrate | report | verify | list (requirement X-7)."""

import argparse
import json
import sys
from pathlib import Path

from .config import build_run_list, load_campaign
from .io import write_json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="utmsim", description="UTM simulator (see README.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run (a shard of) a campaign")
    r.add_argument("--campaign", required=True)
    r.add_argument("--out", default="results")
    r.add_argument("--shard", default="1/1", help="i/N: run every N-th run starting at i (1-based)")
    r.add_argument("--workers", type=int, default=1, help="local worker processes")
    r.add_argument("--force", action="store_true", help="re-run even if a valid result exists")
    r.add_argument("--limit", type=int, default=None, help="only the first N runs of this shard")

    c = sub.add_parser("calibrate", help="thresholds + gate weights from a CALIBRATION campaign")
    c.add_argument("--results", required=True, help="results/<calibration campaign>")
    c.add_argument("--out", default="thresholds.json")
    c.add_argument("--fpr", type=float, default=0.01)
    c.add_argument("--base", default=None, help="frozen thresholds.json: keep its v1 entries, add v2 only")

    p = sub.add_parser("report", help="paper tables from a TEST campaign")
    p.add_argument("--results", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--out", default="report")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--no-attribution", action="store_true")

    v = sub.add_parser("verify", help="check every run of a campaign is present with valid checksums")
    v.add_argument("--campaign", required=True)
    v.add_argument("--out", default="results")

    l = sub.add_parser("list", help="print the run list of a campaign")
    l.add_argument("--campaign", required=True)

    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .runner import run_campaign
        camp = load_campaign(a.campaign)
        root = Path(a.out) / camp["campaign"]["name"]
        n_err = run_campaign(camp, root, a.shard, a.workers, a.force, a.limit)
        return 1 if n_err else 0
    if a.cmd == "calibrate":
        from .report import calibrate
        base = json.loads(Path(a.base).read_text()) if a.base else None
        th = calibrate(a.results, a.fpr, base)
        write_json(a.out, th)
        print(f"wrote {a.out}: " + ", ".join(f"{s}: weights {v['weights']['proposed']}" for s, v in th["scenarios"].items()))
        for s, v in th["scenarios"].items():
            for n, cfg in v.get("v2", {}).items():
                print(f"  {s} {n}: inputs {cfg['inputs']} weights {cfg['w']} lambda {cfg['lambda']:g} (calibration AUC {cfg['calibration_auc']:.3f})")
        return 0
    if a.cmd == "report":
        from .report import report
        th = json.loads(Path(a.thresholds).read_text())
        out = report(a.results, th, a.out, B=a.bootstrap, attribution=not a.no_attribution)
        print(f"wrote {out}")
        return 0
    if a.cmd == "verify":
        from .runner import verify_campaign
        camp = load_campaign(a.campaign)
        return 0 if verify_campaign(camp, Path(a.out) / camp["campaign"]["name"]) else 1
    if a.cmd == "list":
        for spec in build_run_list(load_campaign(a.campaign)):
            print(json.dumps(spec))
        return 0


if __name__ == "__main__":
    sys.exit(main())
