"""Campaign runner: sharding, local worker processes, resume (requirements X-3, X-4, X-6, X-9)."""

import multiprocessing as mp
import time
import traceback
from pathlib import Path

from .config import build_run_list, shard
from .engine import run_one
from .io import verify_run


def _run_dir(out_root, spec):
    return Path(out_root) / spec["scenario"] / spec["run_id"]


def _job(args):
    spec, camp, out_root = args
    t0 = time.time()
    try:
        run_one(spec, camp, out_root)
        return spec["run_id"], "ok", time.time() - t0, None
    except Exception:
        d = _run_dir(out_root, spec)
        d.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (d / "error.txt").write_text(tb)
        return spec["run_id"], "error", time.time() - t0, tb.strip().splitlines()[-1]


def run_campaign(camp, out_root, shard_spec="1/1", workers=1, force=False, limit=None, log=print):
    runs = shard(build_run_list(camp), shard_spec)
    if limit:
        runs = runs[:int(limit)]
    todo = [r for r in runs if force or not verify_run(_run_dir(out_root, r))]
    log(f"[campaign {camp['campaign']['name']}] shard {shard_spec}: {len(runs)} runs, "
        f"{len(runs) - len(todo)} already complete, {len(todo)} to run, {workers} worker(s)")
    jobs = [(r, camp, str(out_root)) for r in todo]
    n_err = 0
    t0 = time.time()
    if workers <= 1:
        it = map(_job, jobs)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, maxtasksperchild=50)
        it = pool.imap_unordered(_job, jobs)
    for k, (run_id, status, secs, err) in enumerate(it, 1):
        n_err += status != "ok"
        log(f"  [{k}/{len(jobs)}] {run_id} {status} {secs:.1f}s" + (f"  {err}" if err else ""))
    if workers > 1:
        pool.close()
        pool.join()
    log(f"[campaign {camp['campaign']['name']}] shard {shard_spec} done in {time.time() - t0:.0f}s, "
        f"{n_err} error(s)")
    return n_err


def verify_campaign(camp, out_root, log=print):
    runs = build_run_list(camp)
    missing = [r["run_id"] for r in runs if not verify_run(_run_dir(out_root, r))]
    log(f"{len(runs) - len(missing)}/{len(runs)} runs present with valid checksums")
    for m in missing[:20]:
        log(f"  missing or invalid: {m}")
    return not missing
