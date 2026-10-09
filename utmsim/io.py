"""Output files, checksums and manifests (requirements X-5, X-10, T-1)."""

import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else f"{float(v):.6g}"
    return str(v)


def write_csv_gz(path, header, rows):
    """Deterministic gzip (mtime=0) so identical runs give byte-identical files."""
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join(fmt(r.get(h)) for h in header) + "\n")
    data = buf.getvalue().encode("utf-8")
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
            gz.write(data)


def read_csv_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        cols = {h: [] for h in header}
        for line in f:
            for h, v in zip(header, line.rstrip("\n").split(",")):
                cols[h].append(v)
    return cols


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_dir):
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else "not-a-git-repo"
    except Exception:
        return "unknown"


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def environment():
    try:
        import sklearn
        skl = sklearn.__version__
    except Exception:
        skl = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": skl,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }


def manifest(run_dir, files, extra):
    m = dict(extra)
    m["files"] = {Path(f).name: sha256(f) for f in files}
    write_json(Path(run_dir) / "manifest.json", m)
    return m


def verify_run(run_dir):
    """True if summary + manifest exist and every listed file matches its checksum (X-4)."""
    run_dir = Path(run_dir)
    mpath = run_dir / "manifest.json"
    if not mpath.exists() or not (run_dir / "summary.json").exists():
        return False
    try:
        files = json.loads(mpath.read_text())["files"]
    except Exception:
        return False
    return all((run_dir / name).exists() and sha256(run_dir / name) == h for name, h in files.items())
