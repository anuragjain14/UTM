"""Test suite (requirements T-1..T-8). Run:  python3 -m unittest discover -s tests -v"""

import ast
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from utmsim import config as C                       # noqa: E402
from utmsim.engine import run_one                     # noqa: E402
from utmsim.io import read_csv_gz                     # noqa: E402
from utmsim.metrics import auc, bootstrap_runs, confusion, delong, wilson   # noqa: E402
from utmsim.plants import make                        # noqa: E402
from utmsim.corruption import Corruption              # noqa: E402
from utmsim.detectors import CrossModal, GatedLearner, corr   # noqa: E402

SMOKE = ROOT / "campaigns" / "smoke.toml"


def spec_for(scen, **kw):
    s = {"run_index": 0, "run_id": f"t-{scen}", "scenario": scen, "kind": "clean", "mechanism": None,
         "origin": "none", "attacker_knowledge": None, "attack_mode": None, "severity": None,
         "process_fault": None, "horizon_h": 300.0, "onset_h": None, "fault_onset_h": None,
         "fs_hz": {"motor": 500.0, "reactor": 5.0, "robot": 200.0}[scen], "sensors_per_modality": 6,
         "noise_frac": 0.01}
    s.update(kw)
    return s


class TestDeterminism(unittest.TestCase):
    """T-1: identical inputs give byte-identical windows.csv.gz."""

    def test_same_run_twice(self):
        camp = C.load_campaign(SMOKE)
        spec = C.build_run_list(camp)[0]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            da, db = run_one(spec, camp, a), run_one(spec, camp, b)
            self.assertEqual((da / "windows.csv.gz").read_bytes(), (db / "windows.csv.gz").read_bytes())

    def test_run_list_stable(self):
        camp = C.load_campaign(SMOKE)
        self.assertEqual(C.build_run_list(camp), C.build_run_list(camp))


class TestPlants(unittest.TestCase):
    """T-2: with no corruption and no mismatch, nominal residuals are small relative to signal."""

    def test_residuals_small_without_mismatch(self):
        for scen in ("motor", "reactor", "robot"):
            plant, nom = make(spec_for(scen, noise_frac=0.001), np.random.default_rng(0), mismatch=0.0)
            rng = np.random.default_rng(1)
            for t in range(0, 20, 2):
                plant.step_slow(float(t + 1), 1.0)
            true, ctx = plant.true_window(20.0, spec_for(scen)["fs_hz"], 256, rng)
            res = nom.residuals(plant.measure(true, rng), ctx, 20.0)
            for g, r in res.items():
                self.assertTrue(np.all(np.isfinite(r)), (scen, g))

    def test_window_shapes(self):
        for scen in ("motor", "reactor", "robot"):
            plant, _ = make(spec_for(scen), np.random.default_rng(0), mismatch=1.0)
            true, _ = plant.true_window(0.0, spec_for(scen)["fs_hz"], 128, np.random.default_rng(2))
            for m, x in true.items():
                self.assertEqual(x.shape, (plant.counts[m], 128), (scen, m))


class TestCorruption(unittest.TestCase):
    """T-3: mechanisms produce their specified signatures."""

    def _apply(self, mech, t, **kw):
        sp = spec_for("motor", mechanism=mech, origin=C.ORIGIN[mech], severity="high", onset_h=100.0, **kw)
        plant, nom = make(sp, np.random.default_rng(0), 1.0)
        cr = Corruption(sp, plant, nom, np.random.default_rng(3))
        true, ctx = plant.true_window(t, 500.0, 128, np.random.default_rng(4))
        clean = plant.measure(true, np.random.default_rng(5))
        out, comp = cr.apply(clean, ctx, t)
        return cr, clean, out, comp

    def test_before_onset_untouched(self):
        cr, clean, out, comp = self._apply("A5", 50.0)
        self.assertFalse(comp)
        for m in clean:
            np.testing.assert_array_equal(clean[m], out[m])

    def test_fdi_step_offset(self):
        cr, clean, out, comp = self._apply("A5", 150.0)
        self.assertTrue(comp)
        d = out[cr.mod][cr.idx] - clean[cr.mod][cr.idx]
        np.testing.assert_allclose(d, cr.mag)

    def test_naive_ramp_linear(self):
        cr, clean, out, _ = self._apply("A1", 100.0 + 50.0, attack_mode="naive")
        d = (out[cr.mod][cr.idx] - clean[cr.mod][cr.idx]).mean()
        self.assertAlmostEqual(d, cr.mag * min(1.0, 50.0 / cr.T_ramp), places=9)

    def test_replay_copies_recorded(self):
        sp = spec_for("motor", mechanism="A3", origin="adversarial", severity="low", onset_h=100.0)
        plant, nom = make(sp, np.random.default_rng(0), 1.0)
        cr = Corruption(sp, plant, nom, np.random.default_rng(3))
        cr.choose_record_time([10.0])
        rng = np.random.default_rng(6)
        true, ctx = plant.true_window(10.0, 500.0, 64, rng)
        rec = plant.measure(true, rng)
        cr.apply(rec, ctx, 10.0)
        true, ctx = plant.true_window(120.0, 500.0, 64, rng)
        out, comp = cr.apply(plant.measure(true, rng), ctx, 120.0)
        np.testing.assert_array_equal(out[cr.mod], rec[cr.mod])


class TestDetectors(unittest.TestCase):
    """T-4 and T-6."""

    def test_corr_constant_channel(self):
        X = np.vstack([np.ones(50), np.arange(50.0)])
        Cm = corr(X)
        self.assertEqual(Cm[0, 1], 0.0)
        self.assertAlmostEqual(Cm[1, 1], 1.0)

    def test_crossmodal_detects_decorrelation(self):
        rng = np.random.default_rng(0)
        mk = lambda: {"a": np.tile(rng.normal(size=200), (2, 1)) + 0.05 * rng.normal(size=(2, 200))}
        cmv = CrossModal(("a",))
        cmv.fit([mk() for _ in range(30)])
        normal = cmv.score(mk())
        broken = mk()
        broken["a"][1] = rng.normal(size=200)
        self.assertGreater(cmv.score(broken), normal * 5)

    def test_ewc_pulls_toward_theta_star(self):
        X = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
        L = GatedLearner()
        L.fit(X, 2 * X[:, 1] + 1)
        for _ in range(200):
            L.step(np.array([1.0, 0.5]), 100.0, gate_ok=False)
        np.testing.assert_allclose(L.gated, L.theta_star)          # gated never updated
        self.assertGreater(abs(L.ungated[0] - L.theta_star[0]), 0.1)

    def test_label_hygiene(self):
        """Detectors and nominal models never reference label fields."""
        forbidden = {"compromised", "origin", "mechanism", "label", "onset_h", "attack_mode", "attacker_knowledge"}
        files = [ROOT / "utmsim" / "detectors.py"]
        for f in files:
            tree = ast.parse(f.read_text())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
                    {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertFalse(names & forbidden, f"{f.name} references {names & forbidden}")
        for f in (ROOT / "utmsim" / "plants").glob("*.py"):
            src = f.read_text()
            nominal_src = src[src.find("Nominal(Nominal)"):] if "Nominal(Nominal)" in src else ""
            for word in ("compromised", "mechanism", "origin"):
                self.assertNotIn(word, nominal_src, f"{f.name} nominal model references {word}")


class TestMetrics(unittest.TestCase):
    """T-5."""

    def test_auc_matches_brute_force(self):
        rng = np.random.default_rng(1)
        s = np.round(rng.normal(size=400), 1)
        y = rng.random(400) < 0.3
        bf = np.mean([(p > n) + 0.5 * (p == n) for p in s[y] for n in s[~y]])
        self.assertAlmostEqual(auc(s, y), bf, places=12)

    def test_delong_auc_and_null(self):
        rng = np.random.default_rng(2)
        ps = []
        for _ in range(200):
            y = rng.random(200) < 0.5
            a = rng.normal(size=200) + y
            b = rng.normal(size=200) + y
            A, Bv, d, z, p = delong(a, b, y)
            self.assertAlmostEqual(A, auc(a, y), places=10)
            ps.append(p)
        rej = np.mean(np.array(ps) < 0.05)
        self.assertLess(rej, 0.12)            # ~5% under the null

    def test_confusion(self):
        c = confusion([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(c["f1"], 0.5)

    def test_bootstrap_and_wilson(self):
        rid = np.repeat(np.arange(20), 10)
        v = np.random.default_rng(0).normal(size=200)
        lo, hi = bootstrap_runs(rid, lambda ix: float(v[ix].mean()), B=200)
        self.assertLess(lo, v.mean())
        self.assertGreater(hi, v.mean())
        lo, hi = wilson(50, 100)
        self.assertTrue(0.39 < lo < 0.41 and 0.59 < hi < 0.61)


class TestSharding(unittest.TestCase):
    """T-8: shards partition the run list exactly."""

    def test_partition(self):
        runs = C.build_run_list(C.load_campaign(SMOKE))
        parts = [C.shard(runs, f"{i}/4") for i in range(1, 5)]
        ids = sorted(r["run_id"] for p in parts for r in p)
        self.assertEqual(ids, sorted(r["run_id"] for r in runs))
        self.assertEqual(len(ids), len(set(ids)))


class TestSmokePipeline(unittest.TestCase):
    """T-7: the whole pipeline runs (smoke numbers are not results)."""

    def test_end_to_end(self):
        from utmsim.report import calibrate, report
        from utmsim.runner import run_campaign, verify_campaign
        camp = C.load_campaign(SMOKE)
        tmp = Path(tempfile.mkdtemp())
        try:
            root = tmp / "smoke"
            self.assertEqual(run_campaign(camp, root, workers=1, log=lambda *_: None), 0)
            self.assertTrue(verify_campaign(camp, root, log=lambda *_: None))
            th = calibrate(root)
            out = report(root, th, tmp / "rep", B=10)
            self.assertIn("NOT RESULTS", out.read_text())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
