import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from bench.protocol import ROOT, make_job, write_json


class CliTests(unittest.TestCase):
    def call(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_training_seed_is_the_aggregation_unit_and_missing_runs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.jsonl"
            jobs = [make_job("matched", "cheetah-run", m, s) for m in
                    ("ct_sac_corrected", "tpht") for s in (0, 1)]
            plan.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            for j in jobs:
                rows = [dict(episode_seed=100_000 + i, time_score=10*j["seed"] +
                             (3 if j["algo"] == "tpht" else 0)) for i in range(20)]
                write_json(root / j["run_id"] / "evaluations/common_final_test_deterministic_irregular_nominal_n20.json",
                           dict(job=j, rows=rows, train_counters={}))
            out = root / "result.json"
            result = self.call("aggregate.py", "--plans", plan, "--root", root,
                               "--clocks", "irregular", "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            reports = json.loads(out.read_text())["results"]
            ours = next(v for v in reports if v["algo"] == "tpht")
            self.assertEqual(ours["training_seed_count"], 2)
            self.assertEqual(ours["paired_difference"]["mean"], 3.)
            missing = root / jobs[-1]["run_id"] / "evaluations/common_final_test_deterministic_irregular_nominal_n20.json"
            missing.unlink()
            result = self.call("aggregate.py", "--plans", plan, "--root", root,
                               "--clocks", "irregular", "--out", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing", result.stderr)

    def test_selection_uses_validation_and_disjoint_training_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [make_job("matched", "cheetah-run", m, s, settings={"learning_rate": lr})
                    for m in ("ct_picard", "tpht") for lr in (.00015, .0006) for s in (100, 101)]
            plan = root / "tuning.jsonl"
            plan.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            for j in jobs:
                score = 50 if j["settings"]["learning_rate"] == .0006 else 10
                rows = [dict(episode_seed=50_000+i, time_score=score) for i in range(10)]
                write_json(root / j["run_id"] / "evaluations/common_final_validation_deterministic_irregular_nominal_n10.json",
                           dict(job=j, split="validation", rows=rows))
            out = root / "test.jsonl"
            result = self.call("select_settings.py", "--plans", plan, "--root", root,
                               "--test-seeds", "0,1", "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            selected = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(selected), 4)
            self.assertTrue(all(j["settings"]["learning_rate"] == .0006 for j in selected))
            result = self.call("select_settings.py", "--plans", plan, "--root", root,
                               "--test-seeds", "100", "--out", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("disjoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
